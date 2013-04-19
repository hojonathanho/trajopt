#include "scene_objectives.hpp"

#include "utils/eigen_conversions.hpp"
#include "utils/interpolation.hpp"
#include "rave_utils.hpp"
#include "sco/modeling_utils.hpp"
#include "sco/sco_common.hpp"
#include "sco/expr_ops.hpp"
#include "sco/expr_vec_ops.hpp"
#include "trajopt/rave_utils.hpp"

namespace trajopt {

/*
btQuaternion toBtQuat(const OR::Vector& q) {
  return btQuaternion(q[1], q[2], q[3], q[0]);
}
btTransform toBt(const OR::Transform& t){
  return btTransform(toBtQuat(t.rot), toBt(t.trans));
}*/

Vector3d toVector3d(const btVector3 &v) {
  return Vector3d(v.x(), v.y(), v.z());
}

Vector4d toVector4d(const btQuaternion &q) {
  // use OpenRAVE's convention
  return Vector4d(q.w(), q.x(), q.y(), q.z());
}

Vector toOR(const Vector3d &v) {
  return OR::Vector(v(0), v(1), v(2));
}

Vector toORQuat(const Vector4d& v) {
  return OR::Vector(v(0), v(1), v(2), v(3));
}

Vector toOR(const btVector3& v) {
  return OR::Vector(v.x(), v.y(), v.z());
}

string toStr(const btVector3 &v) {
  return (boost::format("[%d %d %d]") % v.x() % v.y() % v.z()).str();
}

string toStr(KinBody::LinkPtr l) {
  return (boost::format("%s/%s") % l->GetParent()->GetName() % l->GetName()).str();
}

Matrix3d toRotationMatrix(const Vector4d &q) {
  return Quaterniond(q(0), q(1), q(2), q(3)).matrix();
}

OR::Transform ObjectTraj::GetTransform(int t) const {
  return toRaveTransform(wxyz.row(t), xyz.row(t));
}

vector<SceneStateInfoPtr> SimResult::ToSceneStateInfos() {
  vector<SceneStateInfoPtr> out;
  for (int t = 0; t < robot_traj.rows(); ++t) {
    SceneStateInfoPtr info(new SceneStateInfo);
    info->timestep = t;
    for (Name2ObjectTraj::iterator i = obj_trajs.begin(); i != obj_trajs.end(); ++i) {
      ObjectStateInfoPtr obj_info(new ObjectStateInfo);
      obj_info->name = i->first;
      obj_info->xyz = i->second->xyz.row(t);
      obj_info->wxyz = i->second->wxyz.row(t);
      info->obj_state_infos.push_back(obj_info);
    }
    out.push_back(info);
  }
  return out;
}

void SimResult::Clear() {
  obj_trajs.clear();
  collisions.clear();
  robot_traj.setConstant(robot_traj.rows(), robot_traj.cols(), -999);
}

template<typename T>
static bool isIn(const T& x, const vector<T>& v) {
  return std::find(v.begin(), v.end(), x) != v.end();
}

// output collision convention: A is robot, B is object
static CollisionVec FilterAndFlipCollisions(const CollisionVec &collisions, const string& robot_name, const StrVec& obj_names) {
  CollisionVec out;
  //CollisionVec flipped;
  BOOST_FOREACH(bs::CollisionPtr c, collisions) {
    string nameA = c->linkA->GetParent()->GetName();
    string nameB = c->linkB->GetParent()->GetName();
    if (isIn(nameA, obj_names) && nameB == robot_name) {
      //flipped.push_back(c->Flipped());
    } else if (isIn(nameB, obj_names) && nameA == robot_name) {
      out.push_back(c);
    }
  }
  //out.insert(out.end(), flipped.begin(), flipped.end());
  return out;
}


Simulation::Simulation(TrajOptProb& prob) :
  m_prob(prob),
  m_rad(prob.GetRAD()),
  m_env(prob.GetEnv()),
  m_runs_executed(0),
  m_curr_result(new SimResult),
  m_curr_result_upsampled(new SimResult)
{
  assert(prob.HasSimulation());
}

void Simulation::SetSimParams(const SimParamsInfo& p) {
  m_params = p;
}

SimulationPtr Simulation::GetOrCreate(TrajOptProb& prob) {
  assert(prob.HasSimulation());
  EnvironmentBasePtr env = prob.GetEnv();
  const string name = "trajopt_simulation";
  UserDataPtr ud = GetUserData(*env, name);
  if (!ud) {
    RAVELOG_DEBUG("creating physics simulation for environment");
    ud.reset(new Simulation(prob));
    SetUserData(*env, name, ud);
  } else {
    RAVELOG_DEBUG("already have physics simulation for environment");
  }
  return boost::dynamic_pointer_cast<Simulation>(ud);
}

void Simulation::RunTraj(const TrajArray& traj) {
  cout << "running trajectory " << traj<< endl;
  cout << m_params.dt << ' ' << m_params.dynamic_obj_names << ' ' << m_params.traj_time << endl;

  m_curr_result->Clear();
  m_curr_result_upsampled->Clear();

  // construct bullet mirror
  bs::BulletEnvironment bt_env(m_env, m_params.dynamic_obj_names);
  bt_env.SetContactDistance(.05);
  bs::BulletObjectPtr bt_robot = bt_env.GetObjectByName(m_rad->GetRobot()->GetName());
  vector<bs::BulletObjectPtr> bt_dynamic_objs = bt_env.GetDynamicObjects();
  assert(bt_dynamic_objs.size() == m_params.dynamic_obj_names.size());

  // upsample trajectory, according to traj time and dt
  int len = traj.rows();
  int len_upsampled = ceil(m_params.traj_time / m_params.dt);
  TrajArray traj_upsampled;
  if (len_upsampled <= len) {
    traj_upsampled = traj;
    len_upsampled = len;
  } else {
    traj_upsampled = util::interp2d(
      VectorXd::LinSpaced(len_upsampled, 0, m_params.traj_time),
      VectorXd::LinSpaced(len, 0, m_params.traj_time),
      traj
    );
  }

  // run and record upsampled traj
  Name2ObjectTraj obj_trajs_upsampled;
  BOOST_FOREACH(const string& name, m_params.dynamic_obj_names) {
    ObjectTrajPtr t(new ObjectTraj(len_upsampled));
    obj_trajs_upsampled.insert(make_pair(name, t));
  }
  RobotBase::RobotStateSaver saver = m_rad->Save();
  cout << "upsampled len " << len_upsampled << endl;
  for (int i = 0; i < len_upsampled; ++i) {
    // set joint angles
    m_rad->SetDOFValues(toDblVec(traj_upsampled.row(i)));
    bt_robot->UpdateBullet();

    // quasistatic step
    bt_env.Step(m_params.dt, m_params.max_substeps, m_params.internal_dt);
    BOOST_FOREACH(bs::BulletObjectPtr obj, bt_dynamic_objs) {
      obj->SetLinearVelocity(btVector3(0, 0, 0));
      obj->SetAngularVelocity(btVector3(0, 0, 0));
    }

    // record object states
    BOOST_FOREACH(bs::BulletObjectPtr obj, bt_dynamic_objs) {
      ObjectTrajPtr obj_traj = obj_trajs_upsampled[obj->GetName()];
      btTransform t = obj->GetTransform();
      obj_traj->xyz.row(i) = toVector3d(t.getOrigin());
      obj_traj->wxyz.row(i) = toVector4d(t.getRotation());
    }

    // record collisions
    vector<bs::CollisionPtr> collisions;
    BOOST_FOREACH(bs::BulletObjectPtr obj, bt_dynamic_objs) {
      vector<bs::CollisionPtr> c = FilterAndFlipCollisions(bt_env.ContactTest(obj), m_rad->GetRobot()->GetName(), m_params.dynamic_obj_names);
      collisions.insert(collisions.end(), c.begin(), c.end());
    }
    m_curr_result_upsampled->collisions.push_back(collisions);
//    m_curr_result_upsampled->collisions.push_back(FilterAndFlipCollisions(
//      bt_env.DetectCollisions(),
//      m_rad->GetRobot()->GetName(),
//      m_params.dynamic_obj_names
//    ));
  }
  m_curr_result_upsampled->robot_traj = traj_upsampled;
  m_curr_result_upsampled->obj_trajs = obj_trajs_upsampled;

  // downsample to original trajectory timing
  Name2ObjectTraj obj_trajs;
  BOOST_FOREACH(const string& name, m_params.dynamic_obj_names) {
    ObjectTrajPtr t(new ObjectTraj(len));
    obj_trajs.insert(make_pair(name, t));
  }
  for (int i = 0; i < len; ++i) {
    int ind_of_orig = (int) ((len_upsampled-1.)/(len-1.) * i);
    BOOST_FOREACH(bs::BulletObjectPtr obj, bt_dynamic_objs) {
      ObjectTrajPtr obj_traj = obj_trajs[obj->GetName()];
      ObjectTrajPtr obj_traj_upsampled = obj_trajs_upsampled[obj->GetName()];
      obj_traj->xyz.row(i) = obj_traj_upsampled->xyz.row(ind_of_orig);
      obj_traj->wxyz.row(i) = obj_traj_upsampled->wxyz.row(ind_of_orig);
    }
    m_curr_result->collisions.push_back(m_curr_result_upsampled->collisions[ind_of_orig]);
  }
  m_curr_result->robot_traj = traj;
  m_curr_result->obj_trajs = obj_trajs;

  ++m_runs_executed;
//  cout << "simulation done." << endl;
//  BOOST_FOREACH(const string& name, m_params.dynamic_obj_names) {
//    cout << name << ' ' << obj_trajs[name]->xyz << ' ' << obj_trajs[name]->wxyz << endl;
//  }
}

SimResultPtr Simulation::GetResult() {
  return m_curr_result;
}

SimResultPtr Simulation::GetResultUpsampled() {
  return m_curr_result_upsampled;
}


void Simulation::PreEvaluateCallback(const DblVec& x) {
  RunTraj(getTraj(x, m_prob.GetVars()));
}

Optimizer::Callback Simulation::MakePreEvaluateCallback() {
  return boost::bind(&Simulation::PreEvaluateCallback, this, _2);
}



#if 0
void SimulationPlotterDummyCost::Plot(const DblVec& x, OR::EnvironmentBase&, std::vector<OR::GraphHandlePtr>& handles) {
  SimResultPtr res(m_sim->GetResult());
  for (Name2ObjectTraj::iterator i = res->obj_trajs.begin(); i != res->obj_trajs.end(); ++i) {

  }
}
#endif


ObjectSlideCost::ObjectSlideCost(int timestep, const string& object_name, double dist_pen, double coeff, RobotAndDOFPtr rad, const VarVector& vars0, const VarVector& vars1, SimulationPtr sim) :
    m_timestep(timestep), // TODO: remove
    m_object_name(object_name),
    m_dist_pen(dist_pen),
    m_coeff(coeff),
    m_rad(rad),
    m_vars0(vars0),
    m_vars1(vars1),
    m_sim(sim)
{ }

bool ObjectSlideCost::GetCollisionData(bs::CollisionPtr& out_c0, bs::CollisionPtr& out_c1, Vector3d &out_n) {
  CollisionVec collisions0 = m_sim->GetResult()->collisions[m_timestep];
  CollisionVec collisions1 = m_sim->GetResult()->collisions[m_timestep+1];
  if (collisions0.empty() || collisions1.empty()) {
    cout << "WARNING: zero convexification because collisions are empty: " << collisions0.empty() << ' ' << collisions1.empty() << endl;
    return false;
  }
  cout << "at time " << m_timestep << ": num collisions " << collisions0.size() << "\tpos: " << m_sim->GetResult()->obj_trajs.begin()->second->xyz.row(m_timestep) << endl;
  BOOST_FOREACH(bs::CollisionPtr& c, collisions0) {
    cout << '\t' << toStr(c->linkA) << ' ' << toStr(c->linkB) << ' ' << toStr(c->ptA) << ' ' << toStr(c->ptB) << ' ' << toStr(c->normalB2A) << ' ' << c->distance << endl;
  }

  // TODO: merge multiple collisions, check that they're all about the same
  out_c0 = collisions0[0];
  out_c1 = collisions1[0];
  out_n = Vector3d(-1, 0, 0);
  //Matrix3d rot(toRotationMatrix(m_sim->GetResult()->obj_trajs[m_object_name]->wxyz.row(m_timestep)));
  //Matrix3d rot_orig(toRotationMatrix(m_sim->GetResult()->obj_trajs[m_object_name]->wxyz.row(0)));
  //out_n = rot_orig.inverse().transpose() * rot.transpose() * toVector3d(out_c0->normalB2A);
  //out_n = toVector3d(out_c0->normalB2A);
  return true;
}

ConvexObjectivePtr ObjectSlideCost::convex(const vector<double>& x, Model* model) {
  ConvexObjectivePtr out(new ConvexObjective(model));

  bs::CollisionPtr c0, c1; Vector3d n;
  if (!GetCollisionData(c0, c1, n)) {
    return out;
  }
  //Vector3d n(toVector3d(c0->normalB2A));
  //Vector3d n(-1, 0, 0);
  //OR::Transform obj_init_trans = toRaveTransform(m_sim->GetResult()->obj_trajs[m_object_name]->wxyz.row(0), m_sim->GetResult()->obj_trajs[m_object_name]->xyz.row(0));
  //OR::Vector local_manip_pt = obj_init_trans.inverse() * toOR(c0->ptA);

  //VectorXd vals0 = m_sim->GetResult()->robot_traj.row(m_timestep);
  //VectorXd vals1 = m_sim->GetResult()->robot_traj.row(m_timestep+1);
  DblVec vals0 = getDblVec(x, m_vars0);
  DblVec vals1 = getDblVec(x, m_vars1);

  m_rad->SetDOFValues(vals0);
  DblMatrix jac0 = m_rad->PositionJacobian(c0->linkA->GetIndex(), toOR(c0->ptA));
  //OR::Transform trans0 = m_rad->GetRobot()->GetLink(c0->linkA->GetName())->GetTransform();
  m_rad->SetDOFValues(vals1);
  DblMatrix jac1 = m_rad->PositionJacobian(c1->linkA->GetIndex(), toOR(c1->ptA));

  VectorXd grad0 = -n.transpose() * jac0;
  VectorXd grad1 = -n.transpose() * jac1;

  AffExpr expr(-n.dot(toVector3d(c1->ptA - c0->ptA)));

  exprInc(expr, varDot(grad1, m_vars1));
  exprInc(expr, -grad1.dot(toVectorXd(vals1)));

  exprInc(expr, varDot(-grad0, m_vars0));
  exprInc(expr, grad0.dot(toVectorXd(vals0)));

  cout << "linearization " << m_timestep << ": " << expr << endl;

  out->addHinge(expr, m_coeff);
  return out;
}

double ObjectSlideCost::value(const vector<double>& x) {
//  CollisionVec collisions0 = m_sim->GetResult()->collisions[m_timestep];
//  CollisionVec collisions1 = m_sim->GetResult()->collisions[m_timestep+1];
//  cout << "at time " << m_timestep << ": num collisions " << collisions0.size() << "\tpos: " << m_sim->GetResult()->obj_trajs.begin()->second->xyz.row(m_timestep) << endl;
////  BOOST_FOREACH(bs::CollisionPtr& c, collisions0) {
////    cout << '\t' << toStr(c->linkA) << ' ' << toStr(c->linkB) << ' ' << toStr(c->ptA) << ' ' << toStr(c->ptB) << ' ' << toStr(c->normalB2A) << ' ' << c->distance << endl;
////  }
//  if (collisions0.empty() || collisions1.empty()) {
//    cout << "\tval: ZERO" << endl;
//    return 0.;
//  }
//  // TODO: merge multiple collisions, check that they're all about the same
//  bs::CollisionPtr c0 = collisions0[0];
//  bs::CollisionPtr c1 = collisions1[0];
//  Vector3d n = toVector3d(c0->normalB2A);
//  //Vector3d n(-1, 0, 0);

  bs::CollisionPtr c0, c1; Vector3d n;
  if (!GetCollisionData(c0, c1, n)) {
    return 0;
  }

  double val = pospart(-m_coeff * n.dot(toVector3d(c1->ptA - c0->ptA)));
  cout << "\tval: " << val << endl;
  return val;
}

void ObjectSlideCost::Plot(const DblVec& x, OR::EnvironmentBase& env, std::vector<OR::GraphHandlePtr>& handles) {
//  CollisionVec collisions0 = m_sim->GetResult()->collisions[m_timestep];
//  CollisionVec collisions1 = m_sim->GetResult()->collisions[m_timestep+1];
//  if (collisions0.empty() || collisions1.empty()) {
//    return;
//  }
//  bs::CollisionPtr c0 = collisions0[0];
//  bs::CollisionPtr c1 = collisions1[0];

  bs::CollisionPtr c0, c1; Vector3d n;
  if (!GetCollisionData(c0, c1, n)) {
    return;
  }

  typedef OpenRAVE::RaveVector<float> RaveVectorf;
  RaveVectorf color = RaveVectorf(0,1,0,1);
  Vector3d dir = n * n.transpose() * toVector3d(c1->ptA - c0->ptA);
  handles.push_back(env.drawarrow(toOR(c0->ptA), toOR(toVector3d(c0->ptA) + dir*5), .0025, color));
}


} // namespace trajopt
