#include "scene_objectives.hpp"

#include "simulation/bulletsim_lite.h"
#include "utils/eigen_conversions.hpp"
#include "utils/interpolation.hpp"
#include "rave_utils.hpp"

namespace trajopt {

/*
OR::Vector toOR(const btVector3& v) {
  return OR::Vector(v.x(), v.y(), v.z());
}
btQuaternion toBtQuat(const OR::Vector& q) {
  return btQuaternion(q[1], q[2], q[3], q[0]);
}
btTransform toBt(const OR::Transform& t){
  return btTransform(toBtQuat(t.rot), toBt(t.trans));
}*/

Vector3d toEigen(const btVector3 &v) {
  return Vector3d(v.x(), v.y(), v.z());
}

Vector4d toEigenQuat(const btQuaternion &q) {
  // use OpenRAVE's convention
  return Vector4d(q.w(), q.x(), q.y(), q.z());
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




Simulation::Simulation(TrajOptProb& prob) :
  m_prob(prob),
  m_rad(prob.GetRAD()),
  m_env(prob.GetEnv()),
  m_runs_executed(0),
  m_curr_result(new SimResult),
  m_curr_result_upsampled(new SimResult)
{ }

void Simulation::SetSimParams(const SimParams& p) {
  m_params = p;
}

SimulationPtr Simulation::GetOrCreate(TrajOptProb& prob) {
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
  // construct bullet mirror
  bs::BulletEnvironment bt_env(m_env, m_params.dynamic_obj_names);
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
    obj_trajs_upsampled[name].reset(new ObjectTraj(len_upsampled));
  }
  RobotBase::RobotStateSaver saver = m_rad->Save();
  for (int i = 0; i < len_upsampled; ++i) {
    m_rad->SetDOFValues(toDblVec(traj_upsampled.row(i)));
    bt_robot->UpdateBullet();

    bt_env.Step(m_params.dt, m_params.max_substeps, m_params.internal_dt);

    // record object states
    BOOST_FOREACH(bs::BulletObjectPtr obj, bt_dynamic_objs) {
      ObjectTrajPtr obj_traj = obj_trajs_upsampled[obj->GetName()];
      btTransform t = obj->GetTransform();
      obj_traj->xyz.row(i) = toEigen(t.getOrigin());
      obj_traj->wxyz.row(i) = toEigenQuat(t.getRotation());
    }
  }
  m_curr_result_upsampled->robot_traj = traj_upsampled;
  m_curr_result_upsampled->obj_trajs = obj_trajs_upsampled;

  // downsample to original trajectory timing
  Name2ObjectTraj obj_trajs;
  BOOST_FOREACH(const string& name, m_params.dynamic_obj_names) {
    obj_trajs[name].reset(new ObjectTraj(len));
  }
  for (int i = 0; i < len; ++i) {
    int ind_of_orig = (int) ((len_upsampled-1.)/(len-1.) * i);
    BOOST_FOREACH(bs::BulletObjectPtr obj, bt_dynamic_objs) {
      ObjectTrajPtr obj_traj = obj_trajs[obj->GetName()];
      ObjectTrajPtr obj_traj_upsampled = obj_trajs_upsampled[obj->GetName()];
      obj_traj->xyz.row(ind_of_orig) = obj_traj_upsampled->xyz.row(i);
      obj_traj->wxyz.row(ind_of_orig) = obj_traj_upsampled->wxyz.row(i);
    }
  }
  m_curr_result->robot_traj = traj;
  m_curr_result->obj_trajs = obj_trajs;

  ++m_runs_executed;
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


ConvexObjectivePtr ObjectSlideCost::convex(const vector<double>& x, Model* model) {
  ConvexObjectivePtr out(new ConvexObjective(model));

  return out;
}


double ObjectSlideCost::value(const vector<double>& x) {
  /*
  SimResultPtr res = m_sim->Run(x); // FIXME
  ObjectTraj& obj_traj = res->obj_trajs[m_object_name];

  obj_traj.xyz*/
  return 0;
}

void ObjectSlideCost::Plot(const DblVec& x, OR::EnvironmentBase& env, std::vector<OR::GraphHandlePtr>& handles) {
}


} // namespace trajopt
