#include "scene_objectives.hpp"

#include "simulation/bulletsim_lite.h"
#include "utils/interpolation.hpp"

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





Simulation::Simulation(RobotAndDOFPtr rad) :
  m_rad(rad),
  m_runs_executed(0),
  m_env(rad->GetRobot()->GetEnv()),
  m_curr_result(new SimResult),
  m_curr_result_upsampled(new SimResult)
{ }

void Simulation::SetSimParams(const SimParams& p) {
  m_params = p;
}

SimulationPtr Simulation::GetOrCreate(RobotAndDOFPtr rad) {
  EnvironmentBasePtr env = rad->GetRobot()->GetEnv();
  const string name = "trajopt_simulation";
  UserDataPtr ud = GetUserData(env, name);
  if (!ud) {
    RAVELOG_DEBUG("creating physics simulation for environment");
    ud.reset(new Simulation(rad));
    SetUserData(env, name, ud);
  } else {
    RAVELOG_DEBUG("already have physics simulation for environment");
  }
  return boost::dynamic_pointer_cast<Simulation>(ud);
}

void Simulation::Run(const TrajArray& traj) {
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




ObjectSlideCost::ObjectSlideCost(int timestep, const string& object_name, double dist_pen, double coeff, RobotAndDOFPtr rad, const VarVector& vars0, const VarVector& vars1) :
    m_timestep(timestep),
    m_object_name(object_name),
    m_dist_pen(dist_pen),
    m_coeff(coeff),
    m_rad(rad),
    m_vars0(vars0),
    m_vars1(vars1),
    m_sim(Simulation::GetOrCreate(rad))
{ }


ConvexObjectivePtr ObjectSlideCost::convex(const vector<double>& x, Model* model) {
  ConvexObjectivePtr out(new ConvexObjective(model));

  return out;
}


double ObjectSlideCost::value(const vector<double>& x) {
  SimResultPtr res = m_sim->Run(x); // FIXME
  ObjectTraj& obj_traj = res->obj_trajs[m_object_name];

  obj_traj.xyz
}

void ObjectSlideCost::Plot(const DblVec& x, OR::EnvironmentBase& env, std::vector<OR::GraphHandlePtr>& handles) {
}


} // namespace trajopt
