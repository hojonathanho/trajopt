#include <iostream>
#include <openrave-core.h>
#include "ipi/logging.hpp"
#include "ipi/sco/optimizers.hpp"
#include "trajopt/utils.hpp"
#include "dynamics_problem.hpp"
#include "box.h"

using namespace ipi::sco;
using namespace trajopt;
using namespace trajopt::dynamics;
using namespace util;
using namespace OpenRAVE;
using namespace std;

static const double GRAVITY = 0.0; // :)
static const double GROUND_Z = 0.0;

int main(int argc, char* argv[]) {
  RaveInitialize(false, Level_Debug);
  OR::EnvironmentBasePtr env = RaveCreateEnvironment();
  env->StopSimulation();

  DynamicsProblemPtr prob(new DynamicsProblem(env));
  prob->setNumTimesteps(15);
  prob->setDt(1.5/prob->m_timesteps);
  prob->setGravity(Vector3d(0, 0, GRAVITY));

  GroundPtr ground(new Ground("ground", prob.get(), GROUND_Z));
  prob->addObject(ground);

  BoxState init_state1; init_state1.x = Vector3d(0, 0, 5-.58+.5 -.1);
  BoxProperties props1;
  props1.mass = 1.0;
  props1.half_extents = Vector3d(.5, .5, .5);
  props1.I_body = props1.I_body_inv = Eigen::Matrix3d::Identity();
  BoxPtr box1(new Box("box1", prob.get(), props1, init_state1));
  prob->addObject(box1);

  BoxState init_state2; init_state2.x = Vector3d(0, 0, 5-.58+.5 -.1);
  BoxProperties props2;
  props2.mass = 1.0;
  props2.half_extents = Vector3d(.5, .5, .5);
  props2.I_body = props2.I_body_inv = Eigen::Matrix3d::Identity();
  BoxPtr box2(new Box("box2", prob.get(), props2, init_state2));
  prob->addObject(box2);

//  BoxGroundContactPtr box_ground_cont(new BoxGroundContact("box_ground_cont", box.get(), ground.get()));
//  prob->addContact(box_ground_cont);

  BoxBoxContactPtr bb_cont(new BoxBoxContact("bb_cont", box1.get(), box2.get()));
  prob->addContact(bb_cont);

  DynamicsOptResultPtr result = OptimizeDynamicsProblem(prob, true);

  cout << "x:\n" << getTraj(result->optimizer->x(), box->m_trajvars.x) << endl;
  cout << "v:\n" << getTraj(result->optimizer->x(), box->m_trajvars.v) << endl;
  cout << "force:\n" << getTraj(result->optimizer->x(), box->m_trajvars.force) << endl;
//  cout << "q:\n" << getTraj(result->optimizer->x(), box->m_trajvars.q) << endl;
//  cout << "w:\n" << getTraj(result->optimizer->x(), box->m_trajvars.w) << endl;
//  cout << "T:\n" << getTraj(result->optimizer->x(), box->m_trajvars.torque) << endl;

  cout << "gp:\n" << getTraj(result->optimizer->x(), box_ground_cont->m_trajvars.p) << endl;
  cout << "gf:\n" << getTraj(result->optimizer->x(), box_ground_cont->m_trajvars.f) << endl;

//  boost::thread run_traj(RunTraj, result, env, *rad);


  env.reset();
  RaveDestroy();

  return 0;
}
