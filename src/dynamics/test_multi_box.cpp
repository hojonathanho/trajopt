#include <iostream>
#include <openrave-core.h>
#include "sco/optimizers.hpp"
#include "trajopt/utils.hpp"
#include "dynamics_problem.hpp"
#include "box.hpp"
#include "contacts.hpp"

using namespace sco;
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

  BoxState init_state1; init_state1.x = Vector3d(0, 0, .5);
  BoxProperties props1;
  props1.mass = 1.0;
  props1.half_extents = Vector3d(.5, .5, .5);
  props1.I_body = props1.I_body_inv = Eigen::Matrix3d::Identity();
  BoxPtr box1(new Box("box1", prob.get(), props1, init_state1));
  prob->addObject(box1);

  BoxState init_state2; init_state2.x = Vector3d(1, 0, .5);
  BoxProperties props2;
  props2.mass = 1.0;
  props2.half_extents = Vector3d(.5, .5, .5);
  props2.I_body = props2.I_body_inv = Eigen::Matrix3d::Identity();
  BoxPtr box2(new Box("box2", prob.get(), props2, init_state2));
  prob->addObject(box2);

  BoxGroundContactPtr box1_ground_cont(new BoxGroundContact("box1_ground_cont", box1.get(), ground.get()));
  prob->addContact(box1_ground_cont);

  BoxGroundContactPtr box2_ground_cont(new BoxGroundContact("box2_ground_cont", box2.get(), ground.get()));
  prob->addContact(box2_ground_cont);

  BoxBoxContactPtr bb_cont(new BoxBoxContact("bb_cont", box1.get(), box2.get()));
  prob->addContact(bb_cont);

  DynamicsOptResultPtr result = OptimizeDynamicsProblem(prob, true);
  vector<double> &soln = result->optimizer->x();

  cout << "x1:\n" << getTraj(soln, box1->m_trajvars.x) << endl;
  cout << "v1:\n" << getTraj(soln, box1->m_trajvars.v) << endl;
  cout << "force1:\n" << getTraj(soln, box1->m_trajvars.force) << endl;
  cout << "x2:\n" << getTraj(soln, box2->m_trajvars.x) << endl;
  cout << "v2:\n" << getTraj(soln, box2->m_trajvars.v) << endl;
  cout << "force2:\n" << getTraj(soln, box2->m_trajvars.force) << endl;
//  cout << "q:\n" << getTraj(soln, box->m_trajvars.q) << endl;
//  cout << "w:\n" << getTraj(soln, box->m_trajvars.w) << endl;
//  cout << "T:\n" << getTraj(soln, box->m_trajvars.torque) << endl;

  cout << "box-box contact f:\n" << getTraj(soln, bb_cont->m_trajvars.f) << endl;
  cout << "box-box contact p1:\n" << getTraj(soln, bb_cont->m_trajvars.p1) << endl;
  cout << "box-box contact p2:\n" << getTraj(soln, bb_cont->m_trajvars.p2) << endl;

//  cout << "gp:\n" << getTraj(soln, box_ground_cont->m_trajvars.p) << endl;
//  cout << "gf:\n" << getTraj(soln, box_ground_cont->m_trajvars.f) << endl;

//  boost::thread run_traj(RunTraj, result, env, *rad);


  env.reset();
  RaveDestroy();

  return 0;
}
