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

static const double GRAVITY = -9.8;
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

  BoxState init_state; init_state.x = Vector3d(0, 0, 5-.58+.5 -.1);
  BoxProperties props;
  props.mass = 1.0;
  props.half_extents = Vector3d(.5, .5, .5);
  props.I_body = props.I_body_inv = Eigen::Matrix3d::Identity();
  BoxPtr box(new Box("box", prob.get(), props, init_state));
  prob->addObject(box);

  BoxGroundContactPtr box_ground_cont(new BoxGroundContact("box_ground_cont", box.get(), ground.get()));
  prob->addContact(box_ground_cont);

  prob->addCost(CostPtr(new ZeroCost())); // shut up

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
