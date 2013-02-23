#include <iostream>
#include <openrave-core.h>
#include "ipi/logging.hpp"
#include "ipi/sco/optimizers.hpp"
#include "trajopt/utils.hpp"
#include "osgviewer/osgviewer.hpp"
#include "dynamics_problem.hpp"
#include "box.h"

using namespace ipi::sco;
using namespace trajopt;
using namespace util;
using namespace OpenRAVE;
using namespace std;

static const double GRAVITY = -9.8;
static const double GROUND_Z = 0.0;

int main(int argc, char* argv[]) {
  RaveInitialize(false, Level_Debug);
  OR::EnvironmentBasePtr env = RaveCreateEnvironment();
  env->StopSimulation();

  dynamics::DynamicsProblemPtr prob(new dynamics::DynamicsProblem(env));
  prob->setNumTimesteps(10);
  prob->setDt(1./prob->m_timesteps);
  prob->setGravity(Vector3d(0, 0, GRAVITY));

  dynamics::GroundPtr ground(new dynamics::Ground("ground", prob.get(), GROUND_Z));
  prob->addObject(ground);

  dynamics::BoxState init_state; init_state.x = Vector3d(0, 0, 5-.58+.5 -.1);
  dynamics::BoxProperties props;
  props.mass = 1.0;
  props.half_extents = Vector3d(.5, .5, .5);
  props.I_body = props.I_body_inv = Eigen::Matrix3d::Identity();
  dynamics::BoxPtr box(new dynamics::Box("box", prob.get(), props, init_state));
  prob->addObject(box);

  dynamics::BoxGroundContactPtr box_ground_cont(new dynamics::BoxGroundContact("box_ground_cont", box.get(), ground.get()));
  prob->addContact(box_ground_cont);

  prob->setUpProblem();

  OSGViewerPtr viewer(new OSGViewer(env));
  viewer->Idle();

  //srand(time(NULL));
  //for (int i = 0; i < initSoln.size(); ++i) initSoln[i] += 0.01*((double)rand()/RAND_MAX-.5);
  //for (int i = 0; i < initSoln.size(); ++i) cout << initSoln[i] << ' '; cout << endl;
  prob->addCost(CostPtr(new dynamics::ZeroCost())); // shut up
  BasicTrustRegionSQP optimizer(prob);
  optimizer.min_trust_box_size_ = 1e-7;
  optimizer.min_approx_improve_= 1e-7;
  optimizer.cnt_tolerance_ = 1e-7;
  optimizer.trust_box_size_ = 1;
  optimizer.max_iter_ = 1000;

  optimizer.initialize(prob->makeInitialSolution());
  OptStatus status = optimizer.optimize();
  cout << "x:\n" << getTraj(optimizer.x(), box->m_trajvars.x) << endl;
  cout << "v:\n" << getTraj(optimizer.x(), box->m_trajvars.v) << endl;
  cout << "a:\n" << getTraj(optimizer.x(), box->m_trajvars.force) << endl;
//  cout << "q:\n" << getTraj(optimizer.x(), box->m_trajvars.q) << endl;
//  cout << "w:\n" << getTraj(optimizer.x(), box->m_trajvars.w) << endl;
//  cout << "T:\n" << getTraj(optimizer.x(), box->m_trajvars.torque) << endl;
  //cout << "gp:\n" << getTraj(optimizer.x(), box->m_ground_conts[0]->m_trajvars.p) << endl;
  //cout << "gf:\n" << getTraj(optimizer.x(), box->m_ground_conts[0]->m_trajvars.f) << endl;

//  boost::thread run_traj(RunTraj, result, env, *rad);


  env.reset();
  RaveDestroy();

  return 0;
}
