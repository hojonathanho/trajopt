#include <iostream>
#include "ipi/sco/modeling_utils.hpp"
#include "trajopt/robot_and_dof.hpp"
#include "utils/stl_to_string.hpp"
#include "ipi/logging.hpp"
#include "ipi/sco/expr_op_overloads.hpp"
#include <openrave-core.h>
#include "ipi/sco/optimizers.hpp"
#include "trajopt/utils.hpp"
#include "utils/vector_ops.hpp"
#include "utils/eigen_conversions.hpp"
#include "utils/stl_to_string.hpp"
#include "box.h"
using namespace ipi::sco;
using namespace trajopt;
using namespace util;
using namespace OpenRAVE;
using namespace std;

static const double GRAVITY = -9.8;

/*
void MakeVariablesAndBounds(dynamics::DynamicsProblemPtr prob, const vector<dynamics::BoxState> &init_states, vector<dynamics::BoxPtr> &out_objects) {
  int n_objects = init_states.size();
  vector<double> vlower, vupper;
  vector<string> names;
  out_objects.clear();

  dynamics::BoxProperties props;
  props.mass = 1.0;
  props.half_extents = Vector3d(.5, .5, .5);
  props.I_body = props.I_body_inv = Eigen::Matrix3d::Identity();

  for (int i = 0; i < n_objects; ++i) {
    dynamics::BoxPtr box(new dynamics::Box(prob.get(), props, init_states[i]));
    out_objects.push_back(box);
  }

  for (dynamics::BoxPtr &box : out_objects) {
    box->registerGroundContact();
    box->fillVarNamesAndBounds(names, vlower, vupper);
  }

  assert(names.size() == vlower.size() && names.size() == vupper.size());

  prob->createVariables(names, vlower, vupper);

  vector<Var> vars = prob->getVars();
  int k = 0;
  for (int i = 0; i < n_objects; ++i) {
    k += out_objects[i]->setVariables(vars, k);
  }
  assert(k == vars.size());

  cout << Str(names) << endl;

  for (dynamics::BoxPtr &box : out_objects) {
    box->addConstraints();
    box->addGroundNonpenetrationCnts(GROUND_Z);
  }
}*/

int main(int argc, char* argv[]) {
  RaveInitialize(false, Level_Debug);
  OR::EnvironmentBasePtr env = RaveCreateEnvironment();
  env->StopSimulation();

  dynamics::DynamicsProblemPtr prob(new dynamics::DynamicsProblem(env));
  prob->setNumTimesteps(10);
  prob->setDt(1./prob->m_timesteps);
  prob->setGravity(Vector3d(0, 0, GRAVITY));

  dynamics::BoxState init_state; init_state.x = Vector3d(0, 0, 5-.58+.5);
  dynamics::BoxProperties props;
  props.mass = 1.0;
  props.half_extents = Vector3d(.5, .5, .5);
  props.I_body = props.I_body_inv = Eigen::Matrix3d::Identity();
  dynamics::BoxPtr box(new dynamics::Box(prob.get(), props, init_state));
  prob->addObject(box);

  prob->setUpProblem();

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

  optimizer.initialize(DblVec(prob->getVars().size(), 0));
  OptStatus status = optimizer.optimize();
  cout << "x:\n" << getTraj(optimizer.x(), box->m_trajvars.x) << endl;
  cout << "v:\n" << getTraj(optimizer.x(), box->m_trajvars.p) << endl;
  cout << "a:\n" << getTraj(optimizer.x(), box->m_trajvars.force) << endl;
  cout << "gp:\n" << getTraj(optimizer.x(), box->m_ground_conts[0]->m_trajvars.p) << endl;
  cout << "gf:\n" << getTraj(optimizer.x(), box->m_ground_conts[0]->m_trajvars.f) << endl;

//  boost::thread run_traj(RunTraj, result, env, *rad);
//  ViewerBasePtr viewer = RaveCreateViewer(env,"qtosg");
//  viewer->main();

  env.reset();
  RaveDestroy();

  return 0;
}
