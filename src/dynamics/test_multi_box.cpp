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
  // Scenario: box1 wants to touch box3, but box2 is in the way
  ProblemSpec prob_spec;
  prob_spec.timesteps = 15;
  prob_spec.dt = 1.5/prob_spec.timesteps;
  prob_spec.gravity = Vector3d(0, 0, GRAVITY);

  ///// Objects /////
  GroundSpecPtr ground_spec(new GroundSpec);
  ground_spec->name = "ground";
  ground_spec->z = 0.;
  prob_spec.objects.push_back(ground_spec);

  BoxSpecPtr box1_spec(new BoxSpec(prob_spec));
  box1_spec->name = "box1";
  box1_spec->props.mass = 1.0;
  box1_spec->props.half_extents = Vector3d(.5, .5, .5);
  box1_spec->props.I_body = box1_spec->props.I_body_inv = Eigen::Matrix3d::Identity();
  box1_spec->is_kinematic = true;
  box1_spec->state_0.x = Vector3d(0, 0, .5);
  box1_spec->traj_init = BoxStateTraj::FromConstant(prob_spec.timesteps, box1_spec->state_0);
  prob_spec.objects.push_back(box1_spec);

  BoxSpecPtr box2_spec(new BoxSpec(prob_spec));
  box2_spec->name = "box2";
  box2_spec->props.mass = 1.0;
  box2_spec->props.half_extents = Vector3d(.5, .5, .5);
  box2_spec->props.I_body = box2_spec->props.I_body_inv = Eigen::Matrix3d::Identity();
  box2_spec->is_kinematic = false;
  box2_spec->state_0.x = Vector3d(2, 0, .5);
  box2_spec->traj_init = BoxStateTraj::FromConstant(prob_spec.timesteps, box2_spec->state_0);
  prob_spec.objects.push_back(box2_spec);

  BoxSpecPtr box3_spec(new BoxSpec(prob_spec));
  box3_spec->name = "box3";
  box3_spec->props.mass = 1.0;
  box3_spec->props.half_extents = Vector3d(.5, .5, .5);
  box3_spec->props.I_body = box3_spec->props.I_body_inv = Eigen::Matrix3d::Identity();
  box3_spec->is_kinematic = false;
  box3_spec->state_0.x = Vector3d(4, 0, .5);
  box3_spec->traj_init = BoxStateTraj::FromConstant(prob_spec.timesteps, box3_spec->state_0);
  prob_spec.objects.push_back(box3_spec);

  ///// Contacts /////
  BoxGroundContactSpecPtr box2_ground_cont_spec(new BoxGroundContactSpec);
  box2_ground_cont_spec->name = "box2_ground_cont";
  box2_ground_cont_spec->box_name = "box2";
  box2_ground_cont_spec->ground_name = "ground";
  prob_spec.contacts.push_back(box2_ground_cont_spec);

  BoxGroundContactSpecPtr box3_ground_cont_spec(new BoxGroundContactSpec);
  box3_ground_cont_spec->name = "box3_ground_cont";
  box3_ground_cont_spec->box_name = "box3";
  box3_ground_cont_spec->ground_name = "ground";
  prob_spec.contacts.push_back(box3_ground_cont_spec);

  BoxBoxContactSpecPtr b1b2_cont_spec(new BoxBoxContactSpec);
  b1b2_cont_spec->name = "b1b2_cont";
  b1b2_cont_spec->box1_name = "box1";
  b1b2_cont_spec->box2_name = "box2";
  prob_spec.contacts.push_back(b1b2_cont_spec);

  BoxBoxContactSpecPtr b2b3_cont_spec(new BoxBoxContactSpec);
  b2b3_cont_spec->name = "b2b3_cont";
  b2b3_cont_spec->box1_name = "box2";
  b2b3_cont_spec->box2_name = "box3";
  prob_spec.contacts.push_back(b2b3_cont_spec);

  BoxBoxContactSpecPtr b1b3_cont_spec(new BoxBoxContactSpec);
  b1b3_cont_spec->name = "b1b3_cont";
  b1b3_cont_spec->box1_name = "box1";
  b1b3_cont_spec->box2_name = "box3";
  prob_spec.contacts.push_back(b1b3_cont_spec);

  ///// Initialization /////
  // box1 initial trajectory: straight line to box3 (goes through box2) (along x-axis)
  box1_spec->traj_init.x.col(0) = VectorXd::LinSpaced(
    prob_spec.timesteps,
    box1_spec->state_0.x(0),
    box3_spec->state_0.x(0) - box3_spec->props.half_extents(0) - box1_spec->props.half_extents(0)
  );

  RaveInitialize(false, Level_Debug);
  OR::EnvironmentBasePtr env = RaveCreateEnvironment();
  env->StopSimulation();

  DynamicsProblemPtr prob = CreateDynamicsProblem(env, prob_spec);
  DynamicsOptResultPtr result = OptimizeDynamicsProblem(prob, true);
  vector<double> &soln = result->optimizer->x();

  BoxPtr box1 = boost::dynamic_pointer_cast<Box>(prob->findObject("box1"));
  BoxPtr box2 = boost::dynamic_pointer_cast<Box>(prob->findObject("box2"));
  BoxPtr box3 = boost::dynamic_pointer_cast<Box>(prob->findObject("box3"));

  vector<BoxPtr> boxes{box1, box2, box3};
  for (BoxPtr &b : boxes) {
    cout << "box " << b->getName() << " x:\n" << getTraj(soln, b->m_trajvars.x) << endl;
    if (!b->m_spec.is_kinematic) {
      cout << "box " << b->getName() << " v:\n" << getTraj(soln, b->m_trajvars.v) << endl;
      cout << "box " << b->getName() << " force:\n" << getTraj(soln, b->m_trajvars.force) << endl;
    }
  }
//  cout << "q:\n" << getTraj(soln, box->m_trajvars.q) << endl;
//  cout << "w:\n" << getTraj(soln, box->m_trajvars.w) << endl;
//  cout << "T:\n" << getTraj(soln, box->m_trajvars.torque) << endl;

  BoxBoxContactPtr b1b2_cont = boost::dynamic_pointer_cast<BoxBoxContact>(prob->findContact("b1b2_cont"));
  BoxBoxContactPtr b2b3_cont = boost::dynamic_pointer_cast<BoxBoxContact>(prob->findContact("b2b3_cont"));
  BoxBoxContactPtr b1b3_cont = boost::dynamic_pointer_cast<BoxBoxContact>(prob->findContact("b1b3_cont"));

  vector<BoxBoxContactPtr> contacts{b1b2_cont, b1b3_cont, b2b3_cont};
  for (BoxBoxContactPtr &c : contacts) {
    cout << "contact " << c->getName() << " f:\n" << getTraj(soln, c->m_trajvars.f) << endl;
    cout << "contact " << c->getName() << " p1:\n" << getTraj(soln, c->m_trajvars.p1) << endl;
    cout << "contact " << c->getName() << " p2:\n" << getTraj(soln, c->m_trajvars.p2) << endl;
  }
//  cout << "gp:\n" << getTraj(soln, box_ground_cont->m_trajvars.p) << endl;
//  cout << "gf:\n" << getTraj(soln, box_ground_cont->m_trajvars.f) << endl;

//  boost::thread run_traj(RunTraj, result, env, *rad);


  env.reset();
  RaveDestroy();

  return 0;
}
