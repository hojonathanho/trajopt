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
static const double TABLE_HEIGHT = 0.0; // DO NOT CHANGE (complementarity)

typedef Eigen::Matrix<double, 1, 1> Vector1d;
inline Vector1d makeVector1d(double x) { Vector1d v; v(0) = x; return v; }

struct ComplErrCalc : public VectorOfVector {
  vector<ScalarOfVectorPtr> m_terms;
  ComplErrCalc(const vector<ScalarOfVectorPtr> &terms) : m_terms(terms) { }
  VectorXd operator()(const VectorXd &vals) const {
    double x = 1;
    for (int i = 0; i < m_terms.size(); ++i) {
      x *= m_terms[i]->call(vals);
    }
    return makeVector1d(x);
  }
};

struct ComplConstraint : public ConstraintFromNumDiff {
  ComplConstraint(const vector<ScalarOfVectorPtr> &terms, const VarVector &vars, const string &name="ComplConstraint") :
    ConstraintFromNumDiff(VectorOfVectorPtr(new ComplErrCalc(terms)), vars, EQ, name) { }
};

/*
ConstraintPtr MakeBoxGroundConstraint(VarVector &x, VarVector &ground_force, int i, const string &name) {
  int z = i + 2;
  VarVector vars;
  vars.push_back(ground_force(i,2));
  vars.push_back(x(z,2));
  vector<ScalarOfVectorPtr> &terms;
  terms.push_back(ScalarOfVectorPtr());
  terms.push_back(ScalarOfVectorPtr());
  ConstraintPtr cnt(new ComplConstraint(terms, vars, name));
  return cnt;
}*/

/*
struct ComplErrCalc : public VectorOfVector {
  VectorXd operator()(const VectorXd& vals) const {
    //out(0) = vals.norm() - vals.lpNorm<1>();
    return makeVector1d(vals.prod());
  }
};

struct ComplConstraint : public ConstraintFromNumDiff {
  ComplConstraint(const VarVector &vars, const string &name="ComplConstraint") : ConstraintFromNumDiff(VectorOfVectorPtr(new ComplErrCalc()), vars, EQ, name) { }
};*/



void MakeVariablesAndBounds(dynamics::DynamicsProblemPtr prob, const vector<dynamics::BoxState> &init_states, vector<dynamics::BoxPtr> &out_objects) {
  int n_objects = init_states.size();
  vector<double> vlower, vupper;
  vector<string> names;
  out_objects.clear();

  dynamics::BoxProperties props;
  props.mass = 1.0;
  props.halfextents = Vector3d(.5, .5, .5);
  props.Ibody = props.Ibodyinv = Eigen::Matrix3d::Identity();

  for (int i = 0; i < n_objects; ++i) {
    out_objects.push_back(dynamics::BoxPtr(new dynamics::Box(prob, props, init_states[i])));
    out_objects[i]->fillVarNamesAndBounds(names, vlower, vupper);
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

  for (int i = 0; i < n_objects; ++i) {
    out_objects[i]->addConstraintsToProb();
  }
}

class ZeroCost : public Cost {
  ConvexObjectivePtr convex(const DblVec&, Model* model) {
    ConvexObjectivePtr out(new ConvexObjective(model));
    out->addAffExpr(AffExpr());
    return out;
  }
  double value(const DblVec& x) {
    return 0.;
  }
};


int main(int argc, char* argv[]) {
  dynamics::DynamicsProblemPtr prob(new dynamics::DynamicsProblem());
  prob->n_timesteps = 10;
  prob->dt = 1./prob->n_timesteps;
  prob->gravity = Vector3d(0, 0, -9.8);

  //Vector3d init_x(0, 0, 5), init_v(0, 0, 0);
  //VarArray x, v, a, ground_force;
  //DblVec initSoln;

  vector<dynamics::BoxPtr> objects;
  vector<dynamics::BoxState> init_states;
  dynamics::BoxState bs; bs.x = Vector3d(0, 0, 5);
  init_states.push_back(bs);
  MakeVariablesAndBounds(prob, init_states, objects);

  //srand(time(NULL));
  //for (int i = 0; i < initSoln.size(); ++i) initSoln[i] += 0.01*((double)rand()/RAND_MAX-.5);
  //for (int i = 0; i < initSoln.size(); ++i) cout << initSoln[i] << ' '; cout << endl;
  prob->addCost(CostPtr(new ZeroCost())); // shut up
  BasicTrustRegionSQP optimizer(prob);
  optimizer.min_trust_box_size_ = 1e-7;
  optimizer.min_approx_improve_= 1e-7;
  optimizer.cnt_tolerance_ = 1e-7;
  optimizer.trust_box_size_ = 1;
  optimizer.max_iter_ = 1000;

  optimizer.initialize(DblVec(prob->getVars().size(), 0));
  OptStatus status = optimizer.optimize();
  cout << "x:\n" << getTraj(optimizer.x(), objects[0]->m_trajvars.x) << endl;
  cout << "v:\n" << getTraj(optimizer.x(), objects[0]->m_trajvars.p) << endl;
  cout << "a:\n" << getTraj(optimizer.x(), objects[0]->m_trajvars.force) << endl;
  //cout << "gf:\n" << getTraj(optimizer.x(), ground_force) << endl;

//  boost::thread run_traj(RunTraj, result, env, *rad);
//  ViewerBasePtr viewer = RaveCreateViewer(env,"qtosg");
//  viewer->main();

  return 0;
}
