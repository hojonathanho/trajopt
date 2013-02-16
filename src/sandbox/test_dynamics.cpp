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
using namespace ipi::sco;
using namespace trajopt;
using namespace util;
using namespace OpenRAVE;
using namespace std;

static const double GRAVITY = -9.8;
static const double TABLE_HEIGHT = 0.0; // DO NOT CHANGE (complementarity)

typedef Eigen::Matrix<double, 1, 1> Vector1d;
inline Vector1d makeVector1d(double x) { Vector1d v; v(0) = x; return v; }

/*
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


struct ComplErrCalc : public VectorOfVector {
  VectorXd operator()(const VectorXd& vals) const {
    //out(0) = vals.norm() - vals.lpNorm<1>();
    return makeVector1d(vals.prod());
  }
};

struct ComplConstraint : public ConstraintFromNumDiff {
  ComplConstraint(const VarVector &vars, const string &name="ComplConstraint") : ConstraintFromNumDiff(VectorOfVectorPtr(new ComplErrCalc()), vars, EQ, name) { }
};


void RunTraj(const TrajArray& traj, EnvironmentBasePtr env, RobotAndDOF& rad) {
  for (int i=0; i < traj.rows(); ++i) {
    {
      EnvironmentMutex::scoped_lock lock(env->GetMutex());
      rad.SetDOFValues(toDblVec(traj.row(i)));
    }
    cin.get();
  }
}

void SetViewer(EnvironmentBasePtr penv, const string& viewername) {
  ViewerBasePtr viewer = RaveCreateViewer(penv,viewername);
  BOOST_ASSERT(!!viewer);
  // attach it to the environment:
  penv->Add(viewer);
  // finally call the viewer's infinite loop (this is why a separate thread is needed)
  bool showgui = true;
  viewer->main(showgui);
}

void MakeVariablesAndBounds(int n_steps, double dt, OptProb& prob_out, VarArray &x, VarArray &v, VarArray &a, VarArray &ground_force, const Vector3d &init_x, const Vector3d &init_v, DblVec &out_initSoln) {
  //const int n_dof = 6; // pos rot
  const int n_dof = 3; // x y z

  ModelPtr model = prob_out.getModel();

  vector<double> vlower, vupper;
  vector<string> names;
  out_initSoln.clear();
  // object dof variables
  for (int t = 0; t < n_steps; ++t) {
    for (int i = 0; i < n_dof; ++i) {
      names.push_back((boost::format("x_%i_%i") % t % i).str());
      vlower.push_back(-INFINITY);
      vupper.push_back(INFINITY);
      out_initSoln.push_back(init_x(i));

      names.push_back((boost::format("v_%i_%i") % t % i).str());
      vlower.push_back(-INFINITY);
      vupper.push_back(INFINITY);
      out_initSoln.push_back(init_v(i));

      names.push_back((boost::format("a_%i_%i") % t % i).str());
      vlower.push_back(-INFINITY);
      vupper.push_back(INFINITY);
      out_initSoln.push_back(i == 2 ? GRAVITY : 0);
    }
  }
  // forces
  for (int t = 0; t < n_steps; ++t) {
    for (int i = 0; i < 3; ++i) {
      names.push_back((boost::format("gf_%i_%i") % t % i).str());
      vlower.push_back(-INFINITY);
      vupper.push_back(INFINITY);
      out_initSoln.push_back(0);
    }
  }
  prob_out.createVariables(names, vlower, vupper);
  assert(out_initSoln.size() == prob_out.getVars().size());

  cout << Str(names) << endl;

  x.resize(n_steps, n_dof);
  v.resize(n_steps, n_dof);
  a.resize(n_steps, n_dof);
  ground_force.resize(n_steps, 3);

  vector<Var> vars = prob_out.getVars();
  assert(vars.size() == names.size());
  int k = 0;
  for (int i=0; i < n_steps; ++i) {
    for (int j=0; j < n_dof; ++j) {
      x(i,j) = vars[k++];
      v(i,j) = vars[k++];
      a(i,j) = vars[k++];
    }
  }
  for (int t = 0; t < n_steps; ++t) {
    for (int i = 0; i < 3; ++i) {
      ground_force(t,i) = vars[k++];
    }
  }

  // integration steps
  for (int i=0; i < n_steps; ++i) {
    for (int j=0; j < n_dof; ++j) {
      if (i==0) {
        model->addEqCnt(x(0,j) - init_x(j), "");
        model->addEqCnt(v(0,j) - init_v(j), "");
      } else {
        model->addEqCnt(x(i,j) - x(i-1,j) - dt*v(i,j), "");
        model->addEqCnt(v(i,j) - v(i-1,j) - dt*a(i-1,j), "");
      }
    }
  }

  // F = ma
  for (int i=0; i < n_steps; ++i) {
    for (int j=0; j < n_dof; ++j) {
      AffExpr total_force(ground_force(i,j));
      if (j == 2) {
        total_force.constant += GRAVITY;
      }
      model->addEqCnt(AffExpr(a(i,j)) - total_force, "");
    }
  }

  // table constraint
  for (int i=0; i < n_steps; ++i) {
    AffExpr exp(-x(i,2));
    exp.constant = TABLE_HEIGHT;
    model->addIneqCnt(exp, "");
  }

  // ground force direction constraint
  for (int i = 0; i < n_steps; ++i) {
    for (int j = 0; j < 2; ++j) { // kinda dumb
      model->addEqCnt(AffExpr(ground_force(i,j)), "");
    }
    model->addIneqCnt(AffExpr(-ground_force(i,2)), "");
  }

  model->update();

  // TODO: convert model constrs to prob constrs

  // ground force complementarity
  for (int i = 0; i < n_steps-2 /* last unnecessary */; ++i) {
    int z = i+2;
    VarVector vec;
    vec.push_back(ground_force(i,2));
    vec.push_back(x(z,2));
    ConstraintPtr cnt(new ComplConstraint(vec, (boost::format("compl_%d_%d") % i % z).str()));
    prob_out.addConstr(cnt);
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
  OptProbPtr prob(new OptProb());
  int n_steps = 100;
  double dt = 10.0/n_steps;
  Vector3d init_x(0, 0, 5), init_v(0, 0, 0);
  VarArray x, v, a, ground_force;
  DblVec initSoln;
  MakeVariablesAndBounds(n_steps, dt, *prob, x, v, a, ground_force, init_x, init_v, initSoln);
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

  optimizer.initialize(initSoln);
  OptStatus status = optimizer.optimize();
  cout << "x:\n" << getTraj(optimizer.x(), x) << endl;
  cout << "v:\n" << getTraj(optimizer.x(), v) << endl;
  cout << "a:\n" << getTraj(optimizer.x(), a) << endl;
  cout << "gf:\n" << getTraj(optimizer.x(), ground_force) << endl;

//  boost::thread run_traj(RunTraj, result, env, *rad);
//  ViewerBasePtr viewer = RaveCreateViewer(env,"qtosg");
//  viewer->main();

  return 0;
}
