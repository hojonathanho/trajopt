#pragma once

#include "dynamics_problem.hpp"
#include "ipi/sco/modeling_utils.hpp"

namespace trajopt {
namespace dynamics {

using namespace std;
using namespace Eigen;


//class QuatIntegrationConstraint : public ConstraintFromNumDiff { // TODO: analytical linearization
//
//  struct ErrCalc : public VectorOfVector {
//    QuatIntegrationConstraint *m_cnt;
//    ErrCalc(QuatIntegrationConstraint *cnt) : m_cnt(cnt) { }
//
//    VectorXd operator()(const VectorXd &vals) const {
//      assert(vals.size() == 4 + 4 + 3);
//      Quaterniond q1(toQuat(vals.block<4,1>(0,0)));
//      Quaterniond q0(toQuat(vals.block<4,1>(4,0)));
//      Vector3d w(vals.block<3,1>(8,0));
//      Quaterniond expected_q1(q0*propagatorQuat(w, m_cnt->m_dt));
//      return quatToVec(expected_q1) - quatToVec(q1);
//    }
//
//    static VarVector buildVarVector(const VarVector &qvars1, const VarVector &qvars0, const VarVector &wvars) {
//      assert(qvars0.size() == 4 && qvars1.size() == 4 && wvars.size() == 3);
//      VarVector v;
//      for (int i = 0; i < 4; ++i) v.push_back(qvars1[i]);
//      for (int i = 0; i < 4; ++i) v.push_back(qvars0[i]);
//      for (int i = 0; i < 3; ++i) v.push_back(wvars[i]);
//      return v;
//    }
//  };
//
//public:
//  const double m_dt;
//
//  QuatIntegrationConstraint(double dt, const VarVector &qvars1, const VarVector &qvars0, const VarVector &wvars, const string &name_prefix)
//    : m_dt(dt),
//      ConstraintFromNumDiff(
//        VectorOfVectorPtr(new ErrCalc(this)),
//        ErrCalc::buildVarVector(qvars1, qvars0, wvars),
//        EQ,
//        name_prefix)
//  { }
//};


// box state at single timestep
struct BoxState {
  Vector3d x, v, force;
  Quaterniond q; Vector3d w, torque;

  BoxState() : x(Vector3d::Zero()), v(Vector3d::Zero()), force(Vector3d::Zero()), q(Quaterniond::Identity()), w(Vector3d::Zero()), torque(Vector3d::Zero()) { }

  static inline int Dim() { return 19; }

  VectorXd toVec() const {
    Eigen::Matrix<double, 19, 1> vec;
    vec.block<3, 1>(0, 0) = x;
    vec.block<3, 1>(3, 0) = v;
    vec.block<3, 1>(6, 0) = force;
    vec.block<4, 1>(9, 0) = q.coeffs();
    vec.block<3, 1>(13, 0) = w;
    vec.block<3, 1>(16, 0) = torque;
    return vec;
  }

  static BoxState FromVec(const VectorXd &vec) {
    assert(vec.size() == Dim());
    BoxState bs;
    bs.x = vec.block<3, 1>(0, 0);
    bs.v = vec.block<3, 1>(3, 0);
    bs.force = vec.block<3, 1>(6, 0);
    bs.q = Quaterniond(vec.block<4, 1>(9, 0));
    bs.w = vec.block<3, 1>(13, 0);
    bs.torque = vec.block<3, 1>(16, 0);
    return bs;
  }
};

struct BoxStateTrajVars {
  VarArray x, v, force;
  VarArray q; VarArray w, torque;

  BoxStateTrajVars(int timesteps) {
    x.resize(timesteps, 3);
    v.resize(timesteps, 3);
    force.resize(timesteps, 3);
    q.resize(timesteps, 4);
    w.resize(timesteps, 3);
    torque.resize(timesteps, 3);
    assert(timesteps*BoxState::Dim() == x.size() + v.size() + force.size() + q.size() + w.size() + torque.size());
  }
};

struct BoxProperties {
  double mass;
  Vector3d half_extents;
  Matrix3d I_body;
  Matrix3d I_body_inv;
};

struct BoxGroundContactTrajVars {
  VarArray p; // contact point (local frame)
  VarArray f; // contact force (world frame)

  BoxGroundContactTrajVars(int timesteps) {
    p.resize(timesteps, 3);
    f.resize(timesteps, 3);
  }
};
//typedef boost::shared_ptr<BoxGroundContactTrajVars> BoxGroundContactTrajVarsPtr;


// box state at single timestep
struct ContactState {
  static inline int Dim() { return 6; }
};
struct Box; struct Ground;
struct BoxGroundContact : public Contact {
  Box *m_box;
  Ground *m_ground;
  BoxGroundContactTrajVars m_trajvars;

  BoxGroundContact(const string &name, Box *box, Ground *ground);

  void fillVarNamesAndBounds(vector<string> &out_names, vector<double> &out_vlower, vector<double> &out_vupper, const string &name_prefix);
  void fillInitialSolution(vector<double> &out);
  int setVariables(const vector<Var> &vars, int start_pos);
  void addConstraintsToModel();

  AffExpr getForceExpr(int t, int i);
  vector<DynamicsObject*> getAffectedObjects();
};
typedef boost::shared_ptr<BoxGroundContact> BoxGroundContactPtr;


class Box : public DynamicsObject {
public:
  // constants
  BoxProperties m_props;

  DynamicsProblem *m_prob;

  // traj variables for optimization
  BoxStateTrajVars m_trajvars;
  BoxState m_init_state;

  OR::KinBodyPtr m_kinbody;

  Box(const string &name, DynamicsProblem *prob, const BoxProperties &props, const BoxState &init_state);
  virtual ~Box() { }

  vector<Contact*> m_contacts;

  void fillVarNamesAndBounds(vector<string> &out_names, vector<double> &out_vlower, vector<double> &out_vupper, const string &name_prefix="box");
  void fillInitialSolution(vector<double> &out);
  int setVariables(const vector<Var> &vars, int start_pos);
  void addConstraintsToModel();

  void registerContact(Contact *c) { m_contacts.push_back(c); }
  void addToRave();
  void setRaveState(const vector<double> &x, int t);

private:
};
typedef boost::shared_ptr<Box> BoxPtr;

class Ground : public DynamicsObject {
public:
  double m_z;
  DynamicsProblem *m_prob;
  OR::KinBodyPtr m_kinbody;

  Ground(const string &name, DynamicsProblem *prob, double z) : m_prob(prob), m_z(z), DynamicsObject(name) { }
  virtual ~Ground() { }

  void fillVarNamesAndBounds(vector<string> &out_names, vector<double> &out_vlower, vector<double> &out_vupper, const string &name_prefix) { }
  void fillInitialSolution(vector<double> &out) { }
  int setVariables(const vector<Var> &vars, int start_pos) { return start_pos; }
  void addConstraintsToModel() { }
  void addToRave();
  void setRaveState(const vector<double> &, int) { }
  void registerContact(Contact *) { }
};
typedef boost::shared_ptr<Ground> GroundPtr;

// nonpenetration constraint for a single timestep--lowest point on box must have z-value >= ground z-value
class BoxGroundConstraintND;
struct BoxGroundConstraintNDErrCalc : public VectorOfVector {
  BoxGroundConstraintND *m_cnt;
  BoxGroundConstraintNDErrCalc(BoxGroundConstraintND *cnt) : m_cnt(cnt) { }
  VectorXd operator()(const VectorXd &vals) const;
  static VarVector buildVarVector(Box *box, int t);
};
class BoxGroundConstraintND : public ConstraintFromNumDiff {
public:
  DynamicsProblem *m_prob;
  Box *m_box;
  Ground *m_ground;
  int m_t;

  BoxGroundConstraintND(DynamicsProblem *prob, Box *box, Ground *ground, int t, const string &name_prefix="box_ground")
    : m_prob(prob), m_box(box), m_ground(ground), m_t(t),
      ConstraintFromNumDiff(
        VectorOfVectorPtr(new BoxGroundConstraintNDErrCalc(this)),
        BoxGroundConstraintNDErrCalc::buildVarVector(box, t),
        INEQ,
        (boost::format("%s_%d") % name_prefix % t).str())
  { }
};
typedef boost::shared_ptr<BoxGroundConstraintND> BoxGroundConstraintNDPtr;


class BoxGroundConstraint : public Constraint {
public:
  DynamicsProblem *m_prob;
  Box *m_box;
  Ground *m_ground;
  int m_t;

  BoxGroundConstraint(DynamicsProblem *prob, Box *box, Ground *ground, int t, const string &name_prefix="box_ground");
  virtual ~BoxGroundConstraint() {}

  ConstraintType type() {return INEQ;}
  virtual vector<double> value(const vector<double>& x);
  virtual ConvexConstraintsPtr convex(const vector<double>& x, Model* model);

};


} // namespace dynamics
} // namespace trajopt
