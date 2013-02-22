#pragma once

#include "dynamics_problem.hpp"
#include "ipi/sco/modeling_utils.hpp"

namespace trajopt {
namespace dynamics {

using namespace std;
using namespace Eigen;

// box state at single timestep
struct BoxState {
  Vector3d x, p, force;
  Quaterniond r; Vector3d L, torque;

  BoxState() : x(Vector3d::Zero()), p(Vector3d::Zero()), force(Vector3d::Zero()), r(Quaterniond::Identity()), L(Vector3d::Zero()), torque(Vector3d::Zero()) { }

  static inline int Dim() { return 19; }

  VectorXd toVec() const {
    Eigen::Matrix<double, 19, 1> vec;
    vec.block<3, 1>(0, 0) = x;
    vec.block<3, 1>(3, 0) = p;
    vec.block<3, 1>(6, 0) = force;
    vec.block<4, 1>(9, 0) = quatToVec(r);
    vec.block<3, 1>(13, 0) = L;
    vec.block<3, 1>(16, 0) = torque;
    return vec;
  }

  static BoxState FromVec(const VectorXd &vec) {
    assert(vec.size() == Dim());
    BoxState bs;
    bs.x = vec.block<3, 1>(0, 0);
    bs.p = vec.block<3, 1>(3, 0);
    bs.force = vec.block<3, 1>(6, 0);
    bs.r = toQuat(vec.block<4, 1>(9, 0));
    bs.L = vec.block<3, 1>(13, 0);
    bs.torque = vec.block<3, 1>(16, 0);
    return bs;
  }
};

struct BoxStateTrajVars {
  VarArray x, p, force;
  VarArray r; VarArray L, torque;

  BoxStateTrajVars(int timesteps) {
    x.resize(timesteps, 3);
    p.resize(timesteps, 3);
    force.resize(timesteps, 3);
    r.resize(timesteps, 4);
    L.resize(timesteps, 3);
    torque.resize(timesteps, 3);
    assert(timesteps*BoxState::Dim() == x.size() + p.size() + force.size() + r.size() + L.size() + torque.size());
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
  int setVariables(const vector<Var> &vars, int start_pos);
  void addConstraintsToModel();

  AffExpr getForceExpr(int t, int i);
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

  vector<ContactPtr> m_contacts;
  // ground hack for now
//  vector<BoxGroundContactPtr> m_ground_conts;
//  void registerGroundContact() {
//    BoxGroundContactPtr ctv(new BoxGroundContact(this));
//    m_ground_conts.push_back(ctv);
//  }
//  void addGroundNonpenetrationCnts(double ground_z);
  // end ground hack

  void fillVarNamesAndBounds(vector<string> &out_names, vector<double> &out_vlower, vector<double> &out_vupper, const string &name_prefix="box");
  int setVariables(const vector<Var> &vars, int start_pos);
  void addConstraintsToModel();
  void addToRave();

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
  int setVariables(const vector<Var> &vars, int start_pos) { return start_pos; }
  void addConstraintsToModel() { }
  void addToRave();
};
typedef boost::shared_ptr<Ground> GroundPtr;

// nonpenetration constraint for a single timestep--lowest point on box must have z-value >= ground z-value
class BoxGroundConstraint;
struct BoxGroundConstraintErrCalc : public VectorOfVector {
  BoxGroundConstraint *m_cnt;
  BoxGroundConstraintErrCalc(BoxGroundConstraint *cnt) : m_cnt(cnt) { }
  VectorXd operator()(const VectorXd &vals) const;
  static VarVector buildVarVector(Box *box, int t);
};
class BoxGroundConstraint : public ConstraintFromNumDiff {
public:
  DynamicsProblem *m_prob;
  Box *m_box;
  double m_ground_z;
  int m_t;

  BoxGroundConstraint(DynamicsProblem *prob, double ground_z, Box *box, int t, const string &name_prefix="box_ground")
    : m_prob(prob), m_ground_z(ground_z), m_box(box), m_t(t),
      ConstraintFromNumDiff(
        VectorOfVectorPtr(new BoxGroundConstraintErrCalc(this)),
        BoxGroundConstraintErrCalc::buildVarVector(box, t),
        INEQ,
        (boost::format("%s_%d") % name_prefix % t).str())
  { }
};
typedef boost::shared_ptr<BoxGroundConstraint> BoxGroundConstraintPtr;



} // namespace dynamics
} // namespace trajopt
