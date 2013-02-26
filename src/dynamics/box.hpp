#pragma once

#include "dynamics_problem.hpp"
#include "sco/modeling_utils.hpp"

namespace trajopt {
namespace dynamics {

using namespace std;
using namespace Eigen;


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


} // namespace dynamics
} // namespace trajopt
