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
};

struct BoxStateTraj {
  MatrixX3d x, v, force;
  MatrixX4d q;
  MatrixX3d w, torque;

  BoxStateTraj(int timesteps) : x(MatrixX3d::Zero(timesteps, 3)), v(MatrixX3d::Zero(timesteps, 3)), force(MatrixX3d::Zero(timesteps, 3)), q(MatrixX4d::Zero(timesteps, 4)), w(MatrixX3d::Zero(timesteps, 3)), torque(MatrixX3d::Zero(timesteps, 3)) {
    q.col(3) = VectorXd::Ones(timesteps);
  }

  static BoxStateTraj FromConstant(int timesteps, const BoxState &s) {
    BoxStateTraj traj(timesteps);
    for (int t = 0; t < timesteps; ++t) {
      traj.x.row(t) = s.x;
      traj.v.row(t) = s.v;
      traj.force.row(t) = s.force;
      traj.q.row(t) = s.q.coeffs();
      traj.w.row(t) = s.w;
      traj.torque.row(t) = s.torque;
    }
    return traj;
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

class Box;
typedef boost::shared_ptr<Box> BoxPtr;

struct BoxSpec : public Spec {
  string name;
  BoxProperties props;
  BoxState state_0;
  BoxStateTraj traj_init;
  bool is_kinematic;

  BoxSpec(const ProblemSpec &prob_spec) : traj_init(prob_spec.timesteps) { }
  OptimizationBasePtr realize(DynamicsProblem *prob);
};
typedef boost::shared_ptr<BoxSpec> BoxSpecPtr;

class Box : public DynamicsObject {
public:
  // constants
  BoxSpec m_spec;

  // traj variables for optimization
  BoxStateTrajVars m_trajvars;

  OR::KinBodyPtr m_kinbody;

  Box(const BoxSpec &spec, DynamicsProblem *prob);
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

struct GroundSpec : public Spec {
  string name;
  double z;

  OptimizationBasePtr realize(DynamicsProblem *prob);
};
typedef boost::shared_ptr<GroundSpec> GroundSpecPtr;

class Ground : public DynamicsObject {
public:
  GroundSpec m_spec;
  OR::KinBodyPtr m_kinbody;

  Ground(const GroundSpec &spec, DynamicsProblem *prob) : m_spec(spec), DynamicsObject(spec.name, prob) { }
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
