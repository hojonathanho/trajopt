/*
 * box.h
 *
 *  Created on: Feb 16, 2013
 *      Author: jonathan
 */

#ifndef BOX_H_
#define BOX_H_

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "trajopt/typedefs.hpp"

namespace trajopt {
namespace dynamics {

using namespace Eigen;

class DynamicsProblem : public OptProb {
public:
  double dt;
  int n_timesteps;
  Vector3d gravity;
};
typedef boost::shared_ptr<DynamicsProblem> DynamicsProblemPtr;

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
    vec.block<3, 1>(9, 0) = r.vec();
    vec(12, 0) = r.w();
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
    bs.r.vec() = vec.block<3, 1>(9, 0);
    bs.r.w() = vec(12, 0);
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
  Vector3d halfextents;
  Matrix3d Ibody;
  Matrix3d Ibodyinv;
};

class Box {
public:
  // constants
  BoxProperties m_props;

  // traj variables for optimization
  BoxStateTrajVars m_trajvars;
  DynamicsProblemPtr m_prob;
  BoxState m_init_state;

  Box(DynamicsProblemPtr prob, const BoxProperties &props, const BoxState &init_state) : m_prob(prob), m_props(props), m_trajvars(prob->n_timesteps), m_init_state(init_state) { }

  void fillVarNamesAndBounds(vector<string> &out_names, vector<double> &out_vlower, vector<double> &out_vupper, const string &name_prefix="box") {
    int init_size = out_names.size();
    for (int t = 0; t < m_prob->n_timesteps; ++t) {
      for (int i = 0; i < 3; ++i) {
        out_names.push_back((boost::format("%s_x_%i_%i") % name_prefix % t % i).str());
        out_vlower.push_back(-INFINITY);
        out_vupper.push_back(INFINITY);
      }
      for (int i = 0; i < 3; ++i) {
        out_names.push_back((boost::format("%s_p_%i_%i") % name_prefix % t % i).str());
        out_vlower.push_back(-INFINITY);
        out_vupper.push_back(INFINITY);
      }
      for (int i = 0; i < 3; ++i) {
        out_names.push_back((boost::format("%s_force_%i_%i") % name_prefix % t % i).str());
        out_vlower.push_back(-INFINITY);
        out_vupper.push_back(INFINITY);
      }
      for (int i = 0; i < 4; ++i) {
        out_names.push_back((boost::format("%s_r_%i_%i") % name_prefix % t % i).str());
        out_vlower.push_back(-INFINITY);
        out_vupper.push_back(INFINITY);
      }
      for (int i = 0; i < 3; ++i) {
        out_names.push_back((boost::format("%s_L_%i_%i") % name_prefix % t % i).str());
        out_vlower.push_back(-INFINITY);
        out_vupper.push_back(INFINITY);
      }
      for (int i = 0; i < 3; ++i) {
        out_names.push_back((boost::format("%s_torque_%i_%i") % name_prefix % t % i).str());
        out_vlower.push_back(-INFINITY);
        out_vupper.push_back(INFINITY);
      }
    }
    assert(out_names.size() - init_size == m_prob->n_timesteps*BoxState::Dim());
  }

  int setVariables(const vector<Var> &vars, int start_pos) {
    int k = start_pos;
    for (int t = 0; t < m_prob->n_timesteps; ++t) {
      for (int i = 0; i < 3; ++i) {
        m_trajvars.x(t,i) = vars[k++];
      }
      for (int i = 0; i < 3; ++i) {
        m_trajvars.p(t,i) = vars[k++];
      }
      for (int i = 0; i < 3; ++i) {
        m_trajvars.force(t,i) = vars[k++];
      }
      for (int i = 0; i < 4; ++i) {
        m_trajvars.r(t,i) = vars[k++];
      }
      for (int i = 0; i < 3; ++i) {
        m_trajvars.L(t,i) = vars[k++];
      }
      for (int i = 0; i < 3; ++i) {
        m_trajvars.torque(t,i) = vars[k++];
      }
    }
    assert(k - start_pos == m_prob->n_timesteps*BoxState::Dim());
    return k;
  }

  void addConstraintsToProb() {
    ModelPtr model = m_prob->getModel();
    double dt = m_prob->dt;
    // initial conditions
    for (int i = 0; i < 3; ++i) model->addEqCnt(m_trajvars.x(0,i) - m_init_state.x(i), "");
    for (int i = 0; i < 3; ++i) model->addEqCnt(m_trajvars.p(0,i) - m_init_state.p(i), "");
    for (int i = 0; i < 3; ++i) model->addEqCnt(m_trajvars.r(0,i) - m_init_state.r.vec()(i), "");
    model->addEqCnt(m_trajvars.r(0,3) - m_init_state.r.w(), "");
    for (int i = 0; i < 3; ++i) model->addEqCnt(m_trajvars.L(0,i) - m_init_state.L(i), "");
    // integration
    for (int t = 1; t < m_prob->n_timesteps; ++t) {
      for (int i = 0; i < 3; ++i) model->addEqCnt(m_trajvars.x(t,i) - m_trajvars.x(t-1,i) - dt/m_props.mass*m_trajvars.p(t,i), "");
      for (int i = 0; i < 3; ++i) model->addEqCnt(m_trajvars.p(t,i) - m_trajvars.p(t-1,i) - dt*m_trajvars.force(t-1,i), "");
      // TODO: torque
    }
    // forces
    for (int t = 0; t < m_prob->n_timesteps; ++t) {
      for (int i = 0; i < 3; ++i) {
        AffExpr force_i;
        force_i.constant += m_prob->gravity(i);
        // TODO: add other forces here
        model->addEqCnt(AffExpr(m_trajvars.force(t,i)) - force_i, "");
      }
    }
    // torques
    /*
    for (int t = 0; t < m_prob->n_timesteps; ++t) {
      for (int i = 0; i < 3; ++i) {
      }
    }
    */

    model->update();
  }

private:
};

typedef boost::shared_ptr<Box> BoxPtr;

} // namespace dynamics
} // namespace trajopt


#endif /* BOX_H_ */
