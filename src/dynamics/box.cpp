#include "box.hpp"

#include "trajopt/utils.hpp"
#include "sco/expr_vec_ops.hpp"

namespace trajopt {
namespace dynamics {


OptimizationBasePtr BoxSpec::realize(DynamicsProblem *prob) {
  return BoxPtr(new Box(*this, prob));
}

Box::Box(const BoxSpec &spec, DynamicsProblem *prob)
  : m_spec(spec),
    m_trajvars(prob->m_spec.timesteps),
    DynamicsObject(spec.name, prob)
{ }

void Box::fillVarNamesAndBounds(vector<string> &out_names, vector<double> &out_vlower, vector<double> &out_vupper, const string &name_prefix) {
  int init_size = out_names.size();
  for (int t = 0; t < getProb()->m_spec.timesteps; ++t) {
    for (int i = 0; i < 3; ++i) {
      out_names.push_back((boost::format("%s_x_%i_%i") % name_prefix % t % i).str());
      out_vlower.push_back(-INFINITY);
      out_vupper.push_back(INFINITY);
    }
    for (int i = 0; i < 3; ++i) {
      out_names.push_back((boost::format("%s_v_%i_%i") % name_prefix % t % i).str());
      out_vlower.push_back(-INFINITY);
      out_vupper.push_back(INFINITY);
    }
    for (int i = 0; i < 3; ++i) {
      out_names.push_back((boost::format("%s_force_%i_%i") % name_prefix % t % i).str());
      out_vlower.push_back(-INFINITY);
      out_vupper.push_back(INFINITY);
    }
    for (int i = 0; i < 4; ++i) {
      out_names.push_back((boost::format("%s_q_%i_%i") % name_prefix % t % i).str());
      out_vlower.push_back(-INFINITY);
      out_vupper.push_back(INFINITY);
    }
    for (int i = 0; i < 3; ++i) {
      out_names.push_back((boost::format("%s_w_%i_%i") % name_prefix % t % i).str());
      out_vlower.push_back(-INFINITY);
      out_vupper.push_back(INFINITY);
    }
    for (int i = 0; i < 3; ++i) {
      out_names.push_back((boost::format("%s_torque_%i_%i") % name_prefix % t % i).str());
      out_vlower.push_back(-INFINITY);
      out_vupper.push_back(INFINITY);
    }
  }
  assert(out_names.size() - init_size == getProb()->m_spec.timesteps*BoxState::Dim());
}

void Box::fillInitialSolution(vector<double> &out) {
  assert(m_spec.traj_init.x.rows() >= getProb()->m_spec.timesteps);
  for (int t = 0; t < getProb()->m_spec.timesteps; ++t) {
    for (int i = 0; i < 3; ++i) {
      out.push_back(m_spec.traj_init.x(t,i));
    }
    for (int i = 0; i < 3; ++i) {
      out.push_back(m_spec.traj_init.v(t,i));
    }
    for (int i = 0; i < 3; ++i) {
      out.push_back(m_spec.traj_init.force(t,i));
    }
    for (int i = 0; i < 4; ++i) {
      out.push_back(m_spec.traj_init.q(t,i));
    }
    for (int i = 0; i < 3; ++i) {
      out.push_back(m_spec.traj_init.w(t,i));
    }
    for (int i = 0; i < 3; ++i) {
      out.push_back(m_spec.traj_init.torque(t,i));
    }
  }
}

int Box::setVariables(const vector<Var> &vars, int start_pos) {
  int k = start_pos;
  for (int t = 0; t < getProb()->m_spec.timesteps; ++t) {
    for (int i = 0; i < 3; ++i) {
      m_trajvars.x(t,i) = vars[k++];
    }
    for (int i = 0; i < 3; ++i) {
      m_trajvars.v(t,i) = vars[k++];
    }
    for (int i = 0; i < 3; ++i) {
      m_trajvars.force(t,i) = vars[k++];
    }
    for (int i = 0; i < 4; ++i) {
      m_trajvars.q(t,i) = vars[k++];
    }
    for (int i = 0; i < 3; ++i) {
      m_trajvars.w(t,i) = vars[k++];
    }
    for (int i = 0; i < 3; ++i) {
      m_trajvars.torque(t,i) = vars[k++];
    }
  }
  assert(k - start_pos == getProb()->m_spec.timesteps*BoxState::Dim());
  return k;
}

void Box::addConstraintsToModel() {
  ModelPtr model = getProb()->getModel();
  const double dt = getProb()->m_spec.dt;

  // initial conditions
  for (int i = 0; i < 3; ++i) model->addEqCnt(m_trajvars.x(0,i) - m_spec.state_0.x(i), "");
  if (!m_spec.is_kinematic) for (int i = 0; i < 3; ++i) model->addEqCnt(m_trajvars.v(0,i) - m_spec.state_0.v(i), "");
  for (int i = 0; i < 4; ++i) model->addEqCnt(m_trajvars.q(0,i) - m_spec.state_0.q.coeffs()(i), "");
  if (!m_spec.is_kinematic) for (int i = 0; i < 3; ++i) model->addEqCnt(m_trajvars.w(0,i) - m_spec.state_0.w(i), "");

  if (m_spec.is_kinematic) {
    // HACK: fixed rotation
    for (int t = 1; t < getProb()->m_spec.timesteps; ++t) {
      for (int i = 0; i < 4; ++i) model->addEqCnt(AffExpr(m_trajvars.q(t,i)) - m_spec.state_0.q.coeffs()(i), "");
    }
  } else { // !m_spec.is_kinematic
    // integration (implicit euler)
    for (int t = 1; t < getProb()->m_spec.timesteps; ++t) {
      for (int i = 0; i < 3; ++i) model->addEqCnt(m_trajvars.x(t,i) - m_trajvars.x(t-1,i) - dt*m_trajvars.v(t,i), "");
      for (int i = 0; i < 3; ++i) model->addEqCnt(m_trajvars.v(t,i) - m_trajvars.v(t-1,i) - dt/m_spec.props.mass*m_trajvars.force(t,i), "");

      // HACK: fixed rotation
      for (int i = 0; i < 4; ++i) model->addEqCnt(AffExpr(m_trajvars.q(t,i)) - m_spec.state_0.q.coeffs()(i), "");

      for (int i = 0; i < 3; ++i) model->addEqCnt(m_trajvars.w(t,i) - m_trajvars.w(t-1,i) - dt*m_trajvars.torque(t,i), ""); //FIXME: inertia
    }
    // forces
    for (int t = 0; t < getProb()->m_spec.timesteps; ++t) {
      for (int i = 0; i < 3; ++i) {
        AffExpr force_t_i(getProb()->m_spec.gravity(i));
        // force from contacts
        for (Contact *contact : m_contacts) {
          exprInc(force_t_i, contact->getForceExpr(this, t,i));
        }
        model->addEqCnt(AffExpr(m_trajvars.force(t,i)) - force_t_i, "");
      }
    }
    // torques
    /*
    for (int t = 0; t < getProb()->m_spec.timesteps; ++t) {
      for (int i = 0; i < 3; ++i) {
      }
    }
    */
  }

  model->update();
}

void Box::addToRave() {
  m_kinbody = OR::RaveCreateKinBody(getProb()->m_env, "");
  m_kinbody->SetName(getName());
  vector<OR::AABB> aabb;
  aabb.push_back(OR::AABB(OR::Vector(0, 0, 0), toOR(m_spec.props.half_extents)));
  m_kinbody->InitFromBoxes(aabb, true);
  getProb()->m_env->Add(m_kinbody);
  m_kinbody->SetTransform(OR::Transform(toOR(m_spec.state_0.q), toOR(m_spec.state_0.x)));
}

void Box::setRaveState(const vector<double> &x, int t) {
  Vector3d x_val(getVec(x, m_trajvars.x.row(t)));
  Quaterniond q_val(getQuat(x, m_trajvars.q.row(t)));
  m_kinbody->SetTransform(toOR(x_val, q_val));
}


OptimizationBasePtr GroundSpec::realize(DynamicsProblem *prob) {
  return GroundPtr(new Ground(*this, prob));
}

void Ground::addToRave() {
  m_kinbody = OR::RaveCreateKinBody(getProb()->m_env, "");
  m_kinbody->SetName(getName());
  vector<OR::AABB> aabb;
  aabb.push_back(OR::AABB(OR::Vector(0, 0, m_spec.z - 1), OR::Vector(100, 100, 1)));
  m_kinbody->InitFromBoxes(aabb, true);
  getProb()->m_env->Add(m_kinbody);
}

} // namespace dynamics
} // namespace trajopt
