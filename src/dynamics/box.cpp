#include "box.hpp"

#include "trajopt/utils.hpp"
#include "ipi/sco/expr_vec_ops.hpp"

namespace trajopt {
namespace dynamics {


Box::Box(const string &name, DynamicsProblem *prob, const BoxProperties &props, const BoxState &init_state) :
  m_prob(prob), m_props(props), m_trajvars(prob->m_timesteps), m_init_state(init_state), DynamicsObject(name) { }

void Box::fillVarNamesAndBounds(vector<string> &out_names, vector<double> &out_vlower, vector<double> &out_vupper, const string &name_prefix) {
  int init_size = out_names.size();
  for (int t = 0; t < m_prob->m_timesteps; ++t) {
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
  assert(out_names.size() - init_size == m_prob->m_timesteps*BoxState::Dim());
}

void Box::fillInitialSolution(vector<double> &out) {
  int init_size = out.size();
  for (int t = 0; t < m_prob->m_timesteps; ++t) {
    for (int i = 0; i < 3; ++i) {
      out.push_back(m_init_state.x(i));
    }
    for (int i = 0; i < 3; ++i) {
      out.push_back(m_init_state.v(i));
    }
    for (int i = 0; i < 3; ++i) {
      out.push_back(m_init_state.force(i));
    }
    for (int i = 0; i < 4; ++i) {
      out.push_back(m_init_state.q.coeffs()(i));
    }
    for (int i = 0; i < 3; ++i) {
      out.push_back(m_init_state.w(i));
    }
    for (int i = 0; i < 3; ++i) {
      out.push_back(m_init_state.torque(i));
    }
  }
}

int Box::setVariables(const vector<Var> &vars, int start_pos) {
  int k = start_pos;
  for (int t = 0; t < m_prob->m_timesteps; ++t) {
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
  /*
  cout << k - start_pos << endl;
  cout << m_prob->m_timesteps << endl;
  cout << BoxState::Dim() << endl;
  cout << m_ground_conts.size() << endl;
  cout << ContactState::Dim() << endl;*/
  assert(k - start_pos == m_prob->m_timesteps*BoxState::Dim());
  return k;
}

void Box::addConstraintsToModel() {
  ModelPtr model = m_prob->getModel();
  const double dt = m_prob->m_dt;
  // initial conditions
  for (int i = 0; i < 3; ++i) model->addEqCnt(m_trajvars.x(0,i) - m_init_state.x(i), "");
  for (int i = 0; i < 3; ++i) model->addEqCnt(m_trajvars.v(0,i) - m_init_state.v(i), "");
  for (int i = 0; i < 4; ++i) model->addEqCnt(m_trajvars.q(0,i) - m_init_state.q.coeffs()(i), "");
  for (int i = 0; i < 3; ++i) model->addEqCnt(m_trajvars.w(0,i) - m_init_state.w(i), "");
  // integration (implicit euler)
  for (int t = 1; t < m_prob->m_timesteps; ++t) {
    for (int i = 0; i < 3; ++i) model->addEqCnt(m_trajvars.x(t,i) - m_trajvars.x(t-1,i) - dt*m_trajvars.v(t,i), "");
    for (int i = 0; i < 3; ++i) model->addEqCnt(m_trajvars.v(t,i) - m_trajvars.v(t-1,i) - dt/m_props.mass*m_trajvars.force(t,i), "");

    // HACK: fixed rotation
    for (int i = 0; i < 4; ++i) model->addEqCnt(AffExpr(m_trajvars.q(t,i)) - m_init_state.q.coeffs()(i), "");

    for (int i = 0; i < 3; ++i) model->addEqCnt(m_trajvars.w(t,i) - m_trajvars.w(t-1,i) - dt*m_trajvars.torque(t,i), ""); //FIXME: inertia
  }
  // forces
  for (int t = 0; t < m_prob->m_timesteps; ++t) {
    for (int i = 0; i < 3; ++i) {
      AffExpr force_t_i(m_prob->m_gravity(i));
      // force from contacts
      for (Contact *contact : m_contacts) {
        exprInc(force_t_i, contact->getForceExpr(this, t,i));
      }
      model->addEqCnt(AffExpr(m_trajvars.force(t,i)) - force_t_i, "");
    }
  }
  // torques
  /*
  for (int t = 0; t < m_prob->m_timesteps; ++t) {
    for (int i = 0; i < 3; ++i) {
    }
  }
  */

  model->update();
}

void Box::addToRave() {
  m_kinbody = OR::RaveCreateKinBody(m_prob->m_env, "");
  m_kinbody->SetName(getName());
  vector<OR::AABB> aabb;
  aabb.push_back(OR::AABB(OR::Vector(0, 0, 0), toOR(m_props.half_extents)));
  m_kinbody->InitFromBoxes(aabb, true);
  m_prob->m_env->Add(m_kinbody);
  m_kinbody->SetTransform(OR::Transform(toOR(m_init_state.q), toOR(m_init_state.x)));
}

void Box::setRaveState(const vector<double> &x, int t) {
  Vector3d x_val(getVec(x, m_trajvars.x.row(t)));
  Quaterniond q_val(getQuat(x, m_trajvars.q.row(t)));
  m_kinbody->SetTransform(toOR(x_val, q_val));
}


} // namespace dynamics
} // namespace trajopt
