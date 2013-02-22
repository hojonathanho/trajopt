#include "box.h"

namespace trajopt {
namespace dynamics {

BoxGroundContact::BoxGroundContact(const string &name, Box *box, Ground *ground) :
    m_box(box), m_ground(ground), m_trajvars(box->m_prob->m_timesteps), Contact(name) {
  assert(m_box->m_prob == m_ground->m_prob);
}

void BoxGroundContact::fillVarNamesAndBounds(vector<string> &out_names, vector<double> &out_vlower, vector<double> &out_vupper, const string &name_prefix) {
  for (int t = 0; t < m_box->m_prob->m_timesteps; ++t) {
    for (int i = 0; i < 3; ++i) {
      out_names.push_back((boost::format("%s_p_%i_%i") % name_prefix % t % i).str());
      out_vlower.push_back(-INFINITY);
      out_vupper.push_back(INFINITY);
    }
    for (int i = 0; i < 3; ++i) {
      out_names.push_back((boost::format("%s_f_%i_%i") % name_prefix % t % i).str());
      out_vlower.push_back(-INFINITY);
      out_vupper.push_back(INFINITY);
    }
  }
}

void BoxGroundContact::fillInitialSolution(vector<double> &out) {
  for (int t = 0; t < m_box->m_prob->m_timesteps; ++t) {
    for (int i = 0; i < 3; ++i) {
      out.push_back(0);
    }
    for (int i = 0; i < 3; ++i) {
      out.push_back(0);
    }
  }
}

int BoxGroundContact::setVariables(const vector<Var> &vars, int start_pos) {
  int k = start_pos;
  for (int t = 0; t < m_box->m_prob->m_timesteps; ++t) {
    for (int i = 0; i < 3; ++i) {
      m_trajvars.p(t,i) = vars[k++];
    }
    for (int i = 0; i < 3; ++i) {
      m_trajvars.f(t,i) = vars[k++];
    }
  }
  assert(k - start_pos == m_box->m_prob->m_timesteps*ContactState::Dim());
  return k;
}

void BoxGroundContact::addConstraintsToModel() {
  DynamicsProblem *prob = m_box->m_prob;
  ModelPtr model = prob->getModel();

  // box cannot penetrate ground
  for (int t = 0; t < prob->m_timesteps; ++t) {
    prob->addConstr(ConstraintPtr(new BoxGroundConstraint(prob, m_box, m_ground, t)));
  }

  // contact origin point must stay inside box (in local coords)
  for (int t = 0; t < prob->m_timesteps; ++t) {
    for (int i = 0; i < 3; ++i) {
      model->addIneqCnt(m_trajvars.p(t,i) - m_box->m_props.half_extents(i), "");
      model->addIneqCnt(-m_trajvars.p(t,i) - m_box->m_props.half_extents(i), "");
    }
  }

  // FIXME!!!!!!!!!!!
  // contact force must have z-component >= 0
  for (int t = 0; t < prob->m_timesteps; ++t) {
    model->addIneqCnt(-m_trajvars.f(t,2), "");
  }

  model->update();
}

AffExpr BoxGroundContact::getForceExpr(int t, int i) {
  return AffExpr(m_trajvars.f(t,i));
}

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

//  for (int z = 0; z < m_ground_conts.size(); ++z) {
//    m_ground_conts[z]->fillVarNamesAndBounds(out_names, out_vlower, out_vupper, (boost::format("%s_contact_%d") % name_prefix % z).str());
//  }

//  assert(out_names.size() - init_size == m_prob->m_timesteps*(BoxState::Dim() + m_ground_conts.size()*ContactState::Dim()));
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
      out.push_back(quatToVec(m_init_state.q)(i));
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
//  for (int z = 0; z < m_ground_conts.size(); ++z) {
//    k = m_ground_conts[z]->setVariables(vars, k);
//  }
  /*
  cout << k - start_pos << endl;
  cout << m_prob->m_timesteps << endl;
  cout << BoxState::Dim() << endl;
  cout << m_ground_conts.size() << endl;
  cout << ContactState::Dim() << endl;*/
//  assert(k - start_pos == m_prob->m_timesteps*(BoxState::Dim() + m_ground_conts.size()*ContactState::Dim()));
  assert(k - start_pos == m_prob->m_timesteps*BoxState::Dim());
  return k;
}

void Box::addConstraintsToModel() {
  ModelPtr model = m_prob->getModel();
  const double dt = m_prob->m_dt;
  // initial conditions
  for (int i = 0; i < 3; ++i) model->addEqCnt(m_trajvars.x(0,i) - m_init_state.x(i), "");
  for (int i = 0; i < 3; ++i) model->addEqCnt(m_trajvars.v(0,i) - m_init_state.v(i), "");
  for (int i = 0; i < 3; ++i) model->addEqCnt(m_trajvars.q(0,i) - m_init_state.q.vec()(i), "");
  model->addEqCnt(m_trajvars.q(0,3) - m_init_state.q.w(), "");
  for (int i = 0; i < 3; ++i) model->addEqCnt(m_trajvars.w(0,i) - m_init_state.w(i), "");
  // integration (implicit euler)
  for (int t = 1; t < m_prob->m_timesteps; ++t) {
    for (int i = 0; i < 3; ++i) model->addEqCnt(m_trajvars.x(t,i) - m_trajvars.x(t-1,i) - dt*m_trajvars.v(t,i), "");
    for (int i = 0; i < 3; ++i) model->addEqCnt(m_trajvars.v(t,i) - m_trajvars.v(t-1,i) - dt/m_props.mass*m_trajvars.force(t,i), "");

    m_prob->addConstr(ConstraintPtr(
      new QuatIntegrationConstraint(dt, m_trajvars.q.row(t), m_trajvars.q.row(t-1), m_trajvars.w.row(t), (boost::format("qint_%d") % t).str())
    ));

    for (int i = 0; i < 3; ++i) model->addEqCnt(m_trajvars.w(t,i) - m_trajvars.w(t-1,i) - dt*m_trajvars.torque(t,i), ""); //FIXME: inertia
  }
  // forces
  for (int t = 0; t < m_prob->m_timesteps; ++t) {
    for (int i = 0; i < 3; ++i) {
      AffExpr force_i;
      force_i.constant += m_prob->m_gravity(i);
      // force from contacts
      for (ContactPtr &contact : m_contacts) {
        exprInc(force_i, contact->getForceExpr(t,i));
      }
//      for (BoxGroundContactPtr &ground_cont : m_ground_conts) {
//        exprInc(force_i, ground_cont->m_trajvars.f(t,i));
//      }
      // TODO: add other forces here
      model->addEqCnt(AffExpr(m_trajvars.force(t,i)) - force_i, "");
    }
  }
  // torques
  /*
  for (int t = 0; t < m_prob->m_timesteps; ++t) {
    for (int i = 0; i < 3; ++i) {
    }
  }
  */

//  for (BoxGroundContactPtr &ground_cont : m_ground_conts) {
//    ground_cont->addConstraintsToModel();
//  }

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


VectorXd BoxGroundConstraintErrCalc::operator()(const VectorXd &vals) const {
  assert(vals.size() == 7);
  Vector3d box_x = vals.block<3,1>(0,0);
  Quaterniond box_r = toQuat(vals.block<4,1>(3,0));
  cout << "vals " << vals.transpose() << endl;
  m_cnt->m_box->m_kinbody->SetTransform(toOR(box_x, box_r));

  m_cnt->m_prob->m_cc->SetContactDistance(0.);
  vector<Collision> collisions;
  m_cnt->m_prob->m_cc->BodyVsAll(*m_cnt->m_box->m_kinbody, collisions);
  for (Collision &c : collisions) {
    if (c.linkB == m_cnt->m_ground->m_kinbody->GetLinks()[0].get()) {
      return makeVector1d(abs(c.distance));
    }
  }
  return makeVector1d(0);
}

VarVector BoxGroundConstraintErrCalc::buildVarVector(Box *box, int t) {
  VarVector v;
  for (int i = 0; i < 3; ++i) {
    v.push_back(box->m_trajvars.x(t,i));
  }
  for (int i = 0; i < 4; ++i) {
    v.push_back(box->m_trajvars.q(t,i));
  }
  return v;
}

void Ground::addToRave() {
  m_kinbody = OR::RaveCreateKinBody(m_prob->m_env, "");
  m_kinbody->SetName(getName());
  vector<OR::AABB> aabb;
  aabb.push_back(OR::AABB(OR::Vector(0, 0, m_z - 1), OR::Vector(100, 100, 1)));
  m_kinbody->InitFromBoxes(aabb, true);
  m_prob->m_env->Add(m_kinbody);
}


} // namespace dynamics
} // namespace trajopt
