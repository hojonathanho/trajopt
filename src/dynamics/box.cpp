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

  // box cannot penetrate ground
  for (int t = 0; t < prob->m_timesteps; ++t) {
    prob->addConstr(ConstraintPtr(new BoxGroundConstraint(prob, m_ground->m_z, m_box, t)));
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

//  for (int z = 0; z < m_ground_conts.size(); ++z) {
//    m_ground_conts[z]->fillVarNamesAndBounds(out_names, out_vlower, out_vupper, (boost::format("%s_contact_%d") % name_prefix % z).str());
//  }

//  assert(out_names.size() - init_size == m_prob->m_timesteps*(BoxState::Dim() + m_ground_conts.size()*ContactState::Dim()));
    assert(out_names.size() - init_size == m_prob->m_timesteps*BoxState::Dim());
}


int Box::setVariables(const vector<Var> &vars, int start_pos) {
  int k = start_pos;
  for (int t = 0; t < m_prob->m_timesteps; ++t) {
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
  for (int i = 0; i < 3; ++i) model->addEqCnt(m_trajvars.p(0,i) - m_init_state.p(i), "");
  for (int i = 0; i < 3; ++i) model->addEqCnt(m_trajvars.r(0,i) - m_init_state.r.vec()(i), "");
  model->addEqCnt(m_trajvars.r(0,3) - m_init_state.r.w(), "");
  for (int i = 0; i < 3; ++i) model->addEqCnt(m_trajvars.L(0,i) - m_init_state.L(i), "");
  // integration (implicit euler)
  for (int t = 1; t < m_prob->m_timesteps; ++t) {
    for (int i = 0; i < 3; ++i) model->addEqCnt(m_trajvars.x(t,i) - m_trajvars.x(t-1,i) - dt/m_props.mass*m_trajvars.p(t,i), "");
    for (int i = 0; i < 3; ++i) model->addEqCnt(m_trajvars.p(t,i) - m_trajvars.p(t-1,i) - dt*m_trajvars.force(t,i), "");
    // TODO: torque
  }
  // forces
  for (int t = 0; t < m_prob->m_timesteps; ++t) {
    for (int i = 0; i < 3; ++i) {
      AffExpr force_i;
      force_i.constant += m_prob->m_gravity(i);
      // force from ground contacts
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
  OR::KinBodyPtr kinbody = OR::RaveCreateKinBody(m_prob->m_env, "");
  kinbody->SetName(getName());
  vector<OR::AABB> aabb;
  aabb.push_back(OR::AABB(OR::Vector(0, 0, 0), toOR(m_props.half_extents)));
  kinbody->InitFromBoxes(aabb, true);
  m_prob->m_env->Add(kinbody);
  kinbody->SetTransform(OR::Transform(toOR(m_init_state.r), toOR(m_init_state.x)));
}


VectorXd BoxGroundConstraintErrCalc::operator()(const VectorXd &vals) const {
  assert(vals.size() == 7);
  Vector3d box_x = vals.block<3,1>(0,0);
  Quaterniond box_r = toQuat(vals.block<4,1>(3,0));

  // TODO: take rotation into account
  return makeVector1d(positivePart(m_cnt->m_ground_z - (box_x(2) - m_cnt->m_box->m_props.half_extents(2))));

//  vector<Collision> collisions;
//  m_cnt->m_prob->m_cc->BodyVsAll(*m_cnt->m_box->m_kinbody, collisions);
//  for (Collision &c : collisions) {
//    // FIXME: check what we're colliding against
//    c.
//  }
}

VarVector BoxGroundConstraintErrCalc::buildVarVector(Box *box, int t) {
  VarVector v;
  for (int i = 0; i < 3; ++i) {
    v.push_back(box->m_trajvars.x(t,i));
  }
  for (int i = 0; i < 4; ++i) {
    v.push_back(box->m_trajvars.r(t,i));
  }
  return v;
}

//
//void Box::addGroundNonpenetrationCnts(double ground_z) {
//  for (int t = 0; t < m_prob->m_timesteps; ++t) {
//    m_prob->addConstr(ConstraintPtr(new BoxGroundConstraint(m_prob, ground_z, this, t)));
//  }
//}

void Ground::addToRave() {
  OR::KinBodyPtr kinbody = OR::RaveCreateKinBody(m_prob->m_env, "");
  kinbody->SetName(getName());
  vector<OR::AABB> aabb;
  aabb.push_back(OR::AABB(OR::Vector(0, 0, m_z - 1), OR::Vector(100, 100, 1)));
  kinbody->InitFromBoxes(aabb, true);
  m_prob->m_env->Add(kinbody);
}


} // namespace dynamics
} // namespace trajopt
