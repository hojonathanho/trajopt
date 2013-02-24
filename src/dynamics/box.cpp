#include "box.h"

#include "trajopt/utils.hpp"
#include "ipi/sco/expr_vec_ops.hpp"

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

struct BoxGroundContactComplCntND : public ConstraintFromNumDiff {

  struct ErrCalc : public VectorOfVector {
    BoxGroundContactComplCntND *m_cnt;
    ErrCalc(BoxGroundContactComplCntND *cnt) : m_cnt(cnt) { }
    VectorXd operator()(const VectorXd &vals) const {
      assert(vals.size() == 6);
      Vector3d box_x_val(vals.segment<3>(0));
      Vector3d cont_f_val(vals.segment<3>(3));
      double zdiff = (box_x_val(2) - m_cnt->m_contact->m_box->m_props.half_extents(2)) - m_cnt->m_contact->m_ground->m_z;
      double fnormal = cont_f_val(2);
      double viol = zdiff * fnormal;
      return makeVector1d(viol);
    }
    static VarVector buildVarVector(BoxGroundContact *c, int t) {
      VarVector v;
      for (int i = 0; i < 3; ++i) v.push_back(c->m_box->m_trajvars.x(t,i));
      for (int i = 0; i < 3; ++i) v.push_back(c->m_trajvars.f(t,i));
      return v;
    }
  };

  BoxGroundContact *m_contact;
  int m_t;

  BoxGroundContactComplCntND(BoxGroundContact *contact, int t, const string &name_prefix)
    : m_contact(contact), m_t(t),
      ConstraintFromNumDiff(
        VectorOfVectorPtr(new ErrCalc(this)),
        ErrCalc::buildVarVector(contact, t),
        EQ,
        (boost::format("%s_%d") % name_prefix % t).str())
  { }
};


//struct BoxGroundContactComplCntND : public ConstraintFromNumDiff {
//
//  struct ErrCalc : public VectorOfVector {
//    BoxGroundContactComplCntND *m_cnt;
//    ErrCalc(BoxGroundContactComplCntND *cnt) : m_cnt(cnt) { }
//    VectorXd operator()(const VectorXd &vals) const {
//      assert(vals.size() == 13);
//      Vector3d box_x_val(vals.segment<3>(0));
//      Quaterniond box_q_val(vals.segment<4>(3));
//      Vector3d cont_p_val(vals.segment<3>(7));
//      Vector3d cont_f_val(vals.segment<3>(10));
//
//      Vector3d cont_p_global = box_x_val + cont_p_val; // FIXME: rotations
//      double zdiff = cont_p_global(2) - m_cnt->m_contact->m_ground->m_z;
//      //double zdiff = (box_x_val(2) - m_cnt->m_contact->m_box->m_props.half_extents(2)) - m_cnt->m_contact->m_ground->m_z;
//      double fnormal = cont_f_val(2);
//      double viol = zdiff * fnormal;
//      return makeVector1d(abs(viol));
//    }
//    static VarVector buildVarVector(BoxGroundContact *c, int t) {
//      VarVector v;
//      for (int i = 0; i < 3; ++i) v.push_back(c->m_box->m_trajvars.x(t,i));
//      for (int i = 0; i < 4; ++i) v.push_back(c->m_box->m_trajvars.q(t,i));
//      for (int i = 0; i < 3; ++i) v.push_back(c->m_trajvars.p(t,i));
//      for (int i = 0; i < 3; ++i) v.push_back(c->m_trajvars.f(t,i));
//      return v;
//    }
//  };
//
//  BoxGroundContact *m_contact;
//  int m_t;
//
//  BoxGroundContactComplCntND(BoxGroundContact *contact, int t, const string &name_prefix)
//    : m_contact(contact), m_t(t),
//      ConstraintFromNumDiff(
//        VectorOfVectorPtr(new ErrCalc(this)),
//        ErrCalc::buildVarVector(contact, t),
//        EQ,
//        (boost::format("%s_%d") % name_prefix % t).str())
//  { }
//};

void BoxGroundContact::addConstraintsToModel() {
  DynamicsProblem *prob = m_box->m_prob;
  ModelPtr model = prob->getModel();

  // box cannot penetrate ground
  for (int t = 0; t < prob->m_timesteps; ++t) {
    //prob->addConstr(ConstraintPtr(new BoxGroundConstraintND(prob, m_box, m_ground, t)));
    //prob->addConstr(ConstraintPtr(new BoxGroundConstraint(prob, m_box, m_ground, t)));
    AffExpr exp(-m_box->m_trajvars.x(t,2));
    exp.constant = m_ground->m_z + m_box->m_props.half_extents(2);
    model->addIneqCnt(exp, "");
  }

  // contact origin point must stay inside box (in local coords)
  for (int t = 0; t < prob->m_timesteps; ++t) {
    for (int i = 0; i < 3; ++i) {
      model->addIneqCnt(m_trajvars.p(t,i) - m_box->m_props.half_extents(i), "");
      model->addIneqCnt(-m_trajvars.p(t,i) - m_box->m_props.half_extents(i), "");
    }
  }

  // contact force must have z-component >= 0
  for (int t = 0; t < prob->m_timesteps; ++t) {
    model->addIneqCnt(-m_trajvars.f(t,2), "");
  }

  // FIXME: use actual friction cone instead
  for (int t = 0; t < prob->m_timesteps; ++t) {
    model->addEqCnt(AffExpr(m_trajvars.f(t,0)), "");
    model->addEqCnt(AffExpr(m_trajvars.f(t,1)), "");
  }

  // complementarity constraints
  for (int t = 0; t < prob->m_timesteps; ++t) {
    prob->addConstr(ConstraintPtr(new BoxGroundContactComplCntND(this, t, "box_ground_compl")));
  }

  model->update();
}

AffExpr BoxGroundContact::getForceExpr(int t, int i) {
  return AffExpr(m_trajvars.f(t,i));
}

vector<DynamicsObject*> BoxGroundContact::getAffectedObjects() {
  return vector<DynamicsObject*>{m_box, m_ground};
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

//    m_prob->addConstr(ConstraintPtr(
//      new QuatIntegrationConstraint(dt, m_trajvars.q.row(t), m_trajvars.q.row(t-1), m_trajvars.w.row(t), (boost::format("qint_%d") % t).str())
//    ));
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
        exprInc(force_t_i, contact->getForceExpr(t,i));
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

//template<class T>
//bool in(const T *a, const vector<boost::shared_ptr<T> > &b) {
//  for (int i = 0; i < b.size(); ++i) {
//    if (b[i].get() == a) {
//      return true;
//    }
//  }
//  return false;
//}

Collision flipCollision(const Collision &c) {
  return Collision(c.linkB, c.linkA, c.ptB, c.ptA, -c.normalB2A, c.distance, c.weight, c.time);
}

static const double CONTACT_DIST = 10.;

// output convention: A is the box, B is the ground
static int calls = 0;
static Collision checkBoxGroundCollision(OR::KinBodyPtr box, OR::KinBodyPtr ground, CollisionCheckerPtr cc) {
  ++calls;
  cout << "COLLISION CHECKS: " << calls << endl;
  vector<Collision> collisions;
  cc->BodyVsAll(*box, collisions);
  cout << "NUM COLLISIONS: " << collisions.size() << endl;
  const OR::KinBody::Link* ground_link = ground->GetLinks()[0].get();
  for (Collision &c : collisions) {
    cout << "collision?: " << c.linkA->GetParent()->GetName() << ' ' << c.linkB->GetParent()->GetName() << ' ' << c.distance << endl;
    if (c.linkB == ground_link) {
      return c;
    } else if (c.linkA == ground_link) {
      return flipCollision(c);
    }
  }
  return Collision(NULL, NULL, OR::Vector(), OR::Vector(), OR::Vector(), 0.);
}
static Collision checkBoxGroundCollision(const OR::Transform &trans, OR::KinBodyPtr box, OR::KinBodyPtr ground, CollisionCheckerPtr cc) {
  box->SetTransform(trans);
  return checkBoxGroundCollision(box, ground, cc);
}

VectorXd BoxGroundConstraintNDErrCalc::operator()(const VectorXd &vals) const {
  assert(vals.size() == 7);
  Vector3d box_x = vals.block<3,1>(0,0);
  Quaterniond box_r(vals.block<4,1>(3,0));
  cout << "vals " << box_x.transpose() << " | " << box_r.coeffs().transpose() << endl;

  m_cnt->m_prob->m_cc->SetContactDistance(CONTACT_DIST);
  Collision col(checkBoxGroundCollision(toOR(box_x, box_r), m_cnt->m_box->m_kinbody, m_cnt->m_ground->m_kinbody, m_cnt->m_prob->m_cc));
  if (col.linkA != NULL) {
    return makeVector1d(-col.distance);
  }
  return makeVector1d(0);
}

VarVector BoxGroundConstraintNDErrCalc::buildVarVector(Box *box, int t) {
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


BoxGroundConstraint::BoxGroundConstraint(DynamicsProblem *prob, Box *box, Ground *ground, int t, const string &name_prefix)
  : m_prob(prob), m_box(box), m_ground(ground), m_t(t),
    Constraint((boost::format("%s_%d") % name_prefix % t).str())
{ }

vector<double> BoxGroundConstraint::value(const vector<double>& x) {
//  Vector3d box_x = getVec(x, m_box->m_trajvars.x.row(m_t));
//  Quaterniond box_q(getQuat(x, m_box->m_trajvars.q.row(m_t)));
//  cout << "vals " << box_x.transpose() << " | " << box_q.coeffs().transpose() << endl;
  m_box->setRaveState(x, m_t);

  m_prob->m_cc->SetContactDistance(CONTACT_DIST);
  Collision col(checkBoxGroundCollision(m_box->m_kinbody, m_ground->m_kinbody, m_prob->m_cc));
  if (col.linkA != NULL) {
    return vector<double>(1, -col.distance);
  }
  return vector<double>(1, 0);
}

ConvexConstraintsPtr BoxGroundConstraint::convex(const vector<double>& x, Model* model) {
  ConvexConstraintsPtr out(new ConvexConstraints(model));
  Vector3d box_x = getVec(x, m_box->m_trajvars.x.row(m_t));
//  Quaterniond box_q(getQuat(x, m_box->m_trajvars.q.row(m_t)));
//  cout << "vals " << box_x.transpose() << " | " << box_q.coeffs().transpose() << endl;
//  OR::Transform box_trans = toOR(box_x, box_q);
  m_box->setRaveState(x, m_t);
  m_prob->m_cc->SetContactDistance(CONTACT_DIST);
  Collision col(checkBoxGroundCollision(m_box->m_kinbody, m_ground->m_kinbody, m_prob->m_cc));
  if (col.linkA != NULL) {
    // TODO: rotations
    Vector3d dist_grad = toVector3d(col.normalB2A); // ... .transpose() * Identity (ignoring rotation)
    AffExpr sd(col.distance);
    exprInc(sd, varDot(dist_grad, m_box->m_trajvars.x.row(m_t)));
    exprInc(sd, -dist_grad.dot(box_x));
    out->addIneqCnt(-sd);
  }
  return out;
}

} // namespace dynamics
} // namespace trajopt
