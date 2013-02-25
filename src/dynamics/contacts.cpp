#include "contacts.hpp"

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



AffExpr makeDerivExpr(const VectorXd& grad, const VarVector& vars, const VectorXd& curvals) {
  assert (grad.size() == vars.size());
  return varDot(grad, vars) - grad.dot(curvals);
}

struct BoxGroundComplementarityCost : public Cost { // hinge cost on complementarity violation (no box rotation)
  BoxGroundContact *m_contact;
  int m_t;
  BoxGroundComplementarityCost(BoxGroundContact *contact, int t) : m_contact(contact), m_t(t) { }

  virtual double value(const vector<double>& x) {
    Vector3d box_x_val(getVec(x, m_contact->m_box->m_trajvars.x.row(m_t)));
    Vector3d cont_f_val(getVec(x, m_contact->m_trajvars.f.row(m_t)));
    double zdiff = (box_x_val(2) - m_contact->m_box->m_props.half_extents(2)) - m_contact->m_ground->m_z;
    return positivePart(zdiff * cont_f_val(2));
  }
  virtual ConvexObjectivePtr convex(const vector<double>& x, Model* model) {
    VarVector box_x_vars = m_contact->m_box->m_trajvars.x.row(m_t);
    VarVector cont_f_vars = m_contact->m_trajvars.f.row(m_t);
    Vector3d box_x_val(getVec(x, box_x_vars));
    Vector3d cont_f_val(getVec(x, cont_f_vars));
    double zdiff = (box_x_val(2) - m_contact->m_box->m_props.half_extents(2)) - m_contact->m_ground->m_z;

    Vector3d normalB2A(0, 0, 1);
    AffExpr dDistOfA = makeDerivExpr(normalB2A, box_x_vars, box_x_val);
                     //    + exprDot(m_normalB2A, exprCross(m_bodyA->m_r1_var, m_worldA - m_bodyA->m_x1_val));
    //AffExpr dDistOfB = makeDerivExpr(-normalB2A, m_bodyB->m_x1_var, m_bodyB->m_x1_val);
                      //   + exprDot(-m_normalB2A, exprCross(m_bodyB->m_r1_var, m_worldB - m_bodyB->m_x1_val));
    AffExpr dist_expr = (dDistOfA + /*dDistOfB +*/ zdiff);

    AffExpr final_expr = dist_expr*cont_f_val(2) + zdiff*(cont_f_vars[2] - cont_f_val(2));

    cout << "analytical: " << final_expr << endl;

    // test linearization numerically
#if 1
    VarVector all_vars = concat(box_x_vars, cont_f_vars);
    vector<double> xtmp = x; const double eps = 1e-4;
    AffExpr nd_final_expr(value(x));
    for (int z = 0; z < all_vars.size(); ++z) {
      Var v = all_vars[z];
      int idx = v.var_rep->index;
      xtmp[idx] += eps;
      double v2 = value(xtmp);
      xtmp[idx] -= 2.*eps;
      double v1 = value(xtmp);
      xtmp[idx] = x[idx];
      double deriv = ((v2 - v1) / (2.*eps));
      if (abs(deriv) < 1e-5) continue;
      exprInc(nd_final_expr, deriv*(v - x[idx]));
    }
    cout << "nd: " << nd_final_expr << endl;
    cout << "====================" << endl;
#endif

    ConvexObjectivePtr out(new ConvexObjective(model));
    out->addHinge(final_expr, 1);
    return out;

    //GRBLinExpr compExpr = contact->m_distExpr * contact->m_fn_val + contact->m_dist * (contact->m_fn - contact->m_fn_val);
    //addHingeCost(out, m_coeff, compExpr, model, "compl");

  }
};

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

class BoxGroundConstraintSimple : public Constraint { // no rotation -- just constrains z-value to be above ground+height/2
public:
  virtual ConstraintType type() { return INEQ; }
  virtual vector<double> value(const vector<double>& x) {
    double val = -m_box->m_trajvars.x(m_t,2).value(x) + m_ground->m_z + m_box->m_props.half_extents(2);
    return vector<double>{val};
  }
  virtual ConvexConstraintsPtr convex(const vector<double>& x, Model* model) {
    ConvexConstraintsPtr out(new ConvexConstraints(model));
    AffExpr exp(-m_box->m_trajvars.x(m_t,2));
    exp.constant = m_ground->m_z + m_box->m_props.half_extents(2);
    out->addIneqCnt(exp);
    return out;
  }

  Box *m_box; Ground *m_ground; int m_t;
  BoxGroundConstraintSimple(Box *box, Ground *ground, int t) : m_box(box), m_ground(ground), m_t(t) { }
};

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

  // contact force must have z-component >= 0
  for (int t = 0; t < prob->m_timesteps; ++t) {
    model->addIneqCnt(-m_trajvars.f(t,2), "");
  }

  // FIXME: use actual friction cone instead
  for (int t = 0; t < prob->m_timesteps; ++t) {
    model->addEqCnt(AffExpr(m_trajvars.f(t,0)), "");
    model->addEqCnt(AffExpr(m_trajvars.f(t,1)), "");
  }

  // box cannot penetrate ground
  for (int t = 0; t < prob->m_timesteps; ++t) {
    //prob->addConstr(ConstraintPtr(new BoxGroundConstraintND(prob, m_box, m_ground, t)));
   // prob->addConstr(ConstraintPtr(new BoxGroundConstraint(prob, m_box, m_ground, t)));
//    AffExpr exp(-m_box->m_trajvars.x(t,2));
//    exp.constant = m_ground->m_z + m_box->m_props.half_extents(2);
//    model->addIneqCnt(exp, "");
    prob->addConstr(ConstraintPtr(new BoxGroundConstraintSimple(m_box, m_ground, t)));
  }

  // complementarity
  for (int t = 0; t < prob->m_timesteps; ++t) {
 //   prob->addConstr(ConstraintPtr(new BoxGroundContactComplCntND(this, t, "box_ground_compl")));
    prob->addCost(CostPtr(new BoxGroundComplementarityCost(this, t)));
  }

  model->update();
}

AffExpr BoxGroundContact::getForceExpr(DynamicsObject *o, int t, int i) {
  assert(o == m_box);
  return AffExpr(m_trajvars.f(t,i));
}

vector<DynamicsObject*> BoxGroundContact::getAffectedObjects() {
  return vector<DynamicsObject*>{m_box, m_ground};
}


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
      cout << "NO FLIP" << endl;
      return c;
    } else if (c.linkA == ground_link) {
      cout << "FLIP" << endl;
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
    cout << col << endl;
    Vector3d dist_grad = toVector3d(col.normalB2A).normalized(); // ... .transpose() * Identity (ignoring rotation)
    AffExpr sd(col.distance);
    exprInc(sd, varDot(dist_grad, m_box->m_trajvars.x.row(m_t)));
    exprInc(sd, -dist_grad.dot(box_x));
    cout << "signed dist expr: " << sd << endl;
    out->addIneqCnt(-sd);
  }
  return out;
}

BoxBoxContact::BoxBoxContact(const string &name, Box *box1, Box *box2)
    : m_box1(box1), m_box2(box2), m_trajvars(box1->m_prob->m_timesteps), m_prob(box1->m_prob), Contact(name) {
  assert(m_box1->m_prob == m_box2->m_prob);
}

void BoxBoxContact::fillVarNamesAndBounds(vector<string> &out_names, vector<double> &out_vlower, vector<double> &out_vupper, const string &name_prefix) {
  for (int t = 0; t < m_prob->m_timesteps; ++t) {
    for (int i = 0; i < 3; ++i) {
      out_names.push_back((boost::format("%s_p1_%i_%i") % name_prefix % t % i).str());
      out_vlower.push_back(-INFINITY);
      out_vupper.push_back(INFINITY);
    }
    for (int i = 0; i < 3; ++i) {
      out_names.push_back((boost::format("%s_p2_%i_%i") % name_prefix % t % i).str());
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

void BoxBoxContact::fillInitialSolution(vector<double> &out) {
  for (int t = 0; t < m_prob->m_timesteps; ++t) {
    for (int i = 0; i < 3; ++i) { // p1
      out.push_back(0);
    }
    for (int i = 0; i < 3; ++i) { // p2
      out.push_back(0);
    }
    for (int i = 0; i < 3; ++i) { // f
      out.push_back(0);
    }
  }
}

int BoxBoxContact::setVariables(const vector<Var> &vars, int start_pos) {
  int k = start_pos;
  for (int t = 0; t < m_prob->m_timesteps; ++t) {
    for (int i = 0; i < 3; ++i) {
      m_trajvars.p1(t,i) = vars[k++];
    }
    for (int i = 0; i < 3; ++i) {
      m_trajvars.p2(t,i) = vars[k++];
    }
    for (int i = 0; i < 3; ++i) {
      m_trajvars.f(t,i) = vars[k++];
    }
  }
  assert(k - start_pos == m_prob->m_timesteps*9);
  return k;
}

//class BoxBoxPenetrationConstraintSimple : public IneqConstraint {
//public:
//  Box *m_box1, *m_box2; int m_t;
//  BoxBoxPenetrationConstraintSimple(Box *box1, Box *box2, int t) : m_box1(box1), m_box2(box2), m_t(t) { }
//
//  virtual vector<double> value(const vector<double>& x) {
//    VarArray box1_x_vars = m_box1->m_trajvars.x.row(m_t);
//    VarArray box2_x_vars = m_box2->m_trajvars.x.row(m_t);
//
//    vector<double> v(4);
//    v[0] = (box1_x_vars[0] + m_box1->m_props.half_extents(0)) - (box2_x_vars[0] - m_box2->m_props.half_extents(0));
//  }
//  virtual ConvexConstraintsPtr convex(const vector<double>& x, Model* model) {
//
//  }
//};

static Collision checkBoxBoxCollision(OR::KinBodyPtr box1, OR::KinBodyPtr box2, CollisionCheckerPtr cc) {
  ++calls;
  cout << "COLLISION CHECKS: " << calls << endl;
  vector<Collision> collisions;
  cc->BodyVsAll(*box1, collisions);
  cout << "NUM COLLISIONS: " << collisions.size() << endl;
  const OR::KinBody::Link* box2_link = box2->GetLinks()[0].get();
  for (Collision &c : collisions) {
    cout << "collision?: " << c.linkA->GetParent()->GetName() << ' ' << c.linkB->GetParent()->GetName() << ' ' << c.distance << endl;
    if (c.linkB == box2_link) {
      return c;
    } else if (c.linkA == box2_link) {
      return flipCollision(c);
    }
  }
  return Collision(NULL, NULL, OR::Vector(), OR::Vector(), OR::Vector(), 0.);
}

class BoxBoxPenetrationConstraint : public IneqConstraint {
public:
  Box *m_box1, *m_box2; int m_t; DynamicsProblem *m_prob;
  BoxBoxPenetrationConstraint(Box *box1, Box *box2, int t) : m_box1(box1), m_box2(box2), m_t(t), m_prob(box1->m_prob) { }

  virtual vector<double> value(const vector<double>& x) {
    VarVector box1_x_vars = m_box1->m_trajvars.x.row(m_t);
    VarVector box2_x_vars = m_box2->m_trajvars.x.row(m_t);

    m_box1->setRaveState(x, m_t);
    m_box2->setRaveState(x, m_t);
    m_prob->m_cc->SetContactDistance(CONTACT_DIST);
    Collision col(checkBoxBoxCollision(m_box1->m_kinbody, m_box2->m_kinbody, m_prob->m_cc));
    if (col.linkA != NULL) {
      return vector<double>(1, -col.distance);
    }
    return vector<double>(1, 0);
  }

  virtual ConvexConstraintsPtr convex(const vector<double>& x, Model* model) {
    ConvexConstraintsPtr out(new ConvexConstraints(model));
    VarVector box1_x_vars = m_box1->m_trajvars.x.row(m_t);
    VarVector box2_x_vars = m_box2->m_trajvars.x.row(m_t);
    Vector3d box1_x = getVec(x, m_box1->m_trajvars.x.row(m_t));
    Vector3d box2_x = getVec(x, m_box2->m_trajvars.x.row(m_t));

    m_box1->setRaveState(x, m_t);
    m_box2->setRaveState(x, m_t);
    m_prob->m_cc->SetContactDistance(CONTACT_DIST);
    Collision col(checkBoxBoxCollision(m_box1->m_kinbody, m_box2->m_kinbody, m_prob->m_cc));
    if (col.linkA != NULL) {
      // TODO: rotations
      cout << col << endl;
      Vector3d dist_grad = toVector3d(col.normalB2A).normalized(); // ... .transpose() * Identity (ignoring rotation)
      AffExpr sd(col.distance);

      exprInc(sd, varDot(dist_grad, box1_x_vars));
      exprInc(sd, -dist_grad.dot(box1_x));

      exprInc(sd, varDot(-dist_grad, box2_x_vars));
      exprInc(sd, dist_grad.dot(box2_x));

      cout << "signed dist expr: " << sd << endl;
      out->addIneqCnt(-sd);
    }
    return out;
  }
};


class BoxBoxComplementarityCost : public Cost { // hinge cost on complementarity violation (no box rotation)
public:
  BoxBoxContact *m_contact;
  int m_t;
  BoxBoxComplementarityCost(BoxBoxContact *contact, int t) : m_contact(contact), m_t(t) { }

  virtual double value(const vector<double>& x) {
    double dist;
    Vector3d normal;
    getContactDistAndNormal(x, dist, normal);
    Vector3d cont_f_val(getVec(x, m_contact->m_trajvars.f.row(m_t)));
    return positivePart(dist * normal.dot(cont_f_val));
  }

  virtual ConvexObjectivePtr convex(const vector<double>& x, Model* model) {
    VarVector box1_x_vars = m_contact->m_box1->m_trajvars.x.row(m_t);
    VarVector box2_x_vars = m_contact->m_box2->m_trajvars.x.row(m_t);
    VarVector cont_f_vars = m_contact->m_trajvars.f.row(m_t);
    Vector3d box1_x_val(getVec(x, box1_x_vars));
    Vector3d box2_x_val(getVec(x, box2_x_vars));
    Vector3d cont_f_val(getVec(x, cont_f_vars));

    double dist;
    Vector3d normal;
    getContactDistAndNormal(x, dist, normal);
    double fnormal = normal.dot(cont_f_val);

    AffExpr dDistOfA = makeDerivExpr(normal, box1_x_vars, box1_x_val);
                     //    + exprDot(m_normalB2A, exprCross(m_bodyA->m_r1_var, m_worldA - m_bodyA->m_x1_val));
    AffExpr dDistOfB = makeDerivExpr(-normal, box2_x_vars, box2_x_val);
                      //   + exprDot(-m_normalB2A, exprCross(m_bodyB->m_r1_var, m_worldB - m_bodyB->m_x1_val));
    AffExpr dist_expr = (dDistOfA + dDistOfB + dist);

    AffExpr fnormal_var_expr = varDot(normal, cont_f_vars); // FIXME: is this even right???????
    AffExpr final_expr = dist_expr*fnormal + dist*(fnormal_var_expr - fnormal);

    cout << "analytical: " << final_expr << endl;

    // test linearization numerically
#if 1
    VarVector all_vars = concat(concat(box1_x_vars, box2_x_vars), cont_f_vars);
    vector<double> xtmp = x; const double eps = 1e-4;
    AffExpr nd_final_expr(value(x));
    for (int z = 0; z < all_vars.size(); ++z) {
      Var v = all_vars[z];
      int idx = v.var_rep->index;
      xtmp[idx] += eps;
      double v2 = value(xtmp);
      xtmp[idx] -= 2.*eps;
      double v1 = value(xtmp);
      xtmp[idx] = x[idx];
      double deriv = ((v2 - v1) / (2.*eps));
      if (abs(deriv) < 1e-5) continue;
      exprInc(nd_final_expr, deriv*(v - x[idx]));
    }
    cout << "nd: " << nd_final_expr << endl;
    cout << "====================" << endl;
#endif

    ConvexObjectivePtr out(new ConvexObjective(model));
    //out->addHinge(final_expr, 1);
    out->addHinge(nd_final_expr, 1);
    return out;

    //GRBLinExpr compExpr = contact->m_distExpr * contact->m_fn_val + contact->m_dist * (contact->m_fn - contact->m_fn_val);
    //addHingeCost(out, m_coeff, compExpr, model, "compl");

  }

protected:
  void getContactDistAndNormal(const vector<double> &x, double &out_dist, Vector3d &out_normal) {
    m_contact->m_box1->setRaveState(x, m_t);
    m_contact->m_box2->setRaveState(x, m_t);
    m_contact->m_prob->m_cc->SetContactDistance(CONTACT_DIST);
    Collision col(checkBoxBoxCollision(m_contact->m_box1->m_kinbody, m_contact->m_box2->m_kinbody, m_contact->m_prob->m_cc));
    out_dist = col.linkA != NULL ? col.distance : CONTACT_DIST;

    Vector3d cont_f_val(getVec(x, m_contact->m_trajvars.f.row(m_t)));
    if (col.linkA != NULL) {
      out_normal = toVector3d(col.normalB2A).normalized();
    } else {
      Vector3d x1(getVec(x, m_contact->m_box1->m_trajvars.x.row(m_t)));
      Vector3d x2(getVec(x, m_contact->m_box2->m_trajvars.x.row(m_t)));
      out_normal = (x1 - x2).normalized();
    }
  }
};

void BoxBoxContact::addConstraintsToModel() {
  ModelPtr model = m_prob->getModel();

  // contact origin points must stay inside respective boxes (in local coords)
  for (int t = 0; t < m_prob->m_timesteps; ++t) {
    for (int i = 0; i < 3; ++i) {
      model->addIneqCnt(m_trajvars.p1(t,i) - m_box1->m_props.half_extents(i), "");
      model->addIneqCnt(-m_trajvars.p1(t,i) - m_box1->m_props.half_extents(i), "");

      model->addIneqCnt(m_trajvars.p2(t,i) - m_box2->m_props.half_extents(i), "");
      model->addIneqCnt(-m_trajvars.p2(t,i) - m_box2->m_props.half_extents(i), "");
    }
  }

  // contact force must be repulsive
  for (int t = 0; t < m_prob->m_timesteps; ++t) {
    // TODO
  }

  // friction cone
  for (int t = 0; t < m_prob->m_timesteps; ++t) {
    // TODO
  }

  // boxes cannot penetrate
  for (int t = 0; t < m_prob->m_timesteps; ++t) {
    m_prob->addConstr(ConstraintPtr(new BoxBoxPenetrationConstraint(m_box1, m_box2, t)));
  }

  // complementarity
  for (int t = 0; t < m_prob->m_timesteps; ++t) {
    m_prob->addCost(CostPtr(new BoxBoxComplementarityCost(this, t)));
  }

  model->update();
}

AffExpr BoxBoxContact::getForceExpr(DynamicsObject *o, int t, int i) {
  assert(o == m_box1 || o == m_box2);
  return o == m_box1 ? AffExpr(m_trajvars.f(t,i)) : -AffExpr(m_trajvars.f(t,i));
}

vector<DynamicsObject*> BoxBoxContact::getAffectedObjects() {
  return vector<DynamicsObject*>{m_box1, m_box2};
}

} // namespace dynamics
} // namespace trajopt

