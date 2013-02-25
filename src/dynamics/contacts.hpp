#pragma once

#include "box.hpp"

namespace trajopt {
namespace dynamics {

using namespace std;
using namespace Eigen;


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
  void fillInitialSolution(vector<double> &out);
  int setVariables(const vector<Var> &vars, int start_pos);
  void addConstraintsToModel();

  AffExpr getForceExpr(DynamicsObject *o, int t, int i);
  vector<DynamicsObject*> getAffectedObjects();
};
typedef boost::shared_ptr<BoxGroundContact> BoxGroundContactPtr;


// nonpenetration constraint for a single timestep--lowest point on box must have z-value >= ground z-value
class BoxGroundConstraintND;
struct BoxGroundConstraintNDErrCalc : public VectorOfVector {
  BoxGroundConstraintND *m_cnt;
  BoxGroundConstraintNDErrCalc(BoxGroundConstraintND *cnt) : m_cnt(cnt) { }
  VectorXd operator()(const VectorXd &vals) const;
  static VarVector buildVarVector(Box *box, int t);
};
class BoxGroundConstraintND : public ConstraintFromNumDiff {
public:
  DynamicsProblem *m_prob;
  Box *m_box;
  Ground *m_ground;
  int m_t;

  BoxGroundConstraintND(DynamicsProblem *prob, Box *box, Ground *ground, int t, const string &name_prefix="box_ground")
    : m_prob(prob), m_box(box), m_ground(ground), m_t(t),
      ConstraintFromNumDiff(
        VectorOfVectorPtr(new BoxGroundConstraintNDErrCalc(this)),
        BoxGroundConstraintNDErrCalc::buildVarVector(box, t),
        INEQ,
        (boost::format("%s_%d") % name_prefix % t).str())
  { }
};
typedef boost::shared_ptr<BoxGroundConstraintND> BoxGroundConstraintNDPtr;


class BoxGroundConstraint : public Constraint {
public:
  DynamicsProblem *m_prob;
  Box *m_box;
  Ground *m_ground;
  int m_t;

  BoxGroundConstraint(DynamicsProblem *prob, Box *box, Ground *ground, int t, const string &name_prefix="box_ground");
  virtual ~BoxGroundConstraint() {}

  ConstraintType type() {return INEQ;}
  virtual vector<double> value(const vector<double>& x);
  virtual ConvexConstraintsPtr convex(const vector<double>& x, Model* model);

};

struct BoxBoxContactTrajVars {
  VarArray p1, p2; // contact points (local frames)
  VarArray f; // contact force (world frame) (convention: f applied on box1, -f applied on box2)

  BoxBoxContactTrajVars(int timesteps) {
    p1.resize(timesteps, 3);
    p2.resize(timesteps, 3);
    f.resize(timesteps, 3);
  }
};


struct BoxBoxContact : public Contact {
  DynamicsProblem *m_prob;
  Box *m_box1, *m_box2;
  BoxBoxContactTrajVars m_trajvars;

  BoxBoxContact(const string &name, Box *box1, Box *box2);

  void fillVarNamesAndBounds(vector<string> &out_names, vector<double> &out_vlower, vector<double> &out_vupper, const string &name_prefix);
  void fillInitialSolution(vector<double> &out);
  int setVariables(const vector<Var> &vars, int start_pos);
  void addConstraintsToModel();

  AffExpr getForceExpr(DynamicsObject *o, int t, int i);
  vector<DynamicsObject*> getAffectedObjects();
};
typedef boost::shared_ptr<BoxBoxContact> BoxBoxContactPtr;

} // namespace dynamics
} // namespace trajopt
