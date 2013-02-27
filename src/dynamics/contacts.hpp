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

struct BoxGroundContactSpec : public Spec {
  string name; // name of this contact
  string box_name;
  string ground_name;
  OptimizationBasePtr realize(DynamicsProblem *prob);
};
typedef boost::shared_ptr<BoxGroundContactSpec> BoxGroundContactSpecPtr;

struct BoxGroundContact : public Contact {
  Box *m_box;
  Ground *m_ground;
  BoxGroundContactTrajVars m_trajvars;

  BoxGroundContact(const BoxGroundContactSpec &spec, DynamicsProblem *prob);

  void fillVarNamesAndBounds(vector<string> &out_names, vector<double> &out_vlower, vector<double> &out_vupper, const string &name_prefix);
  void fillInitialSolution(vector<double> &out);
  int setVariables(const vector<Var> &vars, int start_pos);
  void addConstraintsToModel();

  AffExpr getForceExpr(DynamicsObject *o, int t, int i);
  vector<DynamicsObject*> getAffectedObjects();
};
typedef boost::shared_ptr<BoxGroundContact> BoxGroundContactPtr;


struct BoxBoxContactTrajVars {
  VarArray p1, p2; // contact points (local frames)
  VarArray f; // contact force (world frame) (convention: f applied on box1, -f applied on box2)

  BoxBoxContactTrajVars(int timesteps) {
    p1.resize(timesteps, 3);
    p2.resize(timesteps, 3);
    f.resize(timesteps, 3);
  }
};

struct BoxBoxContactSpec : public Spec {
  string name;
  string box1_name, box2_name;
  OptimizationBasePtr realize(DynamicsProblem *prob);
};
typedef boost::shared_ptr<BoxBoxContactSpec> BoxBoxContactSpecPtr;

struct BoxBoxContact : public Contact {
  Box *m_box1, *m_box2;
  BoxBoxContactTrajVars m_trajvars;

  BoxBoxContact(const BoxBoxContactSpec &spec, DynamicsProblem *prob);

  void fillVarNamesAndBounds(vector<string> &out_names, vector<double> &out_vlower, vector<double> &out_vupper, const string &name_prefix);
  void fillInitialSolution(vector<double> &out);
  int setVariables(const vector<Var> &vars, int start_pos);
  void addConstraintsToModel();

  AffExpr getForceExpr(DynamicsObject *o, int t, int i);
  vector<DynamicsObject*> getAffectedObjects();

  void calcContactDistAndNormal(const vector<double> &x, int t, double &out_dist, Vector3d &out_normal);
};
typedef boost::shared_ptr<BoxBoxContact> BoxBoxContactPtr;

} // namespace dynamics
} // namespace trajopt
