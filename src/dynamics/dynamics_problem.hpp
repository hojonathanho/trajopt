#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <vector>
#include "trajopt/typedefs.hpp"
#include "trajopt/rave_utils.hpp"
#include "trajopt/collision_checker.hpp"
#include "util.h"

namespace trajopt {
namespace dynamics {

using namespace std;
using namespace Eigen;

class DynamicsObject {
public:
  virtual ~DynamicsObject() { }
  virtual void fillVarNamesAndBounds(vector<string> &out_names, vector<double> &out_vlower, vector<double> &out_vupper, const string &name_prefix="obj") = 0;
  virtual int setVariables(const vector<Var> &vars, int start_pos) = 0;
  virtual void addConstraintsToModel() = 0;

  // ground hacks
  virtual void registerGroundContact() { }
  virtual void addGroundNonpenetrationCnts(double ground_z) { }
};
typedef boost::shared_ptr<DynamicsObject> DynamicsObjectPtr;

class DynamicsProblem : public OptProb {
public:
  OR::EnvironmentBasePtr m_env;
  CollisionCheckerPtr m_cc;
  vector<DynamicsObjectPtr> m_objects;

  DynamicsProblem(OR::EnvironmentBasePtr env) : m_env(env) { }

  int m_timesteps;
  double m_dt;
  Vector3d m_gravity;
  void setNumTimesteps(int n) { m_timesteps = n; }
  void setDt(double dt) { m_dt = dt; }
  void setGravity(const Vector3d &g) { m_gravity = g; }

  void addObject(DynamicsObjectPtr obj) { m_objects.push_back(obj); }

  void setUpProblem();
};
typedef boost::shared_ptr<DynamicsProblem> DynamicsProblemPtr;


} // namespace dynamics
} // namespace trajopt
