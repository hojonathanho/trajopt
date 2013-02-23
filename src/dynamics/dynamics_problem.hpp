#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <vector>
#include "trajopt/typedefs.hpp"
#include "trajopt/rave_utils.hpp"
#include "trajopt/collision_checker.hpp"
#include "ipi/sco/expr_op_overloads.hpp"
#include "ipi/sco/optimizers.hpp"
#include "util.h"

namespace trajopt {
namespace dynamics {

using namespace std;
using namespace Eigen;

class Contact;
class DynamicsObject {
public:
  string m_name;
  const string &getName() const { return m_name; }
  DynamicsObject(const string &name) : m_name(name) { }
  virtual ~DynamicsObject() { }

  virtual void fillVarNamesAndBounds(vector<string> &out_names, vector<double> &out_vlower, vector<double> &out_vupper, const string &name_prefix) = 0;
  virtual void fillInitialSolution(vector<double> &out) = 0;
  virtual int setVariables(const vector<Var> &vars, int start_pos) = 0;
  virtual void addConstraintsToModel() = 0;

  virtual void registerContact(Contact *) = 0;
  virtual void addToRave() = 0;
  virtual void setRaveState(const vector<double> &x, int t) = 0;
};
typedef boost::shared_ptr<DynamicsObject> DynamicsObjectPtr;

class Contact {
public:
  string m_name;
  const string &getName() const { return m_name; }
  Contact(const string &name) : m_name(name) { }
  virtual ~Contact() { }

  virtual void fillVarNamesAndBounds(vector<string> &out_names, vector<double> &out_vlower, vector<double> &out_vupper, const string &name_prefix) = 0;
  virtual void fillInitialSolution(vector<double> &out) = 0;
  virtual int setVariables(const vector<Var> &vars, int start_pos) = 0;
  virtual void addConstraintsToModel() = 0;

  virtual AffExpr getForceExpr(int t, int i) = 0;
  virtual vector<DynamicsObject*> getAffectedObjects() = 0;
};
typedef boost::shared_ptr<Contact> ContactPtr;

class DynamicsProblem : public OptProb {
public:
  OR::EnvironmentBasePtr m_env;
  CollisionCheckerPtr m_cc;
  vector<DynamicsObjectPtr> m_objects;
  vector<ContactPtr> m_contacts;

  DynamicsProblem(OR::EnvironmentBasePtr env) : m_env(env) { }
  virtual ~DynamicsProblem() { }

  int m_timesteps;
  double m_dt;
  Vector3d m_gravity;
  void setNumTimesteps(int n) { m_timesteps = n; }
  void setDt(double dt) { m_dt = dt; }
  void setGravity(const Vector3d &g) { m_gravity = g; }

  void addObject(DynamicsObjectPtr obj) { m_objects.push_back(obj); }
  void addContact(ContactPtr contact) { m_contacts.push_back(contact); }

  void setUpProblem();
  vector<double> makeInitialSolution();
};
typedef boost::shared_ptr<DynamicsProblem> DynamicsProblemPtr;

struct DynamicsOptResult {
  BasicTrustRegionSQPPtr optimizer;
  OptStatus status;
};
typedef boost::shared_ptr<DynamicsOptResult> DynamicsOptResultPtr;

DynamicsOptResultPtr OptimizeDynamicsProblem(DynamicsProblemPtr prob, bool plotting=false);

} // namespace dynamics
} // namespace trajopt
