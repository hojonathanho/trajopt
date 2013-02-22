#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <vector>
#include "trajopt/typedefs.hpp"
#include "trajopt/rave_utils.hpp"
#include "trajopt/collision_checker.hpp"
#include "ipi/sco/expr_op_overloads.hpp"
#include "util.h"

namespace trajopt {
namespace dynamics {

using namespace std;
using namespace Eigen;

class DynamicsObject {
public:
  string m_name;
  const string &getName() const { return m_name; }
  DynamicsObject(const string &name) : m_name(name) { }
  virtual ~DynamicsObject() { }

  virtual void fillVarNamesAndBounds(vector<string> &out_names, vector<double> &out_vlower, vector<double> &out_vupper, const string &name_prefix) = 0;
  virtual int setVariables(const vector<Var> &vars, int start_pos) = 0;
  virtual void addConstraintsToModel() = 0;
  virtual void addToRave() = 0;
};
typedef boost::shared_ptr<DynamicsObject> DynamicsObjectPtr;

class Contact : public DynamicsObject {
public:
  Contact(const string &name) : DynamicsObject(name) { }
  virtual ~Contact() { }
  virtual AffExpr getForceExpr(int t, int i) = 0;
  void addToRave() { }
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
};
typedef boost::shared_ptr<DynamicsProblem> DynamicsProblemPtr;


} // namespace dynamics
} // namespace trajopt
