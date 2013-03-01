#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <vector>
#include "trajopt/typedefs.hpp"
#include "trajopt/rave_utils.hpp"
#include "trajopt/collision_checker.hpp"
#include "sco/optimizers.hpp"
#include "sco/expr_op_overloads.hpp"
#include "util.h"

namespace trajopt {
namespace dynamics {

using namespace std;
using namespace Eigen;

class DynamicsProblem;
struct OptimizationBase;
typedef boost::shared_ptr<OptimizationBase> OptimizationBasePtr;
struct Spec {
  virtual ~Spec() { }
  virtual OptimizationBasePtr realize(DynamicsProblem *prob) = 0;
  virtual void setState0From(const vector<double> &x, int t) = 0;
};
typedef boost::shared_ptr<Spec> SpecPtr;

struct ProblemSpec {
  int timesteps;
  double dt;
  Vector3d gravity;

  vector<SpecPtr> objects;
  vector<SpecPtr> contacts;
};

class OptimizationBase {
public:
  OptimizationBase(const string &name, DynamicsProblem *prob) : m_name(name), m_prob(prob) { }
  virtual ~OptimizationBase() { }
  const string &getName() const { return m_name; }
  DynamicsProblem *getProb() const { return m_prob; }

  enum Type {
    TYPE_OBJECT = 0, TYPE_CONTACT
  };
  virtual Type getType() const = 0;

  virtual void fillVarNamesAndBounds(vector<string> &out_names, vector<double> &out_vlower, vector<double> &out_vupper, const string &name_prefix) = 0;
  virtual void fillInitialSolution(vector<double> &out) = 0;
  virtual int setVariables(const vector<Var> &vars, int start_pos) = 0;
  virtual void addConstraintsToModel() = 0;

private:
  string m_name;
  DynamicsProblem *m_prob;
};

class Contact;
class DynamicsObject : public OptimizationBase {
public:
  DynamicsObject(const string &name, DynamicsProblem *prob) : OptimizationBase(name, prob) { }
  virtual ~DynamicsObject() { }

  Type getType() const { return TYPE_OBJECT; }
  virtual void registerContact(Contact *) = 0;
  virtual void addToRave() = 0;
  virtual void setRaveState(const vector<double> &x, int t) = 0;
};
typedef boost::shared_ptr<DynamicsObject> DynamicsObjectPtr;

class Contact : public OptimizationBase {
public:
  Contact(const string &name, DynamicsProblem *prob) : OptimizationBase(name, prob) { }
  virtual ~Contact() { }

  Type getType() const { return TYPE_CONTACT; }
  virtual AffExpr getForceExpr(DynamicsObject *o, int t, int i) = 0;
  virtual vector<DynamicsObject*> getAffectedObjects() = 0;
};
typedef boost::shared_ptr<Contact> ContactPtr;


class DynamicsProblem : public OptProb {
public:
  OR::EnvironmentBasePtr m_env;
  CollisionCheckerPtr m_cc;
  vector<DynamicsObjectPtr> m_objects;
  vector<ContactPtr> m_contacts;
  ProblemSpec m_spec;

  DynamicsProblem(OR::EnvironmentBasePtr env, const ProblemSpec &spec) : m_env(env), m_spec(spec) { }
  virtual ~DynamicsProblem() { }

  void addObject(DynamicsObjectPtr obj) { m_objects.push_back(obj); }
  void addContact(ContactPtr contact) { m_contacts.push_back(contact); }

  DynamicsObjectPtr findObject(const string &name);
  ContactPtr findContact(const string &name);

  void setUpProblem();
  vector<double> makeInitialSolution();
};
typedef boost::shared_ptr<DynamicsProblem> DynamicsProblemPtr;

struct DynamicsOptResult {
  BasicTrustRegionSQPPtr optimizer;
  OptStatus status;
};
typedef boost::shared_ptr<DynamicsOptResult> DynamicsOptResultPtr;


DynamicsProblemPtr CreateDynamicsProblem(OR::EnvironmentBasePtr env, ProblemSpec &spec);
DynamicsOptResultPtr OptimizeDynamicsProblem(DynamicsProblemPtr prob, const vector<double> *init_soln=NULL, bool plotting=false);


} // namespace dynamics
} // namespace trajopt
