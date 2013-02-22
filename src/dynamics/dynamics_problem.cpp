#include "dynamics_problem.hpp"

#include "utils/stl_to_string.hpp"
namespace trajopt {
namespace dynamics {

void DynamicsProblem::setUpProblem() {
  // Set up objects
  vector<double> vlower, vupper;
  vector<string> names;
  for (DynamicsObjectPtr &o : m_objects) {
    //o->registerGroundContact();
    o->fillVarNamesAndBounds(names, vlower, vupper, o->getName());
  }
  assert(names.size() == vlower.size() && names.size() == vupper.size());
  createVariables(names, vlower, vupper);
  vector<Var> &vars = getVars();
  int k = 0;
  for (DynamicsObjectPtr &o : m_objects) {
    k = o->setVariables(vars, k);
  }
  assert(k == vars.size());
  cout << Str(names) << endl;
  for (DynamicsObjectPtr &o : m_objects) {
    o->addConstraintsToModel();
    //o->addGroundNonpenetrationCnts(GROUND_Z);
    o->addToRave();
  }
  m_env->UpdatePublishedBodies();

  // Set up constraints
  vlower.clear(); vupper.clear(); names.clear();
  for (ContactPtr &c : m_contacts) {
    c->fillVarNamesAndBounds(names, vlower, vupper, c->getName());
  }
  assert(names.size() == vlower.size() && names.size() == vupper.size());
  createVariables(names, vlower, vupper);
  assert(vars.size() == k + names.size());
  for (ContactPtr &c : m_contacts) {
    k = c->setVariables(vars, k);
  }
  assert(k == vars.size());
  for (ContactPtr &c : m_contacts) {
    c->addConstraintsToModel();
  }

  m_cc = CollisionChecker::GetOrCreate(*m_env);
}


vector<double> DynamicsProblem::makeInitialSolution() {
  vector<double> v;
  for (DynamicsObjectPtr &o : m_objects) {
    o->fillInitialSolution(v);
  }
  for (ContactPtr &c : m_contacts) {
    c->fillInitialSolution(v);
  }
  assert(v.size() == getVars().size());
  return v;
}


} // namespace dynamics
} // namespace trajopt
