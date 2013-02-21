#include "dynamics_problem.hpp"

#include "utils/stl_to_string.hpp"
namespace trajopt {
namespace dynamics {


static const double GROUND_Z = 0.0;

void DynamicsProblem::setUpProblem() {
  const int n_objects = m_objects.size();
  vector<double> vlower, vupper;
  vector<string> names;

  for (DynamicsObjectPtr &o : m_objects) {
    o->registerGroundContact();
    o->fillVarNamesAndBounds(names, vlower, vupper);
  }

  assert(names.size() == vlower.size() && names.size() == vupper.size());

  createVariables(names, vlower, vupper);

  vector<Var> vars = getVars();
  int k = 0;
  for (DynamicsObjectPtr &o : m_objects) {
    k = o->setVariables(vars, k);
  }
  assert(k == vars.size());

  cout << Str(names) << endl;

  for (DynamicsObjectPtr &o : m_objects) {
    o->addConstraintsToModel();
    o->addGroundNonpenetrationCnts(GROUND_Z);
    //box->addToRave();
  }

  m_cc = CollisionChecker::GetOrCreate(*m_env);
}


} // namespace dynamics
} // namespace trajopt
