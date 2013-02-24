#include "dynamics_problem.hpp"

#include "osgviewer/osgviewer.hpp"
#include "utils/stl_to_string.hpp"

namespace trajopt {
namespace dynamics {

void DynamicsProblem::setUpProblem() {
  for (ContactPtr &c : m_contacts) {
    for (DynamicsObject *o : c->getAffectedObjects()) {
      o->registerContact(c.get());
    }
  }

  vector<double> vlower, vupper;
  vector<string> names;
  for (DynamicsObjectPtr &o : m_objects) {
    o->fillVarNamesAndBounds(names, vlower, vupper, o->getName());
  }
  for (ContactPtr &c : m_contacts) {
    c->fillVarNamesAndBounds(names, vlower, vupper, c->getName());
  }
  assert(names.size() == vlower.size() && names.size() == vupper.size());
  createVariables(names, vlower, vupper);

  vector<Var> &vars = getVars();
  int k = 0;
  for (DynamicsObjectPtr &o : m_objects) {
    k = o->setVariables(vars, k);
  }
  for (ContactPtr &c : m_contacts) {
    k = c->setVariables(vars, k);
  }
  assert(k == vars.size());
  cout << Str(names) << endl;

  for (DynamicsObjectPtr &o : m_objects) {
    o->addConstraintsToModel();
    o->addToRave();
  }
  m_env->UpdatePublishedBodies();
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

DynamicsOptResultPtr OptimizeDynamicsProblem(DynamicsProblemPtr prob, bool plotting) {
  prob->setUpProblem();

  if (plotting) {
    OSGViewerPtr viewer(new OSGViewer(prob->m_env));
    viewer->Idle();
  }

  BasicTrustRegionSQPPtr optimizer(new BasicTrustRegionSQP(prob));
  optimizer->min_trust_box_size_ = 1e-7;
  optimizer->min_approx_improve_= 1e-7;
  optimizer->cnt_tolerance_ = 1e-7;
  optimizer->trust_box_size_ = 1;
  optimizer->max_iter_ = 1000;
  optimizer->max_merit_coeff_increases_= 100;

  optimizer->initialize(prob->makeInitialSolution());
  OptStatus status = optimizer->optimize();

  DynamicsOptResultPtr result(new DynamicsOptResult);
  result->optimizer = optimizer;
  result->status = status;
  return result;
}

} // namespace dynamics
} // namespace trajopt
