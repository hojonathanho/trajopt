#include "dynamics_problem.hpp"

#include "osgviewer/osgviewer.hpp"
#include "utils/stl_to_string.hpp"
#include "utils/logging1.hpp"

namespace trajopt {
namespace dynamics {

DynamicsObjectPtr DynamicsProblem::findObject(const string &name) {
  for (DynamicsObjectPtr &o : m_objects) {
    if (o->getName() == name) {
      return o;
    }
  }
  LOG_ERROR("object %s not found!", name.c_str());
  return DynamicsObjectPtr();
}

ContactPtr DynamicsProblem::findContact(const string &name) {
  for (ContactPtr &c : m_contacts) {
    if (c->getName() == name) {
      return c;
    }
  }
  LOG_ERROR("contact %s not found!", name.c_str());
  return ContactPtr();
}

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


DynamicsProblemPtr CreateDynamicsProblem(OR::EnvironmentBasePtr env, ProblemSpec &spec) {
  DynamicsProblemPtr prob(new DynamicsProblem(env, spec));
  for (SpecPtr &obj_spec : spec.objects) {
    DynamicsObjectPtr o = boost::dynamic_pointer_cast<DynamicsObject>(obj_spec->realize(prob.get()));
    prob->addObject(o);
    LOG_INFO("Created object %s", o->getName().c_str());
  }
  for (SpecPtr &contact_spec : spec.contacts) {
    prob->addContact(boost::dynamic_pointer_cast<Contact>(contact_spec->realize(prob.get())));
  }
//  switch (o->getType()) {
//    case OptimizationBase::TYPE_OBJECT:
//      prob->addObject(boost::dynamic_pointer_cast<DynamicsObject>(o));
//      break;
//    case OptimizationBase::TYPE_CONTACT:
//      prob->addContact(boost::dynamic_pointer_cast<Contact>(o));
//      break;
//    default:
//      assert(false);
//      break;
//    }
//  }
  return prob;
}

DynamicsOptResultPtr OptimizeDynamicsProblem(DynamicsProblemPtr prob, const vector<double> *init_soln, bool plotting) {
  if (prob->getNumCosts() == 0) {
    prob->addCost(CostPtr(new ZeroCost())); // shut up
  }
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
  //optimizer->max_merit_coeff_increases_= 10;

  optimizer->initialize(init_soln == NULL ? prob->makeInitialSolution() : *init_soln);
  OptStatus status = optimizer->optimize();

  DynamicsOptResultPtr result(new DynamicsOptResult);
  result->optimizer = optimizer;
  result->status = status;
  return result;
}

DynamicsOptResultPtr OptimizeStepLoop(OR::EnvironmentBasePtr env, ProblemSpec &spec) {
  DynamicsProblemPtr prob;

  ProblemSpec tmp_spec = spec;
  tmp_spec.timesteps = 2;

  vector<double> init_soln;

  while () {
    env->Reset();
    prob = CreateDynamicsProblem(env, tmp_spec);
    if (init_soln.size() == 0) {
      init_soln = prob->makeInitialSolution();
    }
    DynamicsOptResultPtr result = OptimizeDynamicsProblem(prob, &init_soln, true);

    // set tmp_spec's state_0 and init_trajs to be the state of timestep 1 of the result of the current problem
    for (SpecPtr &o : tmp_spec.objects) {
      o->setState0From(result->optimizer->x(), 1);
    }
    init_soln = result->optimizer->x();
  }

  if (prob->getNumCosts() == 0) {
    prob->addCost(CostPtr(new ZeroCost())); // shut up
  }
  const int total_timesteps = prob->m_timesteps;

  DynamicsProblemPtr tmp_prob;

  for (int t = 1; t < total_timesteps; ++t) {
    tmp_prob.reset(new DynamicsProblem(*prob));
    tmp_prob->m_timesteps = 2;
    tmp_prob->setUpProblem();
  }

  OSGViewerPtr viewer;
  if (plotting) {
    viewer.reset(new OSGViewer(prob->m_env));
    viewer->Idle();
  }

  vector<double> curr_x = prob->makeInitialSolution();

  for (int t = 1; t < total_timesteps; ++t) {
    BasicTrustRegionSQPPtr optimizer(new BasicTrustRegionSQP(prob));
    optimizer->min_trust_box_size_ = 1e-7;
    optimizer->min_approx_improve_= 1e-7;
    optimizer->cnt_tolerance_ = 1e-7;
    optimizer->trust_box_size_ = 1;
    optimizer->max_iter_ = 1000;

    optimizer->initialize(curr_x);
    OptStatus status = optimizer->optimize();

    curr_x = optimizer->x();
    for (DynamicsObjectPtr &o : prob->m_objects) {
      o->setRaveState(curr_x, 1);
    }
    prob->m_env->UpdatePublishedBodies();
    if (viewer) {
      viewer->Idle();
    }
  }

  //optimizer->max_merit_coeff_increases_= 10;


}

} // namespace dynamics
} // namespace trajopt
