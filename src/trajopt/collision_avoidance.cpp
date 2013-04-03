#include "trajopt/collision_avoidance.hpp"
#include "trajopt/rave_utils.hpp"
#include "trajopt/utils.hpp"
#include "sco/expr_vec_ops.hpp"
#include "sco/expr_ops.hpp"
#include "sco/sco_common.hpp"
#include <boost/foreach.hpp>
#include "utils/eigen_conversions.hpp"
#include "sco/modeling_utils.hpp"
#include "utils/stl_to_string.hpp"
#include "trajopt/problem_description.hpp"
using namespace OpenRAVE;
using namespace sco;
using namespace util;
using namespace std;

namespace trajopt {


SceneStateSetter::SceneStateSetter(OR::EnvironmentBasePtr env, SceneStateInfoPtr new_state) {
  BOOST_FOREACH(ObjectStateInfoPtr& o, new_state->obj_state_infos) {
    KinBodyPtr body = env->GetKinBody(o->name);
    m_savers.push_back(new KinBody::KinBodyStateSaver(body));
    body->SetTransform(toRaveTransform(o->wxyz, o->xyz));
  }
}
SceneStateSetter::~SceneStateSetter() {
  BOOST_FOREACH(KinBody::KinBodyStateSaver* s, m_savers) {
    delete s;
  }
}


void CollisionsToDistances(const vector<Collision>& collisions, const Link2Int& m_link2ind,
    DblVec& dists, DblVec& weights, NamePairs& bodyNames) {
  // Note: this checking (that the links are in the list we care about) is probably unnecessary
  // since we're using LinksVsAll
  dists.clear();
  weights.clear();
  dists.reserve(collisions.size());
  weights.reserve(collisions.size());
  BOOST_FOREACH(const Collision& col, collisions) {
    Link2Int::const_iterator itA = m_link2ind.find(col.linkA);
    Link2Int::const_iterator itB = m_link2ind.find(col.linkB);
    if (itA != m_link2ind.end() || itB != m_link2ind.end()) {
      dists.push_back(col.distance);
      weights.push_back(col.weight);
      bodyNames.push_back(pair<string, string>(col.linkA->GetParent()->GetName(), col.linkB->GetParent()->GetName()));
    }
  }
}

void CollisionsToDistanceExpressions(const vector<Collision>& collisions, RobotAndDOF& rad, SceneStateInfoPtr scene_state,
    const Link2Int& link2ind, const VarVector& vars, const DblVec& dofvals, vector<AffExpr>& exprs, DblVec& weights, NamePairs& bodyNames) {

  exprs.clear();
  weights.clear();
  exprs.reserve(collisions.size());
  weights.reserve(collisions.size());

  SceneStateSetterPtr state_setter;
  if (scene_state) state_setter.reset(new SceneStateSetter(rad.GetRobot()->GetEnv(), scene_state));

  rad.SetDOFValues(dofvals); // since we'll be calculating jacobians
  BOOST_FOREACH(const Collision& col, collisions) {
    AffExpr dist(col.distance);
    Link2Int::const_iterator itA = link2ind.find(col.linkA);
    if (itA != link2ind.end()) {
      VectorXd dist_grad = toVector3d(col.normalB2A).transpose()*rad.PositionJacobian(itA->second, col.ptA);
      exprInc(dist, varDot(dist_grad, vars));
      exprInc(dist, -dist_grad.dot(toVectorXd(dofvals)));
    }
    Link2Int::const_iterator itB = link2ind.find(col.linkB);
    if (itB != link2ind.end()) {
      VectorXd dist_grad = -toVector3d(col.normalB2A).transpose()*rad.PositionJacobian(itB->second, col.ptB);
      exprInc(dist, varDot(dist_grad, vars));
      exprInc(dist, -dist_grad.dot(toVectorXd(dofvals)));
    }
    if (itA != link2ind.end() || itB != link2ind.end()) {
      exprs.push_back(dist);
      weights.push_back(col.weight);
      bodyNames.push_back(pair<string, string>(col.linkA->GetParent()->GetName(), col.linkB->GetParent()->GetName()));
    }
  }
  RAVELOG_DEBUG("%i distance expressions\n", exprs.size());
}

void CollisionsToDistanceExpressions(const vector<Collision>& collisions, RobotAndDOF& rad, SceneStateInfoPtr scene_state, const Link2Int& link2ind,
    const VarVector& vars0, const VarVector& vars1, const DblVec& vals0, const DblVec& vals1,
    vector<AffExpr>& exprs, DblVec& weights, NamePairs& bodyNames) {
  vector<AffExpr> exprs0, exprs1;
  DblVec weights0, weights1;
  CollisionsToDistanceExpressions(collisions, rad, scene_state, link2ind, vars0, vals0, exprs0, weights0, bodyNames);
  CollisionsToDistanceExpressions(collisions, rad, scene_state, link2ind, vars1, vals1, exprs1, weights1, bodyNames);

  exprs.resize(exprs0.size());
  weights.resize(exprs0.size());

  for (int i=0; i < exprs0.size(); ++i) {
    exprScale(exprs0[i], (1-collisions[i].time));
    exprScale(exprs1[i], collisions[i].time);
    exprs[i] = AffExpr(0);
    exprInc(exprs[i], exprs0[i]);
    exprInc(exprs[i], exprs1[i]);
    weights[i] = (weights0[i] + weights1[i])/2;
  }
}

void CollisionEvaluator::GetCollisionsCached(const DblVec& x, vector<Collision>& collisions) {
  double key = vecSum(x);
  vector<Collision>* it = m_cache.get(key);
  if (it != NULL) {
    RAVELOG_DEBUG("using cached collision check\n");
    collisions = *it;
  }
  else {
    RAVELOG_DEBUG("not using cached collision check\n");
    CalcCollisions(x, collisions);
    m_cache.put(key, collisions);
  }
}

SingleTimestepCollisionEvaluator::SingleTimestepCollisionEvaluator(RobotAndDOFPtr rad, SceneStateInfoPtr scene_state, const VarVector& vars) :
  m_env(rad->GetRobot()->GetEnv()),
  m_cc(CollisionChecker::GetOrCreate(*m_env)),
  m_rad(rad),
  m_scene_state(scene_state),
  m_vars(vars),
  m_link2ind(),
  m_links() {
  RobotBasePtr robot = rad->GetRobot();
  const vector<KinBody::LinkPtr>& robot_links = robot->GetLinks();
  vector<KinBody::LinkPtr> links;
  vector<int> inds;
  rad->GetAffectedLinks(m_links, true, inds);
  for (int i=0; i < m_links.size(); ++i) {
    m_link2ind[m_links[i].get()] = inds[i];
  }
}


void SingleTimestepCollisionEvaluator::CalcCollisions(const DblVec& x, vector<Collision>& collisions) {
  DblVec dofvals = getDblVec(x, m_vars);
  SceneStateSetterPtr state_setter;
  if (m_scene_state) state_setter.reset(new SceneStateSetter(m_env, m_scene_state));
  m_rad->SetDOFValues(dofvals);
  m_cc->LinksVsAll(m_links, collisions);
}

void SingleTimestepCollisionEvaluator::CalcDists(const DblVec& x, DblVec& dists, DblVec& weights, NamePairs& bodyNames) {
  vector<Collision> collisions;
  GetCollisionsCached(x, collisions);
  CollisionsToDistances(collisions, m_link2ind, dists, weights, bodyNames);
}

void SingleTimestepCollisionEvaluator::CalcDistExpressions(const DblVec& x, vector<AffExpr>& exprs, DblVec& weights, NamePairs& bodyNames) {
  vector<Collision> collisions;
  GetCollisionsCached(x, collisions);
  DblVec dofvals = getDblVec(x, m_vars);
  CollisionsToDistanceExpressions(collisions, *m_rad, m_scene_state, m_link2ind, m_vars, dofvals, exprs, weights, bodyNames);
}

////////////////////////////////////////

CastCollisionEvaluator::CastCollisionEvaluator(RobotAndDOFPtr rad, SceneStateInfoPtr scene_state, const VarVector& vars0, const VarVector& vars1) :
  m_env(rad->GetRobot()->GetEnv()),
  m_cc(CollisionChecker::GetOrCreate(*m_env)),
  m_rad(rad),
  m_scene_state(scene_state),
  m_vars0(vars0),
  m_vars1(vars1),
  m_link2ind(),
  m_links() {
  RobotBasePtr robot = rad->GetRobot();
  const vector<KinBody::LinkPtr>& robot_links = robot->GetLinks();
  vector<KinBody::LinkPtr> links;
  vector<int> inds;
  rad->GetAffectedLinks(m_links, true, inds);
  for (int i=0; i < m_links.size(); ++i) {
    m_link2ind[m_links[i].get()] = inds[i];
  }
}

void CastCollisionEvaluator::CalcCollisions(const DblVec& x, vector<Collision>& collisions) {
  DblVec dofvals0 = getDblVec(x, m_vars0);
  DblVec dofvals1 = getDblVec(x, m_vars1);
  SceneStateSetterPtr state_setter;
  if (m_scene_state) state_setter.reset(new SceneStateSetter(m_env, m_scene_state)); // TODO: use casts for the scene objects too?
  m_rad->SetDOFValues(dofvals0);
  m_cc->CastVsAll(*m_rad, m_links, dofvals0, dofvals1, collisions);
}
void CastCollisionEvaluator::CalcDistExpressions(const DblVec& x, vector<AffExpr>& exprs, DblVec& weights, NamePairs& bodyNames) {
  vector<Collision> collisions;
  GetCollisionsCached(x, collisions);
  DblVec dofvals0 = getDblVec(x, m_vars0);
  DblVec dofvals1 = getDblVec(x, m_vars1);
  CollisionsToDistanceExpressions(collisions, *m_rad, m_scene_state, m_link2ind, m_vars0, m_vars1, dofvals0, dofvals1, exprs, weights, bodyNames);
}
void CastCollisionEvaluator::CalcDists(const DblVec& x, DblVec& dists, DblVec& weights, NamePairs& bodyNames) {
  vector<Collision> collisions;
  GetCollisionsCached(x, collisions);
  CollisionsToDistances(collisions, m_link2ind, dists, weights, bodyNames);
}

//////////////////////////////////////////



typedef OpenRAVE::RaveVector<float> RaveVectorf;

void PlotCollisions(const std::vector<Collision>& collisions, OR::EnvironmentBase& env, vector<OR::GraphHandlePtr>& handles, double safe_dist) {
  BOOST_FOREACH(const Collision& col, collisions) {
    RaveVectorf color;
    if (col.distance < 0) color = RaveVectorf(1,0,0,1);
    else if (col.distance < safe_dist) color = RaveVectorf(1,1,0,1);
    else color = RaveVectorf(0,1,0,1);
    handles.push_back(env.drawarrow(col.ptA, col.ptB, .0025, color));
  }
}

void PlotCollisions(const std::vector<Collision>& collisions, OR::EnvironmentBase& env, vector<OR::GraphHandlePtr>& handles, const vector<double>& safe_dists) {
  int i = 0;
  BOOST_FOREACH(const Collision& col, collisions) {
    RaveVectorf color;
    if (col.distance < 0) color = RaveVectorf(1,0,0,1);
    else if (col.distance < safe_dists[i]) color = RaveVectorf(1,1,0,1);
    else color = RaveVectorf(0,1,0,1);
    handles.push_back(env.drawarrow(col.ptA, col.ptB, .0025, color));
    ++i;
  }
}

CollisionCost::CollisionCost(double dist_pen, double coeff, RobotAndDOFPtr rad, SceneStateInfoPtr scene_state, const VarVector& vars) :
    Cost("collision"),
    m_tagged(false),
    m_calc(new SingleTimestepCollisionEvaluator(rad, scene_state, vars)), m_dist_pen(dist_pen), m_coeff(coeff)
{}

CollisionCost::CollisionCost(const Str2Dbl& tag2dist_pen, const Str2Dbl& tag2coeff, RobotAndDOFPtr rad, SceneStateInfoPtr scene_state, const VarVector& vars) :
    Cost("tagged_collision"),
    m_tagged(true),
    m_calc(new SingleTimestepCollisionEvaluator(rad, scene_state, vars)), m_tag2dist_pen(tag2dist_pen), m_tag2coeff(tag2coeff)
{}


CollisionCost::CollisionCost(double dist_pen, double coeff, RobotAndDOFPtr rad, SceneStateInfoPtr scene_state, const VarVector& vars0, const VarVector& vars1) :
    Cost("cast_collision"),
    m_tagged(false),
    m_calc(new CastCollisionEvaluator(rad, scene_state, vars0, vars1)), m_dist_pen(dist_pen), m_coeff(coeff)
{}

CollisionCost::CollisionCost(const Str2Dbl& tag2dist_pen, const Str2Dbl& tag2coeff, RobotAndDOFPtr rad, SceneStateInfoPtr scene_state, const VarVector& vars0, const VarVector& vars1) :
    Cost("tagged_cast_collision"),
    m_tagged(true),
    m_calc(new CastCollisionEvaluator(rad, scene_state, vars0, vars1)), m_tag2dist_pen(tag2dist_pen), m_tag2coeff(tag2coeff)
{}


ConvexObjectivePtr CollisionCost::convex(const vector<double>& x, Model* model) {
  ConvexObjectivePtr out(new ConvexObjective(model));
  vector<AffExpr> exprs;
  DblVec weights;
  NamePairs bodyNames;
  m_calc->CalcDistExpressions(x, exprs, weights, bodyNames);
  for (int i=0; i < exprs.size(); ++i) {
    double dist_pen = m_tagged ? min(m_tag2dist_pen[bodyNames[i].first], m_tag2dist_pen[bodyNames[i].second]) : m_dist_pen;
    double coeff = m_tagged ? (m_tag2coeff[bodyNames[i].first] + m_tag2coeff[bodyNames[i].second]) : m_coeff;
    AffExpr viol = exprSub(AffExpr(dist_pen), exprs[i]);
    out->addHinge(viol, coeff*weights[i]);
  }
  return out;
}
double CollisionCost::value(const vector<double>& x) {
  DblVec dists, weights;
  NamePairs bodyNames;
  m_calc->CalcDists(x, dists, weights, bodyNames);
  double out = 0;
  for (int i=0; i < dists.size(); ++i) {
    double dist_pen = m_tagged ? min(m_tag2dist_pen[bodyNames[i].first], m_tag2dist_pen[bodyNames[i].second]) : m_dist_pen;
    double coeff = m_tagged ? (m_tag2coeff[bodyNames[i].first] + m_tag2coeff[bodyNames[i].second]) : m_coeff;
    out += pospart(dist_pen - dists[i]) * coeff * weights[i];
  }
  return out;
}

void CollisionCost::Plot(const DblVec& x, OR::EnvironmentBase& env, std::vector<OR::GraphHandlePtr>& handles) {
  vector<Collision> collisions;
  m_calc->GetCollisionsCached(x, collisions);
  if (m_tagged) {
    vector<double> dist_pens;
    BOOST_FOREACH(const Collision& col, collisions) {
      dist_pens.push_back(min(m_tag2dist_pen[col.linkA->GetName()], m_tag2dist_pen[col.linkB->GetName()]));
    }
    PlotCollisions(collisions, env, handles, dist_pens);
  } else {
    PlotCollisions(collisions, env, handles, m_dist_pen);
  }
}

}
