#pragma once
#include "trajopt/common.hpp"
#include "trajopt/collision_checker.hpp"
#include "sco/modeling.hpp"
#include "sco/sco_fwd.hpp"
#include "cache.hxx"


namespace trajopt {

typedef std::map<const OR::KinBody::Link*, int> Link2Int;
typedef std::vector< std::pair<string, string> > NamePairs;
typedef std::map<string, double> Str2Dbl;

class SceneStateInfo;
typedef boost::shared_ptr<SceneStateInfo> SceneStateInfoPtr;

class SceneStateSetter {
public:
  SceneStateSetter(OR::EnvironmentBasePtr env, SceneStateInfoPtr new_state);
  ~SceneStateSetter();

private:
  vector<KinBody::KinBodyStateSaver*> m_savers;
};
typedef boost::shared_ptr<SceneStateSetter> SceneStateSetterPtr;


struct CollisionEvaluator {
  virtual void CalcDistExpressions(const DblVec& x, vector<AffExpr>& exprs, DblVec& weights, NamePairs& bodyNames) = 0;
  virtual void CalcDists(const DblVec& x, DblVec& exprs, DblVec& weights, NamePairs& bodyNames) = 0;
  virtual void CalcCollisions(const DblVec& x, vector<Collision>& collisions) = 0;
  void GetCollisionsCached(const DblVec& x, vector<Collision>&);
  virtual ~CollisionEvaluator() {}

  Cache<double, vector<Collision>, 3> m_cache;
};
typedef boost::shared_ptr<CollisionEvaluator> CollisionEvaluatorPtr;

struct SingleTimestepCollisionEvaluator : public CollisionEvaluator {
public:
  SingleTimestepCollisionEvaluator(RobotAndDOFPtr rad, SceneStateInfoPtr scene_state, const VarVector& vars);
  /**
  @brief linearize all contact distances in terms of robot dofs
  
  Do a collision check between robot and environment.
  For each contact generated, return a linearization of the signed distance function
  Since the collision checker may return multiple contact points for a given pair of links,
  the contacts are associated with weights so that each pair of links are associated with a single cost term.
  In particular, if a pair of bodies have k contacts, then the contacts each have weight 1/k.
  */
  void CalcDistExpressions(const DblVec& x, vector<AffExpr>& exprs, DblVec& weights, NamePairs& bodyNames); // appends to this vector
  /**
   * Same as CalcDistExpressions, but just the distances--not the expressions
   */
  void CalcDists(const DblVec& x, DblVec& exprs, DblVec& weights, NamePairs& bodyNames); // appends to this vector
  void CalcCollisions(const DblVec& x, vector<Collision>& collisions);

  OR::EnvironmentBasePtr m_env;
  CollisionCheckerPtr m_cc;
  RobotAndDOFPtr m_rad;
  SceneStateInfoPtr m_scene_state;
  VarVector m_vars;
  Link2Int m_link2ind;
  vector<OR::KinBody::LinkPtr> m_links;

};

struct InterpolatedCollisionEvaluator : public CollisionEvaluator {
public:
  InterpolatedCollisionEvaluator(RobotAndDOFPtr rad, const VarVector& vars0, const VarVector& vars1);
  void CalcDistExpressions(const DblVec& x, vector<AffExpr>& exprs, DblVec& weights); // appends to this vector
  void CalcDists(const DblVec& x, DblVec& exprs, DblVec& weights); // appends to this vector

  OR::EnvironmentBasePtr m_env;
  CollisionCheckerPtr m_cc;
  RobotAndDOFPtr m_rad;
  VarVector m_vars0;
  VarVector m_vars1;
  typedef std::map<const OR::KinBody::Link*, int> Link2Int;
  Link2Int m_link2ind;
  vector<OR::KinBody::LinkPtr> m_links;

};

struct CastCollisionEvaluator : public CollisionEvaluator {
public:
  CastCollisionEvaluator(RobotAndDOFPtr rad, SceneStateInfoPtr scene_state, const VarVector& vars0, const VarVector& vars1);
  void CalcDistExpressions(const DblVec& x, vector<AffExpr>& exprs, DblVec& weights, NamePairs& bodyNames); // appends to this vector
  void CalcDists(const DblVec& x, DblVec& exprs, DblVec& weights, NamePairs& bodyNames); // appends to this vector
  void CalcCollisions(const DblVec& x, vector<Collision>& collisions);

  // parameters:
  OR::EnvironmentBasePtr m_env;
  CollisionCheckerPtr m_cc;
  RobotAndDOFPtr m_rad;
  SceneStateInfoPtr m_scene_state;
  VarVector m_vars0;
  VarVector m_vars1;
  typedef std::map<const OR::KinBody::Link*, int> Link2Int;
  Link2Int m_link2ind;
  vector<OR::KinBody::LinkPtr> m_links;

};


class CollisionCost : public Cost, public Plotter {
public:
  /* constructor for single timestep */
  CollisionCost(double dist_pen, double coeff, RobotAndDOFPtr rad, SceneStateInfoPtr scene_state, const VarVector& vars);
  /* constructor for cast cost */
  CollisionCost(double dist_pen, double coeff, RobotAndDOFPtr rad, SceneStateInfoPtr scene_state, const VarVector& vars0, const VarVector& vars1);
  virtual ConvexObjectivePtr convex(const vector<double>& x, Model* model);
  virtual double value(const vector<double>&);
  virtual void Plot(const DblVec& x, OR::EnvironmentBase&, std::vector<OR::GraphHandlePtr>& handles);
private:
  CollisionEvaluatorPtr m_calc;
  double m_dist_pen;
  double m_coeff;
};

class CollisionTaggedCost : public Cost, public Plotter {
public:
  CollisionTaggedCost(const Str2Dbl& tag2dist_pen, const Str2Dbl& tag2coeff, RobotAndDOFPtr rad, SceneStateInfoPtr scene_state, const VarVector& vars);
  CollisionTaggedCost(const Str2Dbl& tag2dist_pen, const Str2Dbl& tag2coeff, RobotAndDOFPtr rad, SceneStateInfoPtr scene_state, const VarVector& vars0, const VarVector& vars1);
  virtual ConvexObjectivePtr convex(const vector<double>& x, Model* model);
  virtual double value(const vector<double>&);
  virtual void Plot(const DblVec& x, OR::EnvironmentBase&, std::vector<OR::GraphHandlePtr>& handles);
private:
  CollisionEvaluatorPtr m_calc;
  Str2Dbl m_tag2dist_pen;
  Str2Dbl m_tag2coeff;
};

}
