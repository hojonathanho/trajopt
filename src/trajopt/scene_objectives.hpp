#pragma once

#include "problem_description.hpp"
#include "sco/optimizers.hpp"
#include "simulation/bulletsim_lite.h"
#include <Eigen/Dense>

namespace trajopt {

using namespace OpenRAVE;
using namespace Eigen;

struct ObjectTraj {
  MatrixX3d xyz;
  MatrixX4d wxyz;

  ObjectTraj(int len) : xyz(len, 3), wxyz(len, 4) { }
  OR::Transform GetTransform(int t) const;
};
typedef boost::shared_ptr<ObjectTraj> ObjectTrajPtr;

typedef std::map<string, ObjectTrajPtr> Name2ObjectTraj;
typedef vector<bs::CollisionPtr> CollisionVec;
struct SimResult {
  Name2ObjectTraj obj_trajs;
  vector<CollisionVec> collisions;
  TrajArray robot_traj;

  vector<SceneStateInfoPtr> ToSceneStateInfos();
  void Clear();
};
typedef boost::shared_ptr<SimResult> SimResultPtr;


class Simulation;
typedef boost::shared_ptr<Simulation> SimulationPtr;
class Simulation : public OR::UserData {
public:
  virtual ~Simulation() { }

  void SetSimParams(const SimParamsInfo&);
  void RunTraj(const TrajArray&);

  void PreEvaluateCallback(const DblVec&);
  Optimizer::Callback MakePreEvaluateCallback();

  SimResultPtr GetResult();
  SimResultPtr GetResultUpsampled();

  static SimulationPtr GetOrCreate(TrajOptProb& prob);

private:
  TrajOptProb& m_prob;
  RobotAndDOFPtr m_rad;
  EnvironmentBasePtr m_env;
  SimParamsInfo m_params;
  int m_runs_executed;
  SimResultPtr m_curr_result, m_curr_result_upsampled;

  Simulation(TrajOptProb& prob);
};

#if 0
class SimulationPlotterDummyCost : public Cost, public Plotter {
public:
  SimulationPlotterDummyCost(SimulationPtr sim) : m_sim(sim) { }
  virtual ConvexObjectivePtr convex(const vector<double>&, Model* model) { return ConvexObjectivePtr(new ConvexObjective(model)); }
  virtual double value(const vector<double>&) { return 0.; }
  virtual void Plot(const DblVec& x, OR::EnvironmentBase&, std::vector<OR::GraphHandlePtr>& handles);

private:
  SimulationPtr m_sim;
};
#endif


class ObjectSlideCost : public Cost, public Plotter {
public:
  ObjectSlideCost(int timestep, const string& object_name, double dist_pen, double coeff, RobotAndDOFPtr rad, const VarVector& vars0, const VarVector& vars1, SimulationPtr sim);

  virtual ConvexObjectivePtr convex(const vector<double>& x, Model* model);
  virtual double value(const vector<double>&);
  virtual void Plot(const DblVec& x, OR::EnvironmentBase&, std::vector<OR::GraphHandlePtr>& handles);

private:
  int m_timestep;
  string m_object_name;
  double m_dist_pen;
  double m_coeff;
  RobotAndDOFPtr m_rad;
  VarVector m_vars0, m_vars1;
  SimulationPtr m_sim;

  bool GetCollisionData(bs::CollisionPtr& out_c0, bs::CollisionPtr& out_c1, Vector3d &out_n);
};


} // namespace trajopt
