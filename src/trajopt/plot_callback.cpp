#include "trajopt/plot_callback.hpp"
#include "trajopt/common.hpp"
#include "osgviewer/osgviewer.hpp"
#include "utils/eigen_conversions.hpp"
#include <boost/foreach.hpp>
#include "trajopt/problem_description.hpp"
#include "trajopt/collision_avoidance.hpp"
#include "trajopt/scene_objectives.hpp"
#include <set>
using namespace OpenRAVE;
using namespace util;
namespace trajopt {

static int gTrajPlayPos;
static vector<vector<GraphHandlePtr> > gTrajHandles;
static vector<GraphHandlePtr> gSceneTrajHandles;
static bool gCallbacksRegistered = false;

static const float TRAJ_DEFAULT_TRANSPARENCY = .35;
static const float TRAJ_ACTIVE_TRANSPARENCY = 1;
static const float TRAJ_INACTIVE_TRANSPARENCY = .1;

static const float TRAJ_SCENE_DEFAULT_TRANSPARENCY = .2;
static const float TRAJ_SCENE_ACTIVE_TRANSPARENCY = .3;
static const float TRAJ_SCENE_INACTIVE_TRANSPARENCY = .1;

static void PlotTraj(OSGViewer& viewer, RobotAndDOF& rad, const vector<SceneStateInfoPtr> &scene_states, const TrajArray& x, vector<vector<GraphHandlePtr> >& handles, vector<GraphHandlePtr>& scene_handles) {
  RobotBase::RobotStateSaver saver = rad.Save();
  std::set<KinBodyPtr> bodies;
  rad.GetRobot()->GetAttached(bodies);
  for (int i=0; i < x.rows(); ++i) {
    rad.SetDOFValues(toDblVec(x.row(i)));

    vector<GraphHandlePtr> local_handles;
    BOOST_FOREACH(const KinBodyPtr& body, bodies) {
      local_handles.push_back(viewer.PlotKinBody(body));
      SetTransparency(local_handles.back(), TRAJ_DEFAULT_TRANSPARENCY);
    }
    handles.push_back(local_handles);

    if (!scene_states.empty()) {
      assert(scene_states.size() == x.rows());
      SceneStateSetter sss(rad.GetRobot()->GetEnv(), scene_states[i]);
      vector<KinBodyPtr> kinbodies_to_plot;
      BOOST_FOREACH(ObjectStateInfoPtr& o, scene_states[i]->obj_state_infos) {
        kinbodies_to_plot.push_back(rad.GetRobot()->GetEnv()->GetKinBody(o->name));
        //std::cout << i << ' ' << o->name << ' ' << o->xyz.transpose() << ' ' << o->wxyz.transpose() << std::endl;
      }
      scene_handles.push_back(viewer.PlotKinBodies(kinbodies_to_plot));
      SetTransparency(scene_handles.back(), TRAJ_SCENE_DEFAULT_TRANSPARENCY);
    }
  }
}

static void PlotCosts(OSGViewer& viewer, TrajOptProb& prob, const DblVec& x) {
  vector<CostPtr>& costs = prob.getCosts();
  vector<ConstraintPtr> cnts = prob.getConstraints();
  RobotAndDOF& rad = *prob.GetRAD();
  const VarArray& vars = prob.GetVars();

  vector<SceneStateInfoPtr> scene_states;
  if (prob.GetSceneStates().empty() && prob.HasSimulation()) {
    scene_states = Simulation::GetOrCreate(prob)->GetResult()->ToSceneStateInfos();
    std::cout << "plotting: got scene states from simulation " << scene_states.size() << std::endl;
  } else if (!prob.GetSceneStates().empty()) {
    scene_states = prob.GetSceneStates();
  }

  vector<GraphHandlePtr> handles;
  BOOST_FOREACH(CostPtr& cost, costs) {
    if (Plotter* plotter = dynamic_cast<Plotter*>(cost.get())) {
      plotter->Plot(x, *rad.GetRobot()->GetEnv(), handles);
    }
  }
  BOOST_FOREACH(ConstraintPtr& cnt, cnts) {
    if (Plotter* plotter = dynamic_cast<Plotter*>(cnt.get())) {
      plotter->Plot(x, *rad.GetRobot()->GetEnv(), handles);
    }
  }
  gTrajHandles.clear(); gSceneTrajHandles.clear();
  gTrajPlayPos = -1;
  TrajArray traj = getTraj(x, vars);
  PlotTraj(viewer, rad, scene_states, traj, gTrajHandles, gSceneTrajHandles);
  viewer.Idle();
  rad.SetDOFValues(toDblVec(traj.row(traj.rows()-1)));
  gTrajHandles.clear(); gSceneTrajHandles.clear();
}

static void TrajPlayCallback(OSGViewerPtr viewer, char c) {
  switch (c) {
  case '[':
    gTrajPlayPos = std::max(gTrajPlayPos - 1, 0);
    break;
  case ']':
    gTrajPlayPos = std::min(gTrajPlayPos + 1, (int) gTrajHandles.size() - 1);
    break;
  case '{':
    gTrajPlayPos = 0;
    break;
  case '}':
    gTrajPlayPos = gTrajHandles.size() - 1;
    break;
  case '\\':
    gTrajPlayPos = -1;
    break;
  default:
    assert(false);
  }

  for (int i = 0; i < gTrajHandles.size(); ++i) {
    float trans = TRAJ_DEFAULT_TRANSPARENCY;
    if (gTrajPlayPos != -1) {
      trans = gTrajPlayPos == i ? TRAJ_ACTIVE_TRANSPARENCY : TRAJ_INACTIVE_TRANSPARENCY;
    }
    for (int j = 0; j < gTrajHandles[i].size(); ++j) {
      SetTransparency(gTrajHandles[i][j], trans);
    }

    if (!gSceneTrajHandles.empty()) {
      trans = TRAJ_SCENE_DEFAULT_TRANSPARENCY;
      if (gTrajPlayPos != -1) {
        trans = gTrajPlayPos == i ? TRAJ_SCENE_ACTIVE_TRANSPARENCY : TRAJ_SCENE_INACTIVE_TRANSPARENCY;
      }
      SetTransparency(gSceneTrajHandles[i], trans);
    }
  }

  viewer->Draw();
}

static void RegisterTrajPlayCallbacks(OSGViewerPtr viewer) {
  if (gCallbacksRegistered) {
    return;
  }
  viewer->AddKeyCallback('[', boost::bind(&TrajPlayCallback, viewer, '['), "Trajectory playback -- back one step");
  viewer->AddKeyCallback(']', boost::bind(&TrajPlayCallback, viewer, ']'), "Trajectory playback -- forward one step");
  viewer->AddKeyCallback('{', boost::bind(&TrajPlayCallback, viewer, '{'), "Trajectory playback -- go to beginning");
  viewer->AddKeyCallback('}', boost::bind(&TrajPlayCallback, viewer, '}'), "Trajectory playback -- go to end");
  viewer->AddKeyCallback('\\', boost::bind(&TrajPlayCallback, viewer, '\\'), "Trajectory playback -- exit playback mode");
  gCallbacksRegistered = true;
}

Optimizer::Callback PlotCallback(TrajOptProb& prob) {
  OSGViewerPtr viewer = OSGViewer::GetOrCreate(prob.GetEnv());
  RegisterTrajPlayCallbacks(viewer);
  return boost::bind(&PlotCosts, boost::ref(*viewer), boost::ref(prob), _2);
}

}
