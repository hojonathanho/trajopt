#include "trajopt/plot_callback.hpp"
#include "trajopt/common.hpp"
#include "osgviewer/osgviewer.hpp"
#include "utils/eigen_conversions.hpp"
#include <boost/foreach.hpp>
#include "trajopt/problem_description.hpp"
using namespace OpenRAVE;
using namespace util;
namespace trajopt {

static int gTrajPlayPos;
static vector<GraphHandlePtr> gTrajHandles;

static const float TRAJ_DEFAULT_TRANSPARENCY = .35;
static const float TRAJ_ACTIVE_TRANSPARENCY = 1;
static const float TRAJ_INACTIVE_TRANSPARENCY = .1;

static void PlotTraj(OSGViewer& viewer, RobotAndDOF& rad, const TrajArray& x, vector<GraphHandlePtr>& handles) {
  RobotBase::RobotStateSaver saver = rad.Save();
  for (int i=0; i < x.rows(); ++i) {
    rad.SetDOFValues(toDblVec(x.row(i)));
    handles.push_back(viewer.PlotKinBody(rad.GetRobot()));
    SetTransparency(handles.back(), TRAJ_DEFAULT_TRANSPARENCY);
  }
}

static void PlotCosts(OSGViewer& viewer, vector<CostPtr>& costs, vector<ConstraintPtr>& cnts, RobotAndDOF& rad, const VarArray& vars, const DblVec& x) {
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
  gTrajHandles.clear();
  gTrajPlayPos = -1;
  TrajArray traj = getTraj(x, vars);
  PlotTraj(viewer, rad, traj, gTrajHandles);
  viewer.Idle();
  rad.SetDOFValues(toDblVec(traj.row(traj.rows()-1)));
  gTrajHandles.clear();
}

static void TrajPlayCallback(char c) {
  switch (c) {
  case '[':
    gTrajPlayPos = std::min(gTrajPlayPos + 1, (int) gTrajHandles.size() - 1);
    break;
  case ']':
    gTrajPlayPos = std::max(gTrajPlayPos - 1, 0);
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
    SetTransparency(gTrajHandles[i], trans);
  }
}

static void RegisterTrajPlayCallbacks(OSGViewerPtr viewer) {
  viewer->AddKeyCallback('[', boost::bind(&TrajPlayCallback, '['), "Trajectory playback -- back one step");
  viewer->AddKeyCallback(']', boost::bind(&TrajPlayCallback, ']'), "Trajectory playback -- forward one step");
  viewer->AddKeyCallback('{', boost::bind(&TrajPlayCallback, '{'), "Trajectory playback -- go to beginning");
  viewer->AddKeyCallback('}', boost::bind(&TrajPlayCallback, '}'), "Trajectory playback -- go to end");
  viewer->AddKeyCallback('\\', boost::bind(&TrajPlayCallback, '\\'), "Trajectory playback -- exit playback mode");
}

Optimizer::Callback PlotCallback(TrajOptProb& prob) {
  OSGViewerPtr viewer = OSGViewer::GetOrCreate(prob.GetEnv());
  RegisterTrajPlayCallbacks(viewer);
  vector<ConstraintPtr> cnts = prob.getConstraints();
  return boost::bind(&PlotCosts, boost::ref(*viewer),
                      boost::ref(prob.getCosts()),
                      cnts,
                      boost::ref(*prob.GetRAD()),
                      boost::ref(prob.GetVars()),
                      _2);
}

}
