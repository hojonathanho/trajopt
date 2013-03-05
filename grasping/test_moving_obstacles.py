import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--interactive", action="store_true")
parser.add_argument("--position_only", action="store_true")
parser.add_argument("--moving_scene", action="store_true")
args = parser.parse_args()

import openravepy as rave
import trajoptpy
import trajoptpy.kin_utils as ku
import trajoptpy.make_kinbodies as mk
import bulletsimpy
import json
import numpy as np

env = rave.Environment()
env.StopSimulation()
env.Load("robots/pr2-beta-static.zae")
env.Load("../data/table.xml")

#bullet_env = bulletsimpy.LoadFromRave(env, 'table')
#bullet_env.SetGravity([0, 0, -9.8])
#
#dyn_obj_names = []
#dyn_objs = [bullet_env.GetObjectByName(name) for name in dyn_obj_names]

trajoptpy.SetInteractive(args.interactive) # pause every iteration, until you press 'p'. Press escape to disable further plotting
if args.interactive and args.moving_scene:
  viewer = trajoptpy.GetViewer(env)
  viewer.Step()
  viewer.SetKinBodyTransparency(env.GetKinBody('table'), 0.1)

robot = env.GetRobots()[0]
joint_start = [-1.832, -0.332, -1.011, -1.437, -1.1  , -2.106,  3.074]
robot.SetDOFValues(joint_start, robot.GetManipulator('rightarm').GetArmIndices())

quat_target = [0.98555024,  0.12101977,  0.10129305, -0.06151951] # wxyz
xyz_target = [0.65540048, -0.34836676,  0.44726639]
hmat_target = rave.matrixFromPose( np.r_[quat_target, xyz_target] )

# BEGIN ik
manip = robot.GetManipulator("rightarm")
init_joint_target = ku.ik_for_link(hmat_target, manip, "r_gripper_tool_frame",
    filter_options = rave.IkFilterOptions.CheckEnvCollisions)
# END ik

request = {
  "basic_info" : {
    "n_steps" : 10,
    "manip" : "rightarm", # see below for valid values
    "start_fixed" : True # i.e., DOF values at first timestep are fixed based on current robot state
  },
  "costs" : [
  {
    "type" : "joint_vel", # joint-space velocity cost
    "params": {"coeffs" : [1]} # a list of length one is automatically expanded to a list of length n_dofs
    # Also valid: "coeffs" : [7,6,5,4,3,2,1]
  },
  {
    "type" : "continuous_collision",
    "name" :"cont_coll", # shorten name so printed table will be prettier
    "params" : {
      "coeffs" : [20], # penalty coefficients. list of length one is automatically expanded to a list of length n_timesteps
      "dist_pen" : [0.025] # robot-obstacle distance that penalty kicks in. expands to length n_timesteps
    }
  }
  ],
  "constraints" : [
  # BEGIN pose_constraint
  {
    "type" : "pose", 
    "params" : {"xyz" : xyz_target, 
                "wxyz" : quat_target, 
                "link": "r_gripper_tool_frame",
                # "timestep" : 9
                # omitted because timestep = n_steps-1 is default
                # "pos_coeffs" : [1,1,1], # omitted because that's default
                "rot_coeffs" : ([0,0,0] if args.position_only else [1,1,1])
                }
                 
  }
  # END pose_constraint
  ],
  # BEGIN init
  "init_info" : {
      "type" : "straight_line", # straight line in joint space.
      "endpoint" : init_joint_target.tolist() # need to convert numpy array to list
  }
  # END init
}
if args.moving_scene:
  request["scene_states"] = [
    { "timestep": 0, "obj_states": [{"name": "table", "xyz": [0, 0.01*5, 0.00-0.1], "wxyz": [1, 0, 0, 0]}] },
    { "timestep": 1, "obj_states": [{"name": "table", "xyz": [0, 0.02*5, 0.01-0.1], "wxyz": [1, 0, 0, 0]}] },
    { "timestep": 2, "obj_states": [{"name": "table", "xyz": [0, 0.03*5, 0.03-0.1], "wxyz": [1, 0, 0, 0]}] },
    { "timestep": 3, "obj_states": [{"name": "table", "xyz": [0, 0.04*5, 0.04-0.1], "wxyz": [1, 0, 0, 0]}] },
    { "timestep": 4, "obj_states": [{"name": "table", "xyz": [0, 0.05*5, 0.05-0.1], "wxyz": [1, 0, 0, 0]}] },
    { "timestep": 5, "obj_states": [{"name": "table", "xyz": [0, 0.06*5, 0.06-0.1], "wxyz": [1, 0, 0, 0]}] },
    { "timestep": 6, "obj_states": [{"name": "table", "xyz": [0, 0.07*5, 0.07-0.1], "wxyz": [1, 0, 0, 0]}] },
    { "timestep": 7, "obj_states": [{"name": "table", "xyz": [0, 0.08*5, 0.08-0.1], "wxyz": [1, 0, 0, 0]}] },
    { "timestep": 8, "obj_states": [{"name": "table", "xyz": [0, 0.09*5, 0.09-0.1], "wxyz": [1, 0, 0, 0]}] },
    { "timestep": 9, "obj_states": [{"name": "table", "xyz": [0, 0.10*5, 0.10-0.1], "wxyz": [1, 0, 0, 0]}] },
  ]

s = json.dumps(request) # convert dictionary into json-formatted string
prob = trajoptpy.ConstructProblem(s, env) # create object that stores optimization problem
result = trajoptpy.OptimizeProblem(prob) # do optimization
print result

from trajoptpy.check_traj import traj_is_safe
prob.SetRobotActiveDOFs() # set robot DOFs to DOFs in optimization problem
assert traj_is_safe(result.GetTraj(), robot) # Check that trajectory is collision free

# Now we'll check to see that the final constraint was satisfied
robot.SetActiveDOFValues(result.GetTraj()[-1])
posevec = rave.poseFromMatrix(robot.GetLink("r_gripper_tool_frame").GetTransform())
quat, xyz = posevec[0:4], posevec[4:7]

quat *= np.sign(quat.dot(quat_target))
if args.position_only:
    assert (quat - quat_target).max() > 1e-3
else:
    assert (quat - quat_target).max() < 1e-3

