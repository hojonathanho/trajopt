import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--interactive", action="store_true")
args = parser.parse_args()

import openravepy
import trajoptpy
import json
env = openravepy.Environment()
env.StopSimulation()
env.Load('test2.xml')

trajoptpy.SetInteractive(args.interactive) # pause every iteration, until you press 'p'. Press escape to disable further plotting

robot = env.GetRobots()[0]

robot.SetDOFValues([1.361, 0, 0, 0, 0, 0, 0], robot.GetManipulator('leftarm').GetArmIndices())
robot.SetDOFValues([0.7], [34])
robot.SetDOFValues([0.29, 0, 0, 0, 0, 0, 0], robot.GetManipulator('rightarm').GetArmIndices())

joint_target = [-0.639, 0, 0, 0, 0, 0, 0]

robot = ["PR2"]
dynamic_objects = ["c1", "c2"]
static_objects = ["table", "upper-obstacle1", "upper-obstacle2"]

cost_params = []

for name in robot:
    cost_params.append({
        "name": name,
        "coeffs": [1],
        "dist_pen": [0.025],
    })

for name in dynamic_objects:
    cost_params.append({
        "name" : name,
        "coeffs" : [1],
        "dist_pen" : [0.025],
    })

for name in static_objects:
    cost_params.append({
        "name" : name,
        "coeffs" : [20],
        "dist_pen" : [0.025],
    }) 

request = {
    "basic_info": {
        "n_steps": 5,
        "manip": "rightarm",
        "start_fixed": True,
    },

    "costs": [
        {
            "type" : "joint_vel",
            "params": {"coeffs" : [1]},
        },

        {
            "type": "continuous_collision",
            "name": "cont_collision",
            "params": {
                "object_costs": cost_params,
            }
        }
    ],
    "constraints": [
        {
            "type": "joint",
            "params": {"vals": joint_target},
        },
    ],
    "init_info": {
        "type": "straight_line",
        "endpoint": joint_target,
    }
}

#robot.SetDOFValues(
#        [  
#         0, 0, 0,
#         0, 0, 0,
#         0, 0, 0,
#         0, 0, 0,
#         0, 0, 0,
#         0, 0.7, 0,
#         0, 0, 0,
#         0, 0.5, 0,
#         0, 0, 0,
#        -1, 1.18, -0.44,
#         0, 0, 0,
#         0, 0, 0,
#         0, 0, 0
#        ]
#)


#robot.SetDOFValues(
#        [0, 0, 0,
#         0, 0, 0,
#         0, 0, 0,
#         0, 0, 0,
#         0, 0, 0,
#        -0.21, -0.075, 0,
#         0, 0.0, 0,
#         0, 0.5, 0,
#         0, 0, 0,
#        -1, 1.18, -0.44,
#         0, 0, 0,
#         0, 0, 0,
#         0, 0, 0]
#)

s = json.dumps(request) # convert dictionary into json-formatted string
prob = trajoptpy.ConstructProblem(s, env) # create object that stores optimization problem
result = trajoptpy.OptimizeProblem(prob) # do optimization
print result

from trajoptpy.check_traj import traj_is_safe
prob.SetRobotActiveDOFs() # set robot DOFs to DOFs in optimization problem
assert traj_is_safe(result.GetTraj(), robot) # Check that trajectory is collision free
