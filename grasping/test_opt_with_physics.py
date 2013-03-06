import trajoptpy
import trajoptpy.kin_utils as ku
import openravepy as rave
import bulletsimpy
import physics
import optimization
import simple_env
from pprint import pprint
import numpy as np
import json

env = simple_env.create_topple_env()
#env.SetViewer('qtcoin')
pr2 = env.GetRobot('pr2')

dyn_obj_names = ['box_0']
static_obj_names = ['table'] # cyl_0 not included since we'll collide with it in the final pose

bullet_env = bulletsimpy.BulletEnvironment(env, dyn_obj_names)
bullet_env.SetGravity([0, 0, -9.8])
dyn_objs = [bullet_env.GetObjectByName(n) for n in dyn_obj_names]
bullet_box0 = bullet_env.GetObjectByName('box_0')
bullet_cyl0 = bullet_env.GetObjectByName('cyl_0')

# simulate for a few steps first to stabilize
for i in range(20):
  bullet_env.Step(0.01, 100, 0.01)
for o in dyn_objs:
  env.GetKinBody(o.GetName()).SetTransform(o.GetTransform())
env.UpdatePublishedBodies()


N_STEPS = 20
MANIP_NAME = "rightarm"
quat_target = list(physics.get_bulletobj_state(bullet_cyl0)['wxyz'])
xyz_target = list(physics.get_bulletobj_state(bullet_cyl0)['xyz'])
print 'target:', quat_target, xyz_target
hmat_target = rave.matrixFromPose( np.r_[quat_target, xyz_target] )
# BEGIN ik
manip = pr2.GetManipulator("rightarm")
init_joint_target = ku.ik_for_link(hmat_target, manip, "r_gripper_tool_frame")#, filter_options = rave.IkFilterOptions.CheckEnvCollisions)
# END ik


def create_request(curr_iter, prev_traj, prev_scene_states):
  object_costs = []
  for name in dyn_obj_names:
    object_costs.append({
      "name" : name,
      "coeffs" : [.1],
      "dist_pen" : [0.025],
    })
  for name in static_obj_names:
    object_costs.append({
      "name" : name,
      "coeffs" : [20],
      "dist_pen" : [0.025],
    })

  request = {
    "basic_info" : {
      "n_steps" : N_STEPS,
      "manip" : MANIP_NAME,
      "start_fixed" : True # i.e., DOF values at first timestep are fixed based on current robot state
    },
    "costs" : [
    {
      "type" : "joint_vel", # joint-space velocity cost
      #"params": {"coeffs" : [10 if curr_iter == 0 else 1]} # a list of length one is automatically expanded to a list of length n_dofs
      "params": {"coeffs" : [10]} # a list of length one is automatically expanded to a list of length n_dofs
      # Also valid: "coeffs" : [7,6,5,4,3,2,1]
    },
    {
      "type" : "continuous_collision",
      "name" :"cont_coll", # shorten name so printed table will be prettier
      "params" : { "object_costs": object_costs }
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
                  "rot_coeffs" : [1,1,1]
                  }
    }
    # END pose_constraint
    ],
  }

  if prev_scene_states is not None:
    request["scene_states"] = prev_scene_states

  if prev_traj is not None:
    request["init_info"] = {
      "type": "given_traj",
      "data": prev_traj.tolist()
    }
  else:
    request["init_info"] = {
      "type": "straight_line", # straight line in joint space
      "endpoint": init_joint_target.tolist()
    }

  return request



p = optimization.OptParams()
p.traj_time = 5
p.dynamic_obj_names = dyn_obj_names

print optimization.opt_and_sim_loop(p, create_request, env)





# objects to record
#rec_obj_names = dyn_obj_names
#rec_objs = [bullet_env.GetObjectByName(name) for name in rec_obj_names]
#rec = physics.record_sim(bullet_env, rec_objs, n_timesteps=50, update_rave_env=True, pause_per_iter=True)
#pprint(rec)


#viewer = trajoptpy.GetViewer(env)
#viewer.Step()
#viewer.Idle()
#s = json.dumps(request) # convert dictionary into json-formatted string
#trajoptpy.SetInteractive(True)
#prob = trajoptpy.ConstructProblem(s, env) # create object that stores optimization problem
#result = trajoptpy.OptimizeProblem(prob) # do optimization


# now copy the environment and run the trajectory with physics
#env_copy = env.CloneSelf(rave.CloningOptions.Bodies)
#env_copy.SetViewer('qtcoin')
#env_copy.UpdatePublishedBodies()
#raw_input('asdf')
#bullet_env = bulletsimpy.LoadFromRave(env, dyn_obj_names)
#bullet_env.SetGravity([0, 0, -9.8])
#rec_obj_names = dyn_obj_names
#rec_objs = [bullet_env.GetObjectByName(name) for name in rec_obj_names]
#rec = physics.record_sim_with_traj(prob, 'pr2', result.GetTraj(), 5, bullet_env, rec_objs, update_rave_env=False)
#pprint(rec)
#
#raw_input('asdf')

# optimize again with moving obstacles
#object_costs = []
#for name in dyn_obj_names:
#  object_costs.append({
#    "name" : name,
#    "coeffs" : [.1],
#    "dist_pen" : [0.025],
#  })
#for name in static_obj_names:
#  object_costs.append({
#    "name" : name,
#    "coeffs" : [20],
#    "dist_pen" : [0.025],
#  })
#request = {
#  "basic_info" : {
#    "n_steps" : N_STEPS,
#    "manip" : MANIP_NAME,
#    "start_fixed" : True # i.e., DOF values at first timestep are fixed based on current robot state
#  },
#  "costs" : [
#  {
#    "type" : "joint_vel", # joint-space velocity cost
#    "params": {"coeffs" : [1]} # a list of length one is automatically expanded to a list of length n_dofs
#  },
#  {
#    "type" : "continuous_collision",
#    "name" :"cont_coll", # shorten name so printed table will be prettier
#    "params" : { "object_costs": object_costs }
#    #"params" : {
#    #  "coeffs" : [20], # penalty coefficients. list of length one is automatically expanded to a list of length n_timesteps
#    #  "dist_pen" : [0.025] # robot-obstacle distance that penalty kicks in. expands to length n_timesteps
#    #}
#  }
#  ],
#  "constraints" : [
#  # BEGIN pose_constraint
#  {
#    "type" : "pose", 
#    "params" : {"xyz" : xyz_target, 
#                "wxyz" : quat_target, 
#                "link": "r_gripper_tool_frame",
#                "rot_coeffs" : [1,1,1]
#                }
#                 
#  }
#  # END pose_constraint
#  ],
#  # BEGIN init
#  "init_info" : {
#      "type" : "straight_line", # straight line in joint space.
#      "endpoint" : init_joint_target.tolist() # need to convert numpy array to list
#  },
#  "scene_states": rec
#  # END init
#}
#s = json.dumps(request) # convert dictionary into json-formatted string
#trajoptpy.SetInteractive(True)
#for name in dyn_obj_names:
#  trajoptpy.GetViewer(env).SetKinBodyTransparency(env.GetKinBody(name), 0.01)
#prob = trajoptpy.ConstructProblem(s, env) # create object that stores optimization problem
#result = trajoptpy.OptimizeProblem(prob) # do optimization
