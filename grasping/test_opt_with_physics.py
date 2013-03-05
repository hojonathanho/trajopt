import trajoptpy
import trajoptpy.kin_utils as ku
import openravepy as rave
import bulletsimpy
import physics
import simple_env
from pprint import pprint
import numpy as np
import json

env = simple_env.create()
#env.SetViewer('qtcoin')
pr2 = env.GetRobot('pr2')
raw_input('blah')

dyn_obj_names = ['box_0']

bullet_env = bulletsimpy.LoadFromRave(env, dyn_obj_names)
bullet_env.SetGravity([0, 0, -9.8])
print 'gravity:',bullet_env.GetGravity()
dyn_objs = [bullet_env.GetObjectByName(n) for n in dyn_obj_names]
bullet_box0 = bullet_env.GetObjectByName('box_0')
bullet_cyl0 = bullet_env.GetObjectByName('cyl_0')

# simulate for a few steps first to stabilize
for i in range(20):
  bullet_env.Step(0.01, 100, 0.01)

for o in dyn_objs:
  env.GetKinBody(o.GetName()).SetTransform(o.GetTransform())
env.UpdatePublishedBodies()
raw_input('blah')

# objects to record
#rec_obj_names = dyn_obj_names
#rec_objs = [bullet_env.GetObjectByName(name) for name in rec_obj_names]
#rec = physics.record_sim(bullet_env, rec_objs, n_timesteps=50, update_rave_env=True, pause_per_iter=True)
#pprint(rec)


N_STEPS = 10
MANIP_NAME = "rightarm"

quat_target = list(physics.get_bulletobj_state(bullet_cyl0)['wxyz'])
xyz_target = list(physics.get_bulletobj_state(bullet_cyl0)['xyz'])
print 'target:', quat_target, xyz_target
hmat_target = rave.matrixFromPose( np.r_[quat_target, xyz_target] )
# BEGIN ik
manip = pr2.GetManipulator("rightarm")
init_joint_target = ku.ik_for_link(hmat_target, manip, "r_gripper_tool_frame")#, filter_options = rave.IkFilterOptions.CheckEnvCollisions)
# END ik
request = {
  "basic_info" : {
    "n_steps" : N_STEPS,
    "manip" : MANIP_NAME,
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
                "rot_coeffs" : [1,1,1]
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
viewer = trajoptpy.GetViewer(env)
viewer.Step()
viewer.Idle()
s = json.dumps(request) # convert dictionary into json-formatted string
trajoptpy.SetInteractive(True)
prob = trajoptpy.ConstructProblem(s, env) # create object that stores optimization problem
result = trajoptpy.OptimizeProblem(prob) # do optimization
