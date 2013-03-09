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
      "params": {"coeffs" : [10 if curr_iter == 0 else 1]} # a list of length one is automatically expanded to a list of length n_dofs
      #"params": {"coeffs" : [10]} # a list of length one is automatically expanded to a list of length n_dofs
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
p.max_iter = 2

traj = optimization.opt_and_sim_loop(p, env, create_request)
raw_input('about to display traj')
optimization.display_traj(p, env, traj)

traj2 = optimization.direct_optimize(p, env, traj)
#traj2 = np.array( \
#[[-1.92677209, -0.34917486, -1.06330054, -1.51133815, -1.15690464, -2.21494652, -3.23302261],
# [-1.82624602, -0.31279597, -1.03464544, -1.48218664, -0.94340623, -2.13724251, -3.19246927],
# [-1.72572019, -0.27641697, -1.00599024, -1.45303558, -0.7299072 , -2.0595387 ,-3.15191593],
# [-1.62519437, -0.24003783, -0.97733497, -1.42388507, -0.51640765, -1.98183497, -3.11136259],
# [-1.52466831, -0.20365862, -0.9486797 , -1.3947351 , -0.30290782, -1.90413125, -3.07080926],
# [-1.42414187, -0.16727953, -0.92002446, -1.36558555, -0.08940796, -1.82642752, -3.03025592],
# [-1.32361508, -0.13090068, -0.89136932, -1.33643625,  0.12409174, -1.74872377, -2.98970258],
# [-1.22308809, -0.09452207, -0.86271432, -1.30728705,  0.33759119, -1.67102003, -2.94914924],
# [-1.12256104, -0.05814355, -0.83405951, -1.2781379 ,  0.55109041, -1.59331637, -2.9085959 ],
# [-1.02203401, -0.02176496, -0.80540485, -1.24898876,  0.76458946, -1.51561281, -2.86804257],
# [-0.92150702,  0.01461384, -0.77675027, -1.21983966,  0.97808841, -1.43790938, -2.82748923],
# [-0.82098007,  0.05099286, -0.74809567, -1.19069055,  1.19158736, -1.36020602, -2.78693589],
# [-0.72045318,  0.08737202, -0.71944099, -1.16154136,  1.40508636, -1.28250268, -2.74638255],
# [-0.61992636,  0.1237512 , -0.69078619, -1.13239201,  1.61858547, -1.20479925, -2.70582922],
# [-0.51939966,  0.16013029, -0.66213125, -1.10324241,  1.83208471, -1.12709568, -2.66527588],
# [-0.41887311,  0.19650922, -0.6334762 , -1.07409252,  2.04558405, -1.04939196, -2.62472254],
# [-0.31834688,  0.23288801, -0.60482093, -1.04494237,  2.25908342, -0.97168817, -2.5841692 ],
# [-0.21951178,  0.26926676, -0.57616422, -1.01579179,  2.47258271, -0.89398445, -2.54361587],
# [-0.10834813,  0.30735349, -0.55172249, -0.98963051,  2.68952646, -0.81613293, -2.50306253],
# [ 0.08679787,  0.36221963, -0.54904252, -0.97475146,  2.90491333, -0.73826486, -2.46250919]])
print 'traj2'
print traj2, traj2.shape
print 'traj'
print traj, traj.shape

print 'about to display traj 2'
optimization.display_traj(p, env, traj2)
