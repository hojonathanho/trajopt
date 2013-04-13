import openravepy as rave
import numpy as np
import trajoptpy.math_utils as mu
import bulletsimpy
import physics
import rec_util
import trajoptpy
import json

def create_request(n_steps, manip_name, xyz_target, quat_target, init_joint_target):
  request = {
    "basic_info" : {
      "n_steps" : n_steps,
      "manip" : manip_name,
      "start_fixed" : True, # i.e., DOF values at first timestep are fixed based on current robot state
      "dynamic_objects": ['box_0'],
    },
    "costs" : [
    {
      "type" : "joint_vel", # joint-space velocity cost
      "params": {"coeffs" : [1]} # a list of length one is automatically expanded to a list of length n_dofs
    },
    {
      "type": "object_slide",
      "params": { "object_name": "box_0", "coeffs": [1], "dist_pen": [.025] }
    },
#   {
#     "type" : "continuous_collision",
#     "name" :"cont_coll", # shorten name so printed table will be prettier
#     "params" : { "object_costs": object_costs }
#   }
    ],
    "constraints" : [
    # BEGIN pose_constraint
    {
      "type" : "pose", 
      "params" : {"xyz" : xyz_target.tolist(), 
                  "wxyz" : quat_target.tolist(), 
                  "link": "r_gripper_tool_frame",
                  # "timestep" : 9
                  # omitted because timestep = n_steps-1 is default
                  # "pos_coeffs" : [1,1,1], # omitted because that's default
                  "rot_coeffs" : [1,1,1]
                  }
    }
    # END pose_constraint
    ],
    "init_info": {
      "type": "straight_line",
      "endpoint": init_joint_target.tolist()
    }
  }
  return request

def main():
  ### setup ###
  import simple_env
  from trajoptpy import make_kinbodies as mk
  env = simple_env.create_bare()

  table = env.GetKinBody('table')
  table_aabb = table.ComputeAABB()
  table_top_z = table_aabb.pos()[2] + table_aabb.extents()[2]
  table_mid = table_aabb.pos()[:2]

  box_center = table_mid - [.5, .4]
  box_lwh = [0.1, 0.4, 0.2]
  mk.create_box_from_bounds(env, [-box_lwh[0]/2., box_lwh[0]/2., -box_lwh[1]/2., box_lwh[1]/2., -box_lwh[2]/2., box_lwh[2]/2.], name='box_0')
  box = env.GetKinBody('box_0')
  final_box_center = np.array([box_center[0], box_center[1], table_top_z+box_lwh[2]/2.])
  box.SetTransform(rave.matrixFromPose(np.r_[[1, 0, 0, 0], final_box_center]))

  robot = env.GetRobot('pr2')
  manip = robot.GetManipulator('rightarm')
  #env.SetViewer('qtcoin')
  handles = []

  ### make straight-line trajectory ###
  traj_len = 20
  start_pt = final_box_center - np.array([.1, 0, 0])
  final_pt = start_pt + np.array([.2, 0, .2])
  line = mu.linspace2d(start_pt, final_pt, traj_len)
  handles.append(env.drawlinelist(np.array([start_pt,final_pt]),1))
  env.UpdatePublishedBodies()

  robot.SetActiveManipulator(manip.GetName())
  ikmodel = rave.databases.inversekinematics.InverseKinematicsModel(robot,iktype=rave.IkParameterization.Type.Transform6D)
  if not ikmodel.load(): ikmodel.autogenerate()

  #Tstart = manip.GetTransform()
  #line_traj = np.empty((traj_len, len(manip.GetArmIndices())))
  #orig_joints = robot.GetDOFValues()
  #for t in range(traj_len):
  #  T = Tstart.copy()
  #  T[0:3,3] = line[t,:]
  #  sol = manip.FindIKSolution(T, 0)#, rave.IkFilterOptions.CheckEnvCollisions)
  #  line_traj[t,:] = sol
  #  print t, sol
  #  robot.SetDOFValues(sol, manip.GetArmIndices())
  #  env.UpdatePublishedBodies()

  rot_forward = rave.quatFromAxisAngle([0, 1, 0], np.pi/2)
  init_joints = manip.FindIKSolution(rave.matrixFromPose(np.r_[rot_forward, start_pt]), 0)
  final_joints = manip.FindIKSolution(rave.matrixFromPose(np.r_[rot_forward, final_pt]), 0)
  print init_joints, final_joints


  #robot.SetDOFValues(orig_joints)
  robot.SetDOFValues(init_joints, manip.GetArmIndices())

  # make optimization problem
  trajoptpy.SetInteractive(True)
  req = create_request(traj_len, manip.GetName(), final_pt, np.array([1, 0, 0, 0]), final_joints)
  prob = trajoptpy.ConstructProblem(json.dumps(req), env)
  result = trajoptpy.OptimizeProblem(prob)
  print result


if __name__ == '__main__':
  main()
