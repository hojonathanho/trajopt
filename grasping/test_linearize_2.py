import openravepy as rave
import numpy as np
import trajoptpy.math_utils as mu
import bulletsimpy
import physics
import rec_util

def robot_position_jac(manip, link_ind, joints0, pt):
  robot = manip.GetRobot()
  with robot:
    robot.SetActiveDOFs(manip.GetArmIndices())
    robot.SetActiveDOFValues(joints0)
    jac = robot.CalculateActiveJacobian(link_ind, pt)
  return jac


def predict_obj_pos(manip, pos0, joints0, contact0, pos, joints):
  #proj = contact0.n[:,None].dot(contact0.n[None,:])

  alpha = 1 # FIXME

  jac = robot_position_jac(manip, contact0.linkA.GetIndex(), joints0, pos0)

  n = contact0.normalB2A.reshape((3, 1))

  d_from_joints = -alpha*n * max(0, -n.T.dot(jac).dot(joints - joints0)) # TODO: -max(0, contact0.d)

  d_from_pos = -alpha*n * max(0, -n.T.dot(pos - pos0))

  return pos0 + d_from_joints + d_from_pos

# rotation around z-axis
#def predict_obj_zrot(angle0, joints0, contact0, angle, joints):
  #xhat = np.array([1, 0, 0])[:,None]
  #angle = xhat.T.dot(rot).dot(xhat)


#def linearize_obj_pos(traj0, postraj0, contacts):
#  '''traj0: robot traj (T x dof)
#     postraj0: pos traj (T x 3)'''
#  timesteps, dof = traj0.shape
#  assert postraj0.shape == (timesteps, 3)
#  jacs = np.empty((timesteps, 3, dof))
#
#  for t in range(timesteps):
#    jacs[t,:,:] = ;


def linearize_multiple_obj_pos(traj0, postrajs0):
  for obj_name in postrajs0.keys():
    linearize_obj_pos(traj0, postrajs0[obj_name])

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
  env.SetViewer('qtcoin')
  handles = []

  ### make straight-line trajectory ###
  traj_len = 50
  start_pt = final_box_center - np.array([.1, 0, 0])
  final_pt = start_pt + np.array([.2, 0, 0])
  line = mu.linspace2d(start_pt, final_pt, traj_len)
  handles.append(env.drawlinelist(np.array([start_pt,final_pt]),1))
  env.UpdatePublishedBodies()

  robot.SetActiveManipulator(manip.GetName())
  ikmodel = rave.databases.inversekinematics.InverseKinematicsModel(robot,iktype=rave.IkParameterization.Type.Transform6D)
  if not ikmodel.load(): ikmodel.autogenerate()

  Tstart = manip.GetTransform()
  line_traj = np.empty((traj_len, len(manip.GetArmIndices())))
  orig_joints = robot.GetDOFValues()
  for t in range(traj_len):
    T = Tstart.copy()
    T[0:3,3] = line[t,:]
    sol = manip.FindIKSolution(T, 0)#, rave.IkFilterOptions.CheckEnvCollisions)
    line_traj[t,:] = sol
    print t, sol
    robot.SetDOFValues(sol, manip.GetArmIndices())
    env.UpdatePublishedBodies()
  robot.SetDOFValues(orig_joints)


  ### run in bullet and get all contacts ###
  dyn_obj_names = ['box_0']
  bt_env = bulletsimpy.BulletEnvironment(env, ['box_0'])
  dyn_objs = [bt_env.GetObjectByName(n) for n in dyn_obj_names]
  bt_box = dyn_objs[0]

  contacts = {}
  def prestep(t):
    # this prestep saves box/pr2 contacts
    collisions = bt_env.DetectCollisions()
    c = []
    for col in collisions:
      # convention: linkA is robot, linkB is box
      nameA = col.linkA.GetParent().GetName()
      nameB = col.linkB.GetParent().GetName()
      if nameA == 'box_0' and nameB == 'pr2':
        c.append(col.Flipped())
      elif nameB == 'box_0' and nameA == 'pr2':
        c.append(col)
    contacts[t] = c

  rec, inds_of_orig = physics.record_sim_with_traj('pr2', manip.GetArmIndices(), 0, line_traj, 5, bt_env, dyn_objs, prestep_fn=prestep, return_inds_of_orig=True)#, update_rave_env=True, pause_per_iter=True)
  obj_trajs = rec_util.rec2dict(rec, obj_names=['box_0'])

  print 'contacts'
  contacts = {i: contacts[t] for i, t in enumerate(inds_of_orig)}
  for t in range(traj_len):
    print t, [(c.linkA.GetParent().GetName(), c.linkB.GetParent().GetName()) for c in contacts[t]]


  print 'displaying linearized traj'
  obj_pos_traj = obj_trajs['box_0']['xyz']
  joint_traj = line_traj
  for t in range(traj_len):
    predict_obj_pos(manip, obj_pos_traj[t,:], joint_traj[t,:], contact0, pos, joints):



if __name__ == '__main__':
  main()
