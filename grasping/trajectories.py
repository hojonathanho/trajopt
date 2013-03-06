import numpy as np
import openravepy as rave
import transformations as trans
import trajoptpy.math_utils as mu

def add_cartesian_sine(manip, traj, angle_around_axis=0, amplitude=0.05, half_cycles=1):
  traj_len = len(traj)
  robot = manip.GetRobot()
  ss = rave.RobotStateSaver(robot)
  robot.SetActiveDOFs(manip.GetArmIndices())

  robot.SetActiveDOFValues(traj[0,:])
  start_pt = manip.GetTransform()[0:3,3]
  robot.SetActiveDOFValues(traj[-1,:])
  end_pt = manip.GetTransform()[0:3,3]

  axis = mu.normalize(end_pt - start_pt)
  # orig_up = just some vector orthogonal to axis
  orig_up = np.cross(axis, np.array([1, 0, 0]))
  if np.linalg.norm(orig_up) < 1e-6: orig_up = np.cross(axis, [0, 1, 0])
  up = trans.rotation_matrix(angle_around_axis, axis)[:3,:3].dot(orig_up)

  x = np.linspace(0, half_cycles*np.pi, traj_len)
  Y = amplitude*np.sin(x)[:,None] * up

  Y_joints = np.empty_like(traj)
  for t in range(traj_len):
    robot.SetActiveDOFValues(traj[t,:])
    Y_joints[t,:] = np.linalg.pinv(manip.CalculateJacobian()).dot(Y[t,:])
  return traj + Y_joints

if __name__ == '__main__':
  import simple_env
  env = simple_env.create()
  env.SetViewer('qtcoin')
  robot = env.GetRobot('pr2')
  manip = robot.GetManipulator('rightarm')
  traj_len = 50

  start_pt = [ 0.10118172, -0.68224057,  1.22652485]
  final_pt = [ 0.6087673,  -0.10394247,  1.00060628]
  line = mu.linspace2d(start_pt, final_pt, traj_len)
  h = env.drawlinelist(np.array([start_pt,final_pt]),1)
  env.UpdatePublishedBodies()

  robot.SetActiveManipulator(manip.GetName())
  ikmodel = rave.databases.inversekinematics.InverseKinematicsModel(robot,iktype=rave.IkParameterization.Type.Transform6D)
  if not ikmodel.load(): ikmodel.autogenerate()

  Tstart = manip.GetTransform()
  line_traj = np.empty((traj_len, len(manip.GetArmIndices())))
  for t in range(traj_len):
    T = Tstart.copy()
    T[0:3,3] = line[t,:]
    sol = manip.FindIKSolution(T, rave.IkFilterOptions.CheckEnvCollisions)
    line_traj[t,:] = sol
    robot.SetDOFValues(sol, manip.GetArmIndices())
    env.UpdatePublishedBodies()

  handles = []

  def go(**kwargs):
    sine_traj = add_cartesian_sine(manip, line_traj, **kwargs)
    for t in range(traj_len):
      pos0 = manip.GetTransform()[0:3,3]
      robot.SetDOFValues(sine_traj[t,:], manip.GetArmIndices())
      pos1 = manip.GetTransform()[0:3,3]
      handles.append(env.drawlinelist(np.array([pos0, pos1]), 1))
      env.UpdatePublishedBodies()
  for angle in np.linspace(0, 2.*np.pi, 8):
    go(angle_around_axis=angle, amplitude=0.05, half_cycles=5)

  raw_input('enter to continue')
