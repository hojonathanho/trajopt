import numpy as np
import openravepy as rave
import transformations as trans
import trajoptpy.math_utils as mu

def make_sine_perturbation(manip, traj_0, direction, amplitude=0.05, half_cycles=1):
  assert len(direction) == 3
  traj_len = len(traj_0)

  x = np.linspace(0, half_cycles*np.pi, traj_len)
  Y = (amplitude*np.sin(x)[:,None]) * direction

  Y_joints = np.empty_like(traj_0)
  robot = manip.GetRobot()
  with robot:
    robot.SetActiveDOFs(manip.GetArmIndices())
    for t in range(traj_len):
      robot.SetActiveDOFValues(traj_0[t,:])
      Y_joints[t,:] = np.linalg.pinv(manip.CalculateJacobian()).dot(Y[t,:])

  return Y_joints


def make_perturbation_basis(manip, traj_0, ptype='cartesian_sine'):
  types = ['cartesian_sine']
  assert ptype in types

  basis = []
  if ptype == 'cartesian_sine':
    for c in [1, 2]:
      for e in np.eye(3):
        basis.append(make_sine_perturbation(manip, traj_0, direction=e, amplitude=.05, half_cycles=c))
  return np.asarray(basis)


if __name__ == '__main__':
  ### test add_cartesian_sine ###

  import simple_env
  env = simple_env.create()
  env.SetViewer('qtcoin')
  robot = env.GetRobot('pr2')
  manip = robot.GetManipulator('rightarm')
  traj_len = 50
  handles = []

  start_pt = [ 0.10118172, -0.68224057,  1.22652485]
  final_pt = [ 0.6087673,  -0.10394247,  1.00060628]
  line = mu.linspace2d(start_pt, final_pt, traj_len)
  handles.append(env.drawlinelist(np.array([start_pt,final_pt]),1))
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

  def go(**kwargs):
#    sine_traj = add_cartesian_sine(manip, line_traj, **kwargs)
    perts = make_perturbation_basis(manip, line_traj)
    print perts, perts.shape
    for pert in perts:
      traj = line_traj + pert
      for t in range(traj_len):
        pos0 = manip.GetTransform()[0:3,3]
        robot.SetDOFValues(traj[t,:], manip.GetArmIndices())
        pos1 = manip.GetTransform()[0:3,3]
        handles.append(env.drawlinelist(np.array([pos0, pos1]), 1))
        env.UpdatePublishedBodies()
# go(direction=np.array([1, 0, 0]), amplitude=.05, half_cycles=5)
# go(direction=np.array([0, 1, 0]), amplitude=.05, half_cycles=5)
# go(direction=np.array([0, 0, 1]), amplitude=.05, half_cycles=5)
  go()

  raw_input('enter to continue')
