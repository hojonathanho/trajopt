import numpy as np
import bulletsimpy
import util
from trajoptpy import math_utils as mu
import openravepy as rave

BT_STEP_PARAMS = (0.01, 100, 0.01) # dt, max substeps, internal dt

def joint_traj_to_control(joint_traj):
  return joint_traj[1:] - joint_traj[:-1]

class RobotAndObject:
  class idx:
    x = np.arange(0, 6)
    joints = np.arange(6, 13)

    xyz = np.arange(0, 3)
    aa = np.arange(3, 6)

    dim = 13

  def __init__(self, manip, obj_name):
    self.manip = manip
    self.robot = self.manip.GetRobot()
    self.env = self.robot.GetEnv()
    self.obj_name = obj_name
    self.bt_env = bulletsimpy.BulletEnvironment(self.env, [self.obj_name])
    self.bt_obj = self.bt_env.GetObjectByName(self.obj_name)
    self.bt_robot = self.bt_env.GetObjectByName(self.robot.GetName())
    self.stabilize(iters=100)

  def stabilize(self, iters=5):
    for _ in range(iters):
      self.bt_env.Step(*BT_STEP_PARAMS)
    self.bt_obj.UpdateRave()

  def set_joints(self, joints):
    self.robot.SetDOFValues(joints, self.manip.GetArmIndices())
    self.bt_robot.UpdateBullet()

  def set_state(self, x):
    assert len(x) == self.idx.dim
    self.set_joints(x[self.idx.joints])
    self.bt_obj.SetTransform(util.xyzaa_to_mat(x[self.idx.x]))
    self.bt_obj.SetLinearVelocity([0, 0, 0])
    self.bt_obj.SetAngularVelocity([0, 0, 0])

  def get_state(self):
    x = np.zeros(self.idx.dim)
    x[self.idx.x] = util.mat_to_xyzaa(self.bt_obj.GetTransform())
    x[self.idx.joints] = self.robot.GetDOFValues(self.manip.GetArmIndices())
    return x

  def apply_control(self, u, steps=20):
    assert len(u) == len(self.idx.joints)
    curr_joints = self.robot.GetDOFValues(self.manip.GetArmIndices())
    new_joints = curr_joints + u
    traj = mu.linspace2d(curr_joints, new_joints, steps)
    for t in range(steps):
      self.set_joints(traj[t,:])
      self.bt_env.Step(*BT_STEP_PARAMS)

  def run_traj(self, x_init, us):
    xrec = np.empty((len(us)+1, self.idx.dim))
    xrec[0,:] = x_init
    self.set_state(x_init)
    for t in range(len(us)):
      print 'applying control', us[t,:]
      self.apply_control(us[t,:])
      xrec[t+1,:] = self.get_state()
      print 'new state', xrec[t+1,:]
      self.update_and_publish()
    return xrec

  def plot_traj(self, xs, step_by_step=False):
    T = len(xs)
    for t in range(T):
      self.set_state(xs[t,:])
      self.update_and_publish()
      if step_by_step:
        raw_input('iteration %d/%d' % (t, T))

  def update_and_publish(self):
    self.bt_obj.UpdateRave()
    self.env.UpdatePublishedBodies()

  def jac_control(self, x0, u0, eps=.01):
    n = len(u0)
    du = eps * np.eye(n)
    jac = np.empty((self.idx.dim, n))
    for i in range(n):
      self.set_state(x0)
      self.apply_control(u0 + du[i])
      y2 = self.get_state()
      self.set_state(x0)
      self.apply_control(u0 - du[i])
      y1 = self.get_state()
      jac[:,i] = (y2 - y1) / (2.*eps)
    return jac

  def jac_state(self, xprev, uprev, x0, u0, x1=None, n_samples=2*13, eps=.01):
    # x0 should come from f(xprev, uprev)
    # to linearize around state, apply random controls uprev+w to xprev
    # and fit a linear map that takes f(xprev, uprev+w) - x0 to f(f(xprev, uprev+w), u0) - f0
    x_samples = np.empty((n_samples, len(x0)))
    xnext_samples = np.empty((n_samples, len(x0)))
    for s in range(n_samples):
      uprev_rand = np.random.multivariate_normal(uprev, np.eye(len(uprev))*eps)
      self.set_state(xprev)
      self.apply_control(uprev_rand)
      x_samples[s,:] = self.get_state()
      self.apply_control(u0)
      xnext_samples[x,:] = self.get_state()
    if x1 is None:
      self.set_state(x0)
      self.apply_control(u0)
      x1 = self.get_state()
    return (xnext_samples.T - x1).dot(np.linalg.pinv(x_samples.T - x0))

  def linearize(self, x0, u0):
    # gives A, B, c so that f(x, u) ~= Ax + Bu + c
    self.set_state(x0)
    self.apply_control(u0)
    x1 = self.get_state()
    A = self.jac_state(x0, u0, x1=x1)
    B = self.jac_control(x0, u0)
    c = x1 - A.dot(x0) - B.dot(u0)
    return A, B, c


def optimize():
  pass

def main():
  ### setup ###
  from grasping import simple_env
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
  traj_len = 20
  start_pt = final_box_center - np.array([.1, 0, .05])
  final_pt = start_pt + np.array([.3, -.1, .2])
  line = mu.linspace2d(start_pt, final_pt, traj_len)
  handles.append(env.drawlinelist(np.array([start_pt,final_pt]),1))
  env.UpdatePublishedBodies()

  robot.SetActiveManipulator(manip.GetName())
  ikmodel = rave.databases.inversekinematics.InverseKinematicsModel(robot,iktype=rave.IkParameterization.Type.Transform6D)
  if not ikmodel.load(): ikmodel.autogenerate()
  rot_forward = rave.quatFromAxisAngle([0, 1, 0], np.pi/2)

  init_joints = manip.FindIKSolution(rave.matrixFromPose(np.r_[rot_forward, start_pt]), 0)
  robot.SetDOFValues(init_joints, manip.GetArmIndices())

  Tstart = manip.GetTransform()
  line_traj = np.empty((traj_len, len(manip.GetArmIndices())))
  orig_joints = robot.GetDOFValues()
  for t in range(traj_len):
    T = Tstart.copy()
    T[0:3,3] = line[t,:]
    sol = manip.FindIKSolution(T, 0)
    line_traj[t,:] = sol
    #print t, sol
    robot.SetDOFValues(sol, manip.GetArmIndices())
    #env.UpdatePublishedBodies()

  #final_joints = manip.FindIKSolution(rave.matrixFromPose(np.r_[rot_forward, final_pt]), 0)
  #joint_traj = mu.linspace2d(init_joints, final_joints, traj_len)

  robot.SetDOFValues(init_joints, manip.GetArmIndices())
  ro = RobotAndObject(manip, box.GetName())
  init_state = ro.get_state()
  xs = ro.run_traj(init_state, joint_traj_to_control(line_traj))
  ro.plot_traj(xs, step_by_step=True)


  # test setting to random spot
  #ro.set_state(xs[10])
  #for _ in range(30):
  #  ro.bt_env.Step(*BT_STEP_PARAMS)
  #  ro.bt_obj.UpdateRave()
  #  ro.env.UpdatePublishedBodies()
  #  raw_input('blah')


if __name__ == '__main__':
  main()
