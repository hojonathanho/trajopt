import numpy as np
import bulletsimpy
import util
from trajoptpy import math_utils as mu
import openravepy as rave

BT_STEP_PARAMS = (0.01, 100, 0.01) # dt, max substeps, internal dt
DEBUG = True

np.set_printoptions(suppress=True, linewidth=1000)

def joint_traj_to_control(joint_traj):
  return joint_traj[1:] - joint_traj[:-1]

def sample_unit_vectors(n, d):
  X = np.random.randn(n, d)
  return X / np.sqrt((X**2).sum(axis=1))[:,None]

class RobotAndObject:
  class idx:
    x = np.arange(0, 7)
    joints = np.arange(7, 14)

    dim = 14
    udim = 7

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
    corrected_pose = np.copy(x[self.idx.x])
    corrected_pose[:4] = mu.normalize(corrected_pose[:4])
    self.bt_obj.SetTransform(rave.matrixFromPose(corrected_pose))
    self.bt_obj.SetLinearVelocity([0, 0, 0])
    self.bt_obj.SetAngularVelocity([0, 0, 0])

  def get_state(self):
    x = np.zeros(self.idx.dim)
    x[self.idx.x] = rave.poseFromMatrix(self.bt_obj.GetTransform())
    x[self.idx.joints] = self.robot.GetDOFValues(self.manip.GetArmIndices())
    return x

# def state_diff(self, x2, x1):
#   dx = np.zeros(self.idx.dim)

#   dx[self.idx.xyz] = x2[self.idx.xyz] - x1[self.idx.xyz]

#   r2 = rave.matrixFromAxisAngle(x2[self.idx.aa])
#   r1 = rave.matrixFromAxisAngle(x1[self.idx.aa])
#   dx[self.idx.aa] = rave.axisAngleFromRotationMatrix(r2.dot(r1.T))

#   dx[self.idx.joints] = x2[self.idx.joints] - x1[self.idx.joints]

#   return dx

  def apply_control(self, u, steps=10):
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
      self.apply_control(us[t,:])
      xrec[t+1,:] = self.get_state()
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

  #TODO: only look at obj state part, use openrave for robot part

  def jac_state(self, xprev, uprev, x0, u0, x1=None, n_samples=100, eps=.01):
    # x0 should come from f(xprev, uprev)
    # to linearize around state, apply random controls uprev+w to xprev
    # and fit a linear map that takes f(xprev, uprev+w) - x0 to f(f(xprev, uprev+w), u0) - f0
    def sample_controls(u, n_samples, eps):
      return np.random.multivariate_normal(u, np.eye(len(u))*eps, size=n_samples)
      #return u + eps*sample_unit_vectors(n_samples, len(u))

    uprev_samples = sample_controls(uprev, n_samples, eps)
    #print 'control samples:\n', uprev_samples
    x_samples = np.empty((n_samples, len(x0)))
    xnext_samples = np.empty((n_samples, len(x0)))
    #print 'sampling xprev'
    for s in range(n_samples):
      self.set_state(xprev)
      self.apply_control(uprev_samples[s,:])
      #self.update_and_publish(); raw_input('prev')
      x_samples[s,:] = self.get_state()
      self.apply_control(u0)
      xnext_samples[s,:] = self.get_state()
      #self.update_and_publish(); raw_input('curr')
    if x1 is None:
      self.set_state(x0)
      self.apply_control(u0)
      x1 = self.get_state()
    return (xnext_samples.T - x1[:,None]).dot(np.linalg.pinv(x_samples.T - x0[:,None]))


  def jac_state_naive(self, x0, u0, eps=.01, stabilize_iters=10):
    dx = eps * np.eye(self.idx.dim)
    jac = np.empty((self.idx.dim, self.idx.dim))
    for i in range(self.idx.dim):
      self.set_state(x0 + dx[i]); self.stabilize(stabilize_iters)
      self.apply_control(u0)
      y2 = self.get_state()
      self.set_state(x0 - dx[i]); self.stabilize(stabilize_iters)
      self.apply_control(u0)
      y1 = self.get_state()
      jac[:,i] = (y2 - y1) / (2.*eps)
    return jac

  def linearize(self, xprev, uprev, x0, u0, x1):
    # gives A, B, c so that f(x, u) ~= Ax + Bu + c
    if x1 is None:
      self.set_state(x0)
      self.apply_control(u0)
      x1 = self.get_state()
    if xprev is None or uprev is None:
      # assuming a fixed starting state, we don't care about A at t=0
      A = np.zeros((self.idx.dim, self.idx.dim))
    else:
      #A = self.jac_state(xprev, uprev, x0, u0, x1=x1)
      A = self.jac_state_naive(x0, u0)
    B = self.jac_control(x0, u0)
    c = x1 - A.dot(x0) - B.dot(u0)
    return A, B, c

  def linearize_around_traj(self, xs, us):
    assert len(xs) == len(us) + 1
    T, xdim, udim = len(xs), xs.shape[1], us.shape[1]
    As = np.empty((T-1, xdim, xdim))
    Bs = np.empty((T-1, xdim, udim))
    cs = np.empty((T-1, xdim))
    for t in range(T-1):
      As[t,:,:], Bs[t,:,:], cs[t,:] = self.linearize(
        xs[t-1,:] if t > 0 else None,
        us[t-1,:] if t > 0 else None,
        xs[t,:],
        us[t,:],
        xs[t+1,:]
      )
      print 'state jacobian:\n', As[t,:,:]
      print 'control jacobian:\n', Bs[t,:,:]
      print 'offset:\n', cs[t,:]
      #raw_input(str(t) + '...')
    return As, Bs, cs

  def test_linearization(self, As, Bs, cs, xs, us):
    T = len(xs)
    assert len(As) == len(Bs) == len(cs) == len(us) == T - 1
    ok = True
    for t in range(T-1):
      xpred = As[t,:,:].dot(xs[t,:]) + Bs[t,:,:].dot(us[t,:]) + cs[t,:]
      if not np.allclose(xpred, xs[t+1,:]):
        print 'linearization inaccurate at t=%d' % (t,)
        print 'prediction:', xpred
        print 'actual:', xs[t+1,:]
        ok = False
    return ok

  def predict_from_linearization(self, As, Bs, cs, x_init, us):
    assert len(As) == len(Bs) == len(cs) == len(us)
    T = len(us) + 1
    xs = np.empty((T, self.idx.dim))
    xs[0,:] = x_init
    for t in range(T-1):
      xs[t+1,:] = As[t,:,:].dot(xs[t,:]) + Bs[t,:,:].dot(us[t,:]) + cs[t,:]
    return xs


def optimize(ro, x_init, us_init, iters=10):

  def cost(xs, us):
    return ((xs[1:] - xs[:-1])**2).sum().sum()
    #return (us**2).sum().sum()

  xdim, udim = len(x_init), us_init.shape[1]
  H = len(us_init) + 1

  import cvxpy

  ## variables ##
  x_var = cvxpy.variable(H, xdim, name='x')
  u_var = cvxpy.variable(H-1, udim, name='u')
  cost_var = cvxpy.variable(H, 1, name='cost')

  ## parameters ##
  x_init_par = cvxpy.parameter(xdim, name='x_init')
  x_nom_par = cvxpy.parameter(H, xdim, name='x_nom')
  u_nom_par = cvxpy.parameter(H-1, udim, name='u_nom')
  x_trust_par = cvxpy.parameter(name='x_trust')
  u_trust_par = cvxpy.parameter(name='u_trust')
  A_par, B_par, c_par = [], [], []
  for t in range(H-1):
    A_par.append(cvxpy.parameter(xdim, xdim, name=('A_%d' % t)))
    B_par.append(cvxpy.parameter(xdim, udim, name=('B_%d' % t)))
    c_par.append(cvxpy.parameter(xdim, name=('c_%d' % t)))
  params = [x_init_par, x_nom_par, u_nom_par, x_trust_par, u_trust_par] + A_par + B_par + c_par

  ## constraints ##
  constraints = []
  # initial state
  constraints.append(cvxpy.eq(x_var[0,:].T, x_init_par))
  for t in range(H-1):
    # dynamics
    constraints.append(cvxpy.eq(x_var[t+1,:].T, A_par[t]*x_var[t,:].T + B_par[t]*u_var[t,:].T + c_par[t]))
    # cost at time t
    #constraints.append(cvxpy.geq(cost_var[t,0], cvxpy.quad_form(u_var[t,:].T, cvxpy.eye(udim))))
    constraints.append(cvxpy.geq(cost_var[t,0], cvxpy.quad_form(x_var[t,:].T - x_var[t+1,:].T, cvxpy.eye(xdim))))

  constraints.append(cvxpy.geq(cost_var[H-1,0], 0))
  # trust regions
  for t in range(H):
    constraints.append(cvxpy.geq(x_trust_par, cvxpy.norm2(x_var[t,:].T - x_nom_par[t,:].T)))
  for t in range(H-1):
    constraints.append(cvxpy.geq(u_trust_par, cvxpy.norm2(u_var[t,:].T - u_nom_par[t,:].T)))

  objective = cvxpy.minimize(cvxpy.sum(cost_var))
  prog = cvxpy.program(objective, constraints, params)

  us = us_init.copy()
  for i in range(iters):
    # simulate and linearize
    xs = ro.run_traj(x_init, us)
    raw_input('done running traj')

    pre_cost = cost(xs, us)
    print 'iteration', i, 'current cost:', pre_cost

    As, Bs, cs = ro.linearize_around_traj(xs, us)

    curr_params = [cvxpy.matrix(x_init).T, cvxpy.matrix(xs), cvxpy.matrix(us), .1, .1] + map(cvxpy.matrix, list(As)) + map(cvxpy.matrix, list(Bs)) + [cvxpy.matrix(c).T for c in list(cs)]
    assert len(curr_params) == len(params)
    prog.show()
    result = prog(*curr_params)

    print 'initial cost:', pre_cost
    print 'solved qp cost:', result
    print 'computed qp cost:', cost(np.asarray(x_var.value), np.asarray(u_var.value))

    us = np.asarray(u_var.value)

    raw_input('hi')






def make_line_traj(manip, start_pt, end_pt, rot, traj_len):
  line = mu.linspace2d(start_pt, end_pt, traj_len)
  line_traj = np.empty((traj_len, len(manip.GetArmIndices())))
  robot = manip.GetRobot()
  with robot:
    robot.SetActiveManipulator(manip.GetName())
    ikmodel = rave.databases.inversekinematics.InverseKinematicsModel(robot,iktype=rave.IkParameterization.Type.Transform6D)
    if not ikmodel.load(): ikmodel.autogenerate()
    init_joints = manip.FindIKSolution(rave.matrixFromPose(np.r_[rot, start_pt]), 0)
    robot.SetDOFValues(init_joints, manip.GetArmIndices())
    for t in range(traj_len):
      line_traj[t,:] = manip.FindIKSolution(rave.matrixFromPose(np.r_[rot, line[t,:]]), 0)
      robot.SetDOFValues(line_traj[t,:], manip.GetArmIndices())
  return line_traj

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
  #env.SetViewer('qtcoin')
  handles = []

  ### make straight-line trajectory ###
  traj_len = 20
  start_pt = final_box_center - np.array([.1, 0, .05])
  final_pt = start_pt + np.array([.3, -.1, .2])
  rot_forward = rave.quatFromAxisAngle([0, 1, 0], np.pi/2)

  line_traj = make_line_traj(manip, start_pt, final_pt, rot_forward, traj_len)

  #line = mu.linspace2d(start_pt, final_pt, traj_len)
  #handles.append(env.drawlinelist(np.array([start_pt,final_pt]),1))
  #env.UpdatePublishedBodies()

  #robot.SetActiveManipulator(manip.GetName())
  #ikmodel = rave.databases.inversekinematics.InverseKinematicsModel(robot,iktype=rave.IkParameterization.Type.Transform6D)
  #if not ikmodel.load(): ikmodel.autogenerate()

  #init_joints = manip.FindIKSolution(rave.matrixFromPose(np.r_[rot_forward, start_pt]), 0)
  #robot.SetDOFValues(init_joints, manip.GetArmIndices())

  #Tstart = manip.GetTransform()
  #line_traj = np.empty((traj_len, len(manip.GetArmIndices())))
  #orig_joints = robot.GetDOFValues()
  #for t in range(traj_len):
  #  T = Tstart.copy()
  #  T[0:3,3] = line[t,:]
  #  sol = manip.FindIKSolution(T, 0)
  #  line_traj[t,:] = sol
  #  #print t, sol
  #  robot.SetDOFValues(sol, manip.GetArmIndices())
  #  #env.UpdatePublishedBodies()

  ##final_joints = manip.FindIKSolution(rave.matrixFromPose(np.r_[rot_forward, final_pt]), 0)
  ##joint_traj = mu.linspace2d(init_joints, final_joints, traj_len)

  robot.SetDOFValues(line_traj[0,:], manip.GetArmIndices())
  ro = RobotAndObject(manip, box.GetName())
  init_state = ro.get_state()
  us = joint_traj_to_control(line_traj)
  xs = ro.run_traj(init_state, us)
  ro.plot_traj(xs, step_by_step=False)


  # test setting to random spot
  #ro.set_state(xs[10])
  #for _ in range(30):
  #  ro.bt_env.Step(*BT_STEP_PARAMS)
  #  ro.bt_obj.UpdateRave()
  #  ro.env.UpdatePublishedBodies()
  #  raw_input('blah')


  print 'linearizing'
  import time
  t_start = time.time()
  As, Bs, cs = ro.linearize_around_traj(xs, us)
  #A = np.empty((len(us), xs.shape[1], xs.shape[1]))
  #B = np.empty((len(us), xs.shape[1], us.shape[1]))
  #c = np.empty((len(us), xs.shape[1]))
  #for t in range(len(us)-1):
  #  A[t,:,:], B[t,:,:], c[t,:] = ro.linearize(xs[t,:], us[t,:], xs[t+1,:], us[t+1,:])
  #  print 'state jacobian:\n', A[t,:,:]
  #  print 'control jacobian:\n', B[t,:,:]
  #  print 'offset:\n', c[t,:]
  #  raw_input(str(t) + '...')
  print 'took', time.time() - t_start
  #print 'bad:'
  #tol = 50
  #for t in range(len(us)-1):
  #  if (A[t,:,:] > tol).any() or (B[t,:,:]>tol).any() or (c[t,:] > tol).any():
  #    print t, '\n', A[t,:,:], '\n', B[t,:,:], '\n', c[t,:]
  print 'ok?', ro.test_linearization(As, Bs, cs, xs, us)

  from grasping import trajectories
  pert_traj = line_traj + trajectories.make_sine_perturbation(manip, line_traj, np.array([0,1,0]), amplitude=.1)
  pert_us = joint_traj_to_control(pert_traj)
  xpred = ro.predict_from_linearization(As, Bs, cs, init_state, pert_us)
  env.SetViewer('qtcoin')
  ro.plot_traj(xpred, step_by_step=False)

  optimize(ro, init_state, us, iters=10)

if __name__ == '__main__':
  import pdb, traceback, sys
  try:
    main()
  except:
    if DEBUG:
      type, value, tb = sys.exc_info()
      traceback.print_exc()
      pdb.post_mortem(tb)
    else:
      raise
