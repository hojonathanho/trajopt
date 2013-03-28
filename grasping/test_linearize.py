from scipy.linalg import block_diag
import numpy as np
from collections import defaultdict
import optimization
import bulletsimpy
import physics
import rec_util

def fit_linear(X_Nn, Y_Nm):
  N, n = X_Nn.shape
  m = Y_Nm.shape[1]
  assert Y_Nm.shape[0] == N

  Z = np.zeros((N, m, n*m))
  for i in range(N):
    Z[i,:,:] = block_diag(*np.tile(X_Nn[i][:,None], m).T)
  Z.shape = (N*m, n*m)

  a, residuals, rank, s = np.linalg.lstsq(Z, Y_Nm.reshape((N*m, 1)))
  return a.reshape((m, n))

def fit_linear2(X_Nn, Y_Nm):
  assert Y_Nm.shape[0] == X_Nn.shape[0]
  return Y_Nm.T.dot(np.linalg.pinv(X_Nn.T))

class SceneLinearization(object):
  def __init__(self, base_traj, base_scene_traj, jacs):
    self.base_traj = base_traj # robot traj linearized around (timesteps x dof)
    self.base_scene_traj = base_scene_traj # {obj_name: {field: {val_over_time}}}, e.g. output from rec_util.rec2dict
    self.jacs = jacs # {obj_name: {field: {jacobians over time} }}

  def predict_single_timestep(self, t, joints, obj_names=None):
    assert 0 <= t < len(self.base_traj)
    djoints = (joints - self.base_traj[t,:]).T
    obj_states = []
    for obj_name in self.jacs:
      if obj_names is not None and obj_name not in obj_names:
        continue
      state = {'name': obj_name}
      for field, jac in self.jacs[obj_name].iteritems():
        dfield = jac[t,:,:].dot(djoints)
        state[field] = dfield + self.base_scene_traj[obj_name][field][t,:].T

        # special case for rotations: need to re-normalize the quaternions
        if field == 'wxyz':
          state['wxyz'] /= np.linalg.norm(state['wxyz'])

      obj_states.append(state)
    return obj_states

  def predict_traj(self, traj, obj_names=None):
    assert len(traj) == len(self.base_traj)
    return rec_util.rec2dict([{
      'timestep': t,
      'obj_states': self.predict_single_timestep(t, traj[t,:], obj_names)
    } for t in range(len(traj))])

def linearize_objs(p, env, traj0, dynamic_obj_names):
  bullet_env, rec_full = optimization.clone_and_run(p, env, traj0)
  rec = rec_full[:-p.extra_steps] if p.extra_steps > 0 else rec_full
  name2trajs = rec_util.rec2dict(rec, dynamic_obj_names)
  timesteps = len(rec)

  djoints0 = traj0[1:,:] - traj0[:-1,:]

  jacs = defaultdict(dict)
  for obj_name in dynamic_obj_names:
    trajs = name2trajs[obj_name]

    xyzs, quats = trajs['xyz'], trajs['wxyz']
    dxyzs = xyzs[1:,:] - xyzs[:-1,:]
    dquats = quats[1:,:] - quats[:-1,:]
    print dxyzs.shape[0], dquats.shape[0], djoints0.shape[0], timesteps-1
    assert dxyzs.shape[0] == dquats.shape[0] == djoints0.shape[0] == timesteps-1
    xyz_jacs = np.zeros((timesteps, dxyzs.shape[1], djoints0.shape[1]))
    quat_jacs = np.zeros((timesteps, dquats.shape[1], djoints0.shape[1]))
    for t in range(timesteps-1):
      sample_djoints, sample_dxyz, sample_dquat = [], [], []
      # djoints0[t,:] induces the change dxyzs[t,:]
      sample_djoints.append(djoints0[t,:])
      sample_dxyz.append(dxyzs[t,:])
      sample_dquat.append(dquats[t,:])
      # TODO: other basis vectors

      sample_djoints = np.asarray(sample_djoints)
      sample_dxyz = np.asarray(sample_dxyz)
      sample_dquat = np.asarray(sample_dquat)
      xyz_jacs[t,:,:] = fit_linear(sample_djoints, sample_dxyz)
      quat_jacs[t,:,:] = fit_linear(sample_djoints, sample_dquat)

    xyz_jacs[timesteps-1,:,:] = xyz_jacs[timesteps-2,:,:]
    quat_jacs[timesteps-1,:,:] = quat_jacs[timesteps-2,:,:]

    jacs[obj_name]['xyz'] = xyz_jacs
    jacs[obj_name]['wxyz'] = quat_jacs

  return SceneLinearization(traj0.copy(), name2trajs, jacs)

def linearize_objs_numdiff(p, env, traj0, dynamic_obj_names):
  # this is pretty fucking bad (O(len(traj)^2))
  # seems to give bad results too

  timesteps = len(traj0)
  num_joints = traj0.shape[1]
  eps = 1e-2
  djoints = eps*np.eye(num_joints)

  jacs = defaultdict(dict)
  orig_traj_time = p.traj_time

  for obj_name in dynamic_obj_names:
    jacs[obj_name]['xyz'] = np.zeros((timesteps, 3, num_joints))
    jacs[obj_name]['wxyz'] = np.zeros((timesteps, 4, num_joints))

  for t in range(timesteps):
    print t, timesteps
    base_traj = traj0[:t+1,:]
    p.traj_time = orig_traj_time * float(t+1) / float(timesteps)
    bullet_env, rec_full = optimization.clone_and_run(p, env, base_traj)
    rec = rec_full[:-p.extra_steps] if p.extra_steps > 0 else rec_full
    base_name2trajs = rec_util.rec2dict(rec, dynamic_obj_names)

    for curr_joint in range(num_joints):
      curr_traj = base_traj.copy()
      curr_traj[t,:] = base_traj[t,:] + djoints[curr_joint,:]
      bullet_env, rec_full = optimization.clone_and_run(p, env, curr_traj)
      rec = rec_full[:-p.extra_steps] if p.extra_steps > 0 else rec_full
      curr_name2trajs = rec_util.rec2dict(rec, dynamic_obj_names)

      for obj_name in dynamic_obj_names:
        base, curr = base_name2trajs[obj_name], curr_name2trajs[obj_name]
        for n in ['xyz', 'wxyz']:
          jacs[obj_name][n][t,:,curr_joint] = (curr[n][t,:] - base[n][t,:]).T / eps

  p.traj_time = orig_traj_time
  return SceneLinearization(traj0.copy(), base_name2trajs, jacs)


def main():
  #X = np.array([[1, 2, 3], [4, 5, 6]])
  #Y = np.array([[0, 1], [0, 0]])
  X = np.random.rand(2, 7)
  Y = np.random.rand(2, 3)
  x = np.random.rand(7, 1)
  for A in [fit_linear(X, Y), fit_linear2(X, Y)]:
    print 'A', A
    print 'Y', Y
    print 'out', X.dot(A.T)
    print A.dot(x)
    print

  import openravepy as rave
  import simple_env
  from trajoptpy import kin_utils as ku
  from trajoptpy import math_utils as mu
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

  p = optimization.OptParams()
  p.traj_time = 5
  p.dt = p.internal_dt = 0.01
  p.dynamic_obj_names = dyn_obj_names
  p.max_iter = 2
  p.dof_inds = pr2.GetManipulator(MANIP_NAME).GetArmIndices()
  p.affine_dofs = 0

  init_joint_start = pr2.GetDOFValues(p.dof_inds)
  traj = mu.linspace2d(init_joint_start, init_joint_target, N_STEPS)
  optimization.display_traj(p, env, traj)

  out0 = linearize_objs(p, env, traj, dyn_obj_names)
  print out0.jacs
  #out1 = linearize_objs_numdiff(p, env, traj, dyn_obj_names)
  #print out1.jacs

  # test the linearization
  import trajectories
  traj2 = traj + trajectories.make_sine_perturbation(manip, traj, np.array([0, 0, 1]), amplitude=.1, half_cycles=2)
  optimization.display_traj_and_rec(MANIP_NAME, env, traj, rec_util.dict2rec(out0.base_scene_traj))
  for out in [out0]:
    predicted = out.predict_traj(traj2)
    print 'linearized traj'
    optimization.display_traj_and_rec(MANIP_NAME, env, traj2, rec_util.dict2rec(predicted))


if __name__ == '__main__':
  main()
