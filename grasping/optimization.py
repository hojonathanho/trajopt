import numpy as np
import trajoptpy
import openravepy as rave
import bulletsimpy
import physics
import costs
import json
import trajectories

class OptParams:
  def __init__(self):
    self.dt = 0.01
    self.max_substeps = 100
    self.internal_dt = 0.01

    self.traj_time = 2
    #self.traj_steps = 20
    self.extra_steps = 10

    self.dynamic_obj_names = []
    self.robot_name = 'pr2'
    self.manip_name = 'rightarm'
    #self.init_joints = []

    self.interactive = True
    self.max_iter = 2

    self.dof_inds = None
    self.affine_dofs = None

def clone_with_bullet(p, base_env):
  env = base_env.CloneSelf(rave.CloningOptions.Bodies)
  env.StopSimulation()
  bullet_env = bulletsimpy.BulletEnvironment(env, p.dynamic_obj_names)
  bullet_env.SetGravity([0, 0, -9.8])
  bullet_dynamic_objs = [bullet_env.GetObjectByName(n) for n in p.dynamic_obj_names]
  return env, bullet_env

def clone_and_run(p, env, traj):
  bullet_env = bulletsimpy.BulletEnvironment(env, p.dynamic_obj_names)
  bullet_env.SetGravity([0, 0, -9.8])
  bullet_dynamic_objs = [bullet_env.GetObjectByName(n) for n in p.dynamic_obj_names]
  assert p.dof_inds is not None and p.affine_dofs is not None
  rec = physics.record_sim_with_traj(p.robot_name, p.dof_inds, p.affine_dofs, traj, p.traj_time, bullet_env, bullet_dynamic_objs, extra_steps=p.extra_steps, update_rave_env=False, dt=p.dt, max_substeps=p.max_substeps, internal_dt=p.internal_dt)
  return bullet_env, rec

def eval_scene_cost(p, env, traj):
  rec = clone_and_run(p, env, traj)[1]
  return costs.scene_rec_cost(rec, p.dynamic_obj_names)

def opt_and_sim_loop(p, base_env, create_opt_request_fn): # p is an OptParams
  static_obj_names = [b.GetName() for b in base_env.GetBodies() if b.GetName() not in p.dynamic_obj_names]
  assert p.robot_name in static_obj_names
  assert isinstance(p, OptParams)

  env = base_env.CloneSelf(rave.CloningOptions.Bodies)
  env.StopSimulation()

  # loop: inner-optimize, simulate, repeat
  curr_iter = 0
  prev_scene_states = None
  prev_traj = None
  while curr_iter < p.max_iter:
    print '=== OUTER LOOP ITERATION %d ===' % curr_iter

    ### INNER-OPTIMIZE ###
    if p.interactive:
      trajoptpy.SetInteractive(True)
      viewer = trajoptpy.GetViewer(env)
      for name in p.dynamic_obj_names:
        viewer.SetKinBodyTransparency(env.GetKinBody(name), 0.01)
    request = create_opt_request_fn(curr_iter, prev_traj, prev_scene_states)
    print request
    assert request["basic_info"]["manip"] == p.manip_name
    prob = trajoptpy.ConstructProblem(json.dumps(request), env)
    result = trajoptpy.OptimizeProblem(prob)
    p.dof_inds = prob.GetDOFIndices()
    p.affine_dofs = prob.GetAffineDOFs()

    ### SIMULATE ###
    cost = eval_scene_cost(p, base_env, result.GetTraj())
#   env_copy = base_env.CloneSelf(rave.CloningOptions.Bodies)
#   env_copy.StopSimulation()

#   bullet_env = bulletsimpy.BulletEnvironment(env_copy, p.dynamic_obj_names)
#   bullet_env.SetGravity([0, 0, -9.8])
#   bullet_dynamic_objs = [bullet_env.GetObjectByName(n) for n in p.dynamic_obj_names]
#   bullet_static_objs = [bullet_env.GetObjectByName(n) for n in static_obj_names]

#   # run for a few steps first to stabilize
#   for _ in range(10):
#     bullet_env.Step(p.dt, p.max_substeps, p.internal_dt)
#   for o in bullet_dynamic_objs: o.UpdateRave()

#   rec = physics.record_sim_with_traj(p.robot_name, p.dof_inds, p.affine_dofs, result.GetTraj(), p.traj_time, bullet_env, bullet_dynamic_objs, update_rave_env=False, dt=p.dt, max_substeps=p.max_substeps, internal_dt=p.internal_dt)
#   # (if we want update_rave_env=True, then we need to clone the rave env first)
#   env_copy.Destroy()

    # evaluate cost from simulation
    print "============= COST ==========", cost

    curr_iter += 1
    prev_scene_states = rec
    prev_traj = result.GetTraj()

  env.Destroy()
  return prev_traj

import scipy.optimize as sio
def direct_optimize(p, base_env, traj_0, method='nelder-mead', options={'disp': True}):
  env, bullet_env = clone_with_bullet(p, base_env)

  manip = env.GetRobot(p.robot_name).GetManipulator(p.manip_name)
  basis = trajectories.make_perturbation_basis(manip, traj_0)
  init_coeffs = np.zeros(len(basis))

  def coeffs_to_traj(coeffs):
    return traj_0 + basis.T.dot(coeffs).T

  def costfunc(coeffs):
    traj = coeffs_to_traj(coeffs)
    val = eval_scene_cost(p, env, traj) + .1*costs.manip_traj_cost(traj)
    print list(coeffs), val
    return val

  #out_coeffs = sio.minimize(costfunc, init_coeffs, method=method, options=options)
  if method == 'nelder-mead':
    out_coeffs = sio.fmin(costfunc, init_coeffs, **options)
  elif method == 'powell':
    out_coeffs = sio.fmin_powell(costfunc, init_coeffs, **options)
  elif method == 'cg':
    out_coeffs = sio.fmin_cg(costfunc, init_coeffs, **options)
  else:
    assert False
  print 'final coeffs', out_coeffs
  return coeffs_to_traj(out_coeffs)

def anneal(p, base_env, traj_0):
  HARD_CUTOFF = 2
  env, bullet_env = clone_with_bullet(p, base_env)

  manip = env.GetRobot(p.robot_name).GetManipulator(p.manip_name)
  basis = trajectories.make_perturbation_basis(manip, traj_0)
  init_coeffs = np.zeros(len(basis))

  lb = init_coeffs - 1
  ub = init_coeffs + 1

  def coeffs_to_traj(coeffs):
    return traj_0 + basis.T.dot(coeffs).T

  def costfunc(coeffs):
    # barrier for bounds
    if np.any(coeffs < -HARD_CUTOFF) or np.any(coeffs > HARD_CUTOFF): return 99999

    #eps = 1e-6
    #barrier = sum(-np.log(coeffs + HARD_CUTOFF + eps)) + sum(np.log(HARD_CUTOFF - coeffs + eps))
    #barrier = 0

    traj = coeffs_to_traj(coeffs)

    scene_cost = eval_scene_cost(p, env, traj) 
    traj_cost = 0.1*costs.manip_traj_cost(traj)
    val = scene_cost + traj_cost
    print 'anneal', list(coeffs), scene_cost, traj_cost, val
    return val

  result = sio.anneal(costfunc, init_coeffs, lower=lb, upper=ub, dwell=15, full_output=True)
  out_xmin, out_Jmin, out_T, out_feval, out_iters, out_accept, out_retval = result
  print 'Annealing result:'
  print '\txmin:', out_xmin
  print '\tJmin:', out_Jmin
  print '\tfeval:', out_feval
  print '\titers:', out_iters
  print '\tretval:', out_retval
  return coeffs_to_traj(out_xmin)


def display_traj(p, base_env, traj):
  assert p.dof_inds is not None and p.affine_dofs is not None
  env, bullet_env = clone_with_bullet(p, base_env)
  bullet_objs = [bullet_env.GetObjectByName(name) for name in p.dynamic_obj_names]
  rec = physics.record_sim_with_traj(p.robot_name, p.dof_inds, p.affine_dofs, traj, p.traj_time, bullet_env, bullet_objs, dt=p.dt, max_substeps=p.max_substeps, internal_dt=p.internal_dt)
  viewer = trajoptpy.GetViewer(env)
  viewer.PlotTraj(p.manip_name, traj.tolist(), rec)
  env.Destroy()
