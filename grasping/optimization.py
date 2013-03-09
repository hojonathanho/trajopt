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
  # run for a few steps first to stabilize
  for _ in range(10):
    bullet_env.Step(p.dt, p.max_substeps, p.internal_dt)
  for o in bullet_dynamic_objs: o.UpdateRave()
  return env, bullet_env

def opt_and_sim_loop(p, base_env, create_opt_request_fn): # p is an OptParams
  static_obj_names = [b.GetName() for b in base_env.GetBodies() if b.GetName() not in p.dynamic_obj_names]
  assert p.robot_name in static_obj_names

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
    env_copy = base_env.CloneSelf(rave.CloningOptions.Bodies)
    env_copy.StopSimulation()

    bullet_env = bulletsimpy.BulletEnvironment(env_copy, p.dynamic_obj_names)
    bullet_env.SetGravity([0, 0, -9.8])
    bullet_dynamic_objs = [bullet_env.GetObjectByName(n) for n in p.dynamic_obj_names]
    bullet_static_objs = [bullet_env.GetObjectByName(n) for n in static_obj_names]

    # run for a few steps first to stabilize
    for _ in range(10):
      bullet_env.Step(p.dt, p.max_substeps, p.internal_dt)
    for o in bullet_dynamic_objs: o.UpdateRave()

    rec = physics.record_sim_with_traj(p.robot_name, p.dof_inds, p.affine_dofs, result.GetTraj(), p.traj_time, bullet_env, bullet_dynamic_objs, update_rave_env=False, dt=p.dt, max_substeps=p.max_substeps, internal_dt=p.internal_dt)
    # (if we want update_rave_env=True, then we need to clone the rave env first)
    env_copy.Destroy()

    # evaluate cost from simulation
    print "============= COST ==========", costs.scene_rec_cost(rec, p.dynamic_obj_names)

    curr_iter += 1
    prev_scene_states = rec
    prev_traj = result.GetTraj()

  env.Destroy()
  return prev_traj

import scipy.optimize as sio
def direct_optimize(p, base_env, traj_0, method='nelder-mead', options={'xtol': 1e-4, 'disp': True}):
  env, bullet_env = clone_with_bullet(p, base_env)
# env = base_env.CloneSelf(rave.CloningOptions.Bodies)
# env.StopSimulation()
# ### intialization. maybe not necessary ###
# bullet_env = bulletsimpy.BulletEnvironment(env, p.dynamic_obj_names)
# bullet_env.SetGravity([0, 0, -9.8])
# bullet_dynamic_objs = [bullet_env.GetObjectByName(n) for n in p.dynamic_obj_names]
# # run for a few steps first to stabilize
# for _ in range(10):
#   bullet_env.Step(p.dt, p.max_substeps, p.internal_dt)
# print "BLAHBLH6", rave.RaveGetEnvironmentId(bullet_env.GetRaveEnv())
# for o in bullet_dynamic_objs: o.UpdateRave()
# ### end initialization ###

  manip = env.GetRobot(p.robot_name).GetManipulator(p.manip_name)
  basis = trajectories.make_perturbation_basis(manip, traj_0)
  init_coeffs = np.zeros(len(basis)); init_coeffs[0] = 1 # coeff 0 corresponds to traj_0

  def coeffs_to_traj(coeffs):
    return basis.T.dot(coeffs).T

  def costfunc(coeffs):
    bullet_env = bulletsimpy.BulletEnvironment(env, p.dynamic_obj_names)
    bullet_env.SetGravity([0, 0, -9.8])
    bullet_dynamic_objs = [bullet_env.GetObjectByName(n) for n in p.dynamic_obj_names]
    traj = coeffs_to_traj(coeffs)
    assert p.dof_inds is not None and p.affine_dofs is not None
    rec = physics.record_sim_with_traj(p.robot_name, p.dof_inds, p.affine_dofs, traj, p.traj_time, bullet_env, bullet_dynamic_objs, update_rave_env=False, dt=p.dt, max_substeps=p.max_substeps, internal_dt=p.internal_dt)
    return costs.scene_rec_cost(rec, p.dynamic_obj_names)

  #out_coeffs = sio.minimize(costfunc, init_coeffs, method=method, options=options)
  out_coeffs = sio.fmin(costfunc, init_coeffs, **options)
  return coeffs_to_traj(out_coeffs)

def display_traj(p, base_env, traj):
  assert p.dof_inds is not None and p.affine_dofs is not None
  env, bullet_env = clone_with_bullet(p, base_env)
  bullet_objs = [bullet_env.GetObjectByName(name) for name in p.dynamic_obj_names]
  rec = physics.record_sim_with_traj(p.robot_name, p.dof_inds, p.affine_dofs, traj, p.traj_time, bullet_env, bullet_objs, dt=p.dt, max_substeps=p.max_substeps, internal_dt=p.internal_dt)

  viewer = trajoptpy.GetViewer(env)
  viewer.PlotTraj(p.manip_name, traj.tolist(), rec)
  
  raw_input('blah')

# robot = env.GetRobot(p.robot_name)
# robot.SetActiveDOFs(p.dof_inds, p.affine_dofs)
# 
# env.SetViewer('qtcoin')

# for t in range(len(traj)):
#   for obj in bullet_objs:
#     name2trans = dict((x["name"], rave.matrixFromPose(np.r_[x["wxyz"], x["xyz"]])) for x in rec[t]["obj_states"])
#     env.GetKinBody(obj.GetName()).SetTransform(name2trans[obj.GetName()])
#   robot.SetActiveDOFValues(traj[t,:])
#   env.UpdatePublishedBodies()
#   raw_input("%d/%d" % (t, len(traj)))

  env.Destroy()

  # hack: use the trajopt viewer by constructing a dummy problem
# print 'traj len', len(traj), 'traj', traj.shape
# import IPython
# IPython.embed()
# request = {
#   "basic_info" : {
#     "n_steps" : len(traj),
#     "manip" : p.manip_name,
#     "start_fixed" : True
#   },
#   "costs" : [
#   {
#     "type" : "joint_vel",
#     "params": {"coeffs" : [1]}
#   },
#   ],
#   "constraints" : [],
#   "scene_states": rec,
#   "init_info": {
#     "type": "given_traj",
#     "data": traj.tolist()
#   }
# }
# try:
#   prob = trajoptpy.ConstructProblem(json.dumps(request), env)
#   result = trajoptpy.OptimizeProblem(prob)
# finally:
#   return
