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
    #self.manip_name = 'rightarm'
    #self.init_joints = []

    self.interactive = True


def opt_and_sim_loop(p, create_opt_request_fn, base_env): # p is an OptParams
  static_obj_names = [b.GetName() for b in base_env.GetBodies() if b.GetName() not in p.dynamic_obj_names]
  assert p.robot_name in static_obj_names

  env = base_env.CloneSelf(rave.CloningOptions.Bodies)
  env.StopSimulation()

  # loop: inner-optimize, simulate, repeat
  curr_iter = 0
  prev_scene_states = None
  prev_traj = None
  while True:
    print '=== OUTER LOOP ITERATION %d ===' % curr_iter

    ### INNER-OPTIMIZE ###
    if p.interactive:
      trajoptpy.SetInteractive(True)
      viewer = trajoptpy.GetViewer(env)
      for name in p.dynamic_obj_names:
        viewer.SetKinBodyTransparency(env.GetKinBody(name), 0.01)
    request = create_opt_request_fn(curr_iter, prev_traj, prev_scene_states)
    prob = trajoptpy.ConstructProblem(json.dumps(request), env)
    result = trajoptpy.OptimizeProblem(prob)

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
    print "BLAHBLH6", rave.RaveGetEnvironmentId(bullet_env.GetRaveEnv())
    for o in bullet_dynamic_objs: o.UpdateRave()

    rec = physics.record_sim_with_traj(prob, p.robot_name, result.GetTraj(), p.traj_time, bullet_env, bullet_dynamic_objs, update_rave_env=False, dt=p.dt, max_substeps=p.max_substeps, internal_dt=p.internal_dt)
    # (if we want update_rave_env=True, then we need to clone the rave env first)

    # evaluate cost from simulation
    print "============= COST ==========", costs.scene_rec_cost(rec, p.dynamic_obj_names)

    curr_iter += 1
    prev_scene_states = rec
    prev_traj = result.GetTraj()

  return prev_traj


