import openravepy as rave
import numpy as np
import trajoptpy
import trajoptpy.math_utils as mu
import bulletsimpy
import json

def bulletobj_update_rave(env, bullet_obj):
  kb = env.GetKinBody(bullet_obj.GetName())
  print [b.GetName() for b in env.GetBodies()]
  print bullet_obj.GetName(), kb, rave.RaveGetEnvironmentId(env)
  kb.SetTransform(bullet_obj.GetTransform())

def get_bulletobj_state(bullet_obj):
  posevec = rave.poseFromMatrix(bullet_obj.GetTransform())
  return {"name": bullet_obj.GetName(), "xyz": list(posevec[4:7]), "wxyz": list(posevec[0:4])}

def record_sim(bullet_env, bullet_rec_objs, n_timesteps, dt=0.01, max_substeps=100, internal_dt=0.01, prestep_fn=None, pause_per_iter=False, update_rave_env=False):
  assert all(not o.IsKinematic() for o in bullet_rec_objs) # not really necessary, but would be silly otherwise
  out = []
  rave_env = bullet_env.GetRaveEnv()

  for t in range(n_timesteps):
    if prestep_fn is not None:
      prestep_fn(t)

    bullet_env.Step(dt, max_substeps, internal_dt)

    if update_rave_env:
      for o in bullet_rec_objs:
        #rave_env.GetKinBody(o.GetName()).SetTransform(o.GetTransform())
        #bulletobj_update_rave(rave_env, o)
        o.UpdateRave()
      rave_env.UpdatePublishedBodies()
      if pause_per_iter:
        raw_input('press enter to continue')

    obj_states = [get_bulletobj_state(o) for o in bullet_rec_objs]
    out.append({"timestep": t, "obj_states": obj_states})

  return out

def record_sim_with_traj(prob, robot_name, traj, traj_total_time, bullet_env, bullet_rec_objs, dt=0.01, **kwargs):
  assert 'n_timesteps' not in kwargs

  rave_env = bullet_env.GetRaveEnv()
  robot, bullet_robot = rave_env.GetRobot(robot_name), bullet_env.GetObjectByName(robot_name)
  assert bullet_robot.IsKinematic() # if not, the prestep and record_sim could clash
  robot.SetActiveDOFs(prob.GetDOFIndices(), prob.GetAffineDOFs())
  ss = rave.RobotStateSaver(robot)

  orig_traj_len = len(traj)
  expanded_traj_len = np.ceil(traj_total_time / dt)
  if expanded_traj_len <= orig_traj_len:
    expanded_traj = traj
  else:
    expanded_traj = mu.interp2d(np.linspace(0, traj_total_time, expanded_traj_len), np.linspace(0, traj_total_time, orig_traj_len), traj)

  def prestep(t):
    robot.SetActiveDOFValues(expanded_traj[t,:])
    bullet_robot.UpdateBullet()

  expanded_out = record_sim(bullet_env, bullet_rec_objs, n_timesteps=len(expanded_traj), prestep_fn=prestep, dt=dt, **kwargs)

  inds_of_orig = np.floor((expanded_traj_len-1.)/(orig_traj_len-1.)*np.arange(0, orig_traj_len)).astype(int)
  out = []
  for i, expanded_i in enumerate(inds_of_orig):
    assert expanded_out[expanded_i]['timestep'] == expanded_i
    out.append({"timestep": i, "obj_states": expanded_out[expanded_i]["obj_states"]})
  assert len(out) == orig_traj_len
  return out


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
    env_copy = env.CloneSelf(rave.CloningOptions.Bodies)
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
    #bulletobj_update_rave(env_copy, bullet_dynamic_objs[0])
    #for o in bullet_dynamic_objs: 
    #  print '>>>', o.GetName()
    #  bulletobj_update_rave(env_copy, o) #o.UpdateRave()

    rec = record_sim_with_traj(prob, p.robot_name, result.GetTraj(), p.traj_time, bullet_env, bullet_dynamic_objs, update_rave_env=False, dt=p.dt, max_substeps=p.max_substeps, internal_dt=p.internal_dt)
    # (if we want update_rave_env=True, then we need to clone the rave env first)

    curr_iter += 1
    prev_scene_states = rec
    prev_traj = result.GetTraj()

  return prev_traj
