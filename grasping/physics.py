import numpy as np
import openravepy as rave
import bulletsimpy
import trajoptpy
import trajoptpy.math_utils as mu
from collections import defaultdict

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
        o.UpdateRave()
      rave_env.UpdatePublishedBodies()
      if pause_per_iter:
        raw_input('press enter to continue')

    obj_states = [get_bulletobj_state(o) for o in bullet_rec_objs]
    out.append({"timestep": t, "obj_states": obj_states})

  return out

def record_sim_with_traj(robot_name, dof_inds, affine_dofs, traj, traj_total_time, bullet_env, bullet_rec_objs, dt=0.01, extra_steps=0, **kwargs):
  assert 'n_timesteps' not in kwargs

  rave_env = bullet_env.GetRaveEnv()
  robot, bullet_robot = rave_env.GetRobot(robot_name), bullet_env.GetObjectByName(robot_name)
  assert bullet_robot.IsKinematic() # if not, the prestep and record_sim could clash
  ss = rave.RobotStateSaver(robot)
  robot.SetActiveDOFs(dof_inds, affine_dofs)

  orig_traj_len = len(traj)
  expanded_traj_len = np.ceil(traj_total_time / dt)
  if expanded_traj_len <= orig_traj_len:
    expanded_traj = traj
    expanded_extra_steps = extra_steps
  else:
    expanded_traj = mu.interp2d(np.linspace(0, traj_total_time, expanded_traj_len), np.linspace(0, traj_total_time, orig_traj_len), traj)
    expanded_extra_steps = int(extra_steps * float(expanded_traj_len)/float(orig_traj_len))

  def prestep(t):
    if t < len(expanded_traj):
      robot.SetActiveDOFValues(expanded_traj[t,:])
      bullet_robot.UpdateBullet()

  expanded_out = record_sim(bullet_env, bullet_rec_objs, n_timesteps=len(expanded_traj)+expanded_extra_steps, prestep_fn=prestep, dt=dt, **kwargs)

  inds_of_orig = np.floor((expanded_traj_len+expanded_extra_steps-1.)/(orig_traj_len+extra_steps-1.)*np.arange(0, orig_traj_len+extra_steps)).astype(int)
  out = []
  for i, expanded_i in enumerate(inds_of_orig):
    assert expanded_out[expanded_i]['timestep'] == expanded_i
    out.append({"timestep": i, "obj_states": expanded_out[expanded_i]["obj_states"]})
  assert len(out) == orig_traj_len + extra_steps
  return out


def rec2dict(rec, obj_names):
  # converts output of record_sim_*
  # into a dictionary {obj_name: {field: vals_over_time}}
  # e.g. {'box': {'xyz': np.array(...), 'wxyz': np.array(...)}, 'cup': ...}
  timesteps = len(rec)
  name2trajs = defaultdict(dict)

  for t, state in enumerate(rec):
    for s in state['obj_states']:
      name = s['name']
      if name not in obj_names:
        continue

      trajs = name2trajs[name]
      for key, val in s.iteritems():
        if key == 'name':
          continue
        if key not in trajs:
          trajs[key] = np.zeros((timesteps, len(val)))
        trajs[key][t,:] = val

  return name2trajs
