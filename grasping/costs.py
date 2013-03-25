import numpy as np
import transformations as trans

def tipping_cost(hmat):
  z = np.array([0, 0, 1])
  angle_from_z = np.arccos(hmat[:3,:3].dot(z).dot(z))
  return angle_from_z / (np.pi/2.)

def scene_rec_cost(rec, dynamic_obj_names):
  obj2trajs = physics.rec2dict(rec, dynamic_obj_names)
  timesteps = len(rec)
  cost = 0

  for name in dynamic_obj_names:
    trajs = obj2trajs[name]

    for t in range(timesteps):
      cost += tipping_cost(trans.quaternion_matrix(trajs['wxyz'][t,:]))

    cost += 100*((trajs['xyz'][1:,:] - trajs['xyz'][:-1,:])**2).sum().sum()

  return cost

def manip_traj_cost(traj):
  cost = 0
  cost += ((traj[1:,:] - traj[:-1,:])**2).sum().sum() # squared joint velocities
  return cost
