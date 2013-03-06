import numpy as np
import transformations as trans

def tipping_cost(hmat):
  z = np.array([0, 0, 1])
  angle_from_z = np.arccos(hmat[:3,:3].dot(z).dot(z))
  return angle_from_z / (np.pi/2.)

def scene_rec_cost(rec, dynamic_obj_names):
  cost = 0
  for t, state in enumerate(rec):
    obj_states = state['obj_states']
    for s in obj_states:
      if s['name'] in dynamic_obj_names:
        cost += tipping_cost(trans.quaternion_matrix(s['wxyz']))
  return cost
