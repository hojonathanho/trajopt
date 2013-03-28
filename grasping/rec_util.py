'''
There are two formats for storing scene trajectory data:

rec (recording format):
[ {'timestep': t, 'obj_states': [ {'name': obj_name, 'xyz': xyz, ...}, ... ]}, ... ]

dict (aka name2dict):
{ 'obj_name': { 'xyz': array((T, 3)), 'wxyz': array((T, 4)), ... }, ... }

'''

import numpy as np
from collections import defaultdict

def rec2dict(rec, obj_names=None):
  # converts output of record_sim_*
  # into a dictionary {obj_name: {field: vals_over_time}}
  # e.g. {'box': {'xyz': np.array(...), 'wxyz': np.array(...)}, 'cup': ...}
  timesteps = len(rec)
  name2trajs = defaultdict(dict)

  for t, state in enumerate(rec):
    for s in state['obj_states']:
      name = s['name']
      if obj_names is not None and name not in obj_names:
        continue

      trajs = name2trajs[name]
      for key, val in s.iteritems():
        if key == 'name':
          continue
        if key not in trajs:
          trajs[key] = np.zeros((timesteps, len(val)))
        trajs[key][t,:] = val

  return name2trajs

def some_item_of(d):
  if len(d) == 0: raise IndexError
  return d[d.keys()[0]]

def dict2rec(d):
  timesteps = some_item_of(some_item_of(d)).shape[0]
  obj_names = d.keys()
  rec = []
  for t in range(timesteps):
    obj_states = []
    for obj_name in obj_names:
      state = { 'name': obj_name }
      for field, vals in d[obj_name].iteritems():
        assert field != 'name'
        state[field] = vals[t,:].tolist()
      obj_states.append(state)
    rec.append({
      'timestep': t,
      'obj_states': obj_states
    })
  return rec

if __name__ == '__main__':
  test_rec = [{'timestep': 0, 'obj_states': [{'xyz': [0.0, 0.0, -0.0009799998952075839], 'wxyz': [1.0, 0.0, 0.0, 0.0], 'name': 'table'}, {'xyz': [1.0, 0.0, 1.499019980430603], 'wxyz': [1.0, 0.0, 0.0, 0.0], 'name': 'box'}]}, {'timestep': 1, 'obj_states': [{'xyz': [0.0, 0.0, -0.0029399995692074299], 'wxyz': [1.0, 0.0, 0.0, 0.0], 'name': 'table'}, {'xyz': [1.0, 0.0, 1.4970599412918091], 'wxyz': [1.0, 0.0, 0.0, 0.0], 'name': 'box'}]}, {'timestep': 2, 'obj_states': [{'xyz': [0.0, 0.0, -0.0058799996040761471], 'wxyz': [1.0, 0.0, 0.0, 0.0], 'name': 'table'}, {'xyz': [1.0, 0.0, 1.4941198825836182], 'wxyz': [1.0, 0.0, 0.0, 0.0], 'name': 'box'}]}, {'timestep': 3, 'obj_states': [{'xyz': [0.0, 0.0, -0.0097999991849064827], 'wxyz': [1.0, 0.0, 0.0, 0.0], 'name': 'table'}, {'xyz': [1.0, 0.0, 1.4901999235153198], 'wxyz': [1.0, 0.0, 0.0, 0.0], 'name': 'box'}]}, {'timestep': 4, 'obj_states': [{'xyz': [0.0, 0.0, -0.014699999243021011], 'wxyz': [1.0, 0.0, 0.0, 0.0], 'name': 'table'}, {'xyz': [1.0, 0.0, 1.4852999448776245], 'wxyz': [1.0, 0.0, 0.0, 0.0], 'name': 'box'}]}]

  print test_rec
  print rec2dict(test_rec)
  print dict2rec(rec2dict(test_rec))
