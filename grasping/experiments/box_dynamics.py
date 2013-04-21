import numpy as np
import scipy
import scipy.spatial as ss
import openravepy as rave
import util
import trajoptpy.math_utils as mu

inv = np.linalg.inv

def normalized(v):
  return v / np.linalg.norm(v)

import pickle
with open('recording.txt') as f:
  recording = pickle.load(f)
kdtree = ss.KDTree(recording['local_push_points'])

def lookup_closest_rec(p):
  return kdtree.query(p)[1]

def project_p(p):
  return recording['local_push_points'][lookup_closest_rec(p)]

def get_p_bounds():
  # only works for a box
  pts = recording['local_push_points']
  return pts.min(axis=0), pts.max(axis=0)


# x: state [quat, trans]
# p: contact point on surface (object frame)
# u: scalar change in contact on surface frame (must be >= 0 for pushing inwards)
def dynamics_simple(x, p, u):
  assert len(x) == 7 and len(p) == 3 and u >= 0

  rot, pos = normalized(x[:4]), x[4:]
  ind = lookup_closest_rec(p)
  push_ratio = u / recording['push_dists'][ind]

  dpos = recording['obj_trans_diff'][ind,:3,3] * push_ratio
  new_pos = pos + dpos

  aa = rave.axisAngleFromRotationMatrix(recording['obj_trans_diff'][ind,:3,:3])
  drot = rave.matrixFromAxisAngle(aa * push_ratio)
  new_rot = rave.quatFromRotationMatrix(drot.dot(rave.matrixFromQuat(rot)))

  new_x = np.r_[new_rot, new_pos]

  old_trans = rave.matrixFromPose(x)
  new_trans = rave.matrixFromPose(new_x)
  new_p = util.transform_point(np.linalg.inv(new_trans).dot(old_trans), p + recording['local_push_dirs'][ind]*u)
  new_p = project_p(new_p)

  return new_x, new_p

def sim_u_sequence(env, box, us, init_x, init_p):
  steps = len(us)
  x = init_x.copy()
  p = init_p.copy()
  handles = []

  for t in range(steps):
    curr_trans = rave.matrixFromPose(x)
    box.SetTransform(curr_trans)
    handles.append(env.plot3(points=rave.poseTransformPoints(x, [p]), pointsize=5))
    env.UpdatePublishedBodies()
    raw_input('at time ' + str(t))
    x, p = dynamics_simple(x, p, us[t])


def dynamics_simple_world(x, c1, c2):
  # convert contact pt in world frame
  # to point in object's current local frame
  curr_trans = rave.matrixFromPose(x)
  p = util.transform_point(inv(curr_trans), c1)
  cvel = c2 - c1
  u = util.transform_normals(curr_trans, np.array([1, 0, 0])).dot(cvel) # component of velocity in the normal direction
  return dynamics_simple(x, p, u)

# c: contact pt trajectory in world frame
def sim_contact_pt_sequence(env, box, cs, init_x):
  steps = len(cs)
  x = init_x.copy()
  handles = []

  for t in range(steps-1):
    curr_trans = rave.matrixFromPose(x)
    box.SetTransform(curr_trans)
    print 'c', cs[t]
    handles.append(env.plot3(points=cs[t], pointsize=5))
    env.UpdatePublishedBodies()
    raw_input('at time ' + str(t))

    x, _ = dynamics_simple_world(x, cs[t], cs[t+1])


if __name__ == '__main__':
  import record_pushes
  env = record_pushes.setup_testing_env()
  box = env.GetKinBody('box')


  init_x = rave.poseFromMatrix(recording['obj_init_trans'])
  init_p = np.array([-0.05, -0.01055556+.08, 0])

  sim_u_sequence(env, box, [.01]*100, init_x, init_p)

  #init_c = rave.poseTransformPoints(init_x, [init_p])[0]
  #c_traj = mu.linspace2d(init_c, init_c+[.5,0,0], 100)
  #sim_contact_pt_sequence(env, box, c_traj, init_x)
