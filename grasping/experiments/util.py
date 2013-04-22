import numpy as np
import openravepy as rave

def transform_point(hmat, pt):
  return hmat.dot(np.r_[pt, 1])[:3]

def transform_points(hmat, points):
  return rave.poseTransformPoints(rave.poseFromMatrix(hmat), points)

def transform_normals(hmat, normals):
  return normals.dot(np.linalg.inv(hmat[:3,:3]))

def xyzaa_to_mat(xyz, aa=None):
  if aa is None:
    return rave.matrixFromPose(np.r_[rave.quatFromAxisAngle(xyz[3:]), xyz[:3]])
  return rave.matrixFromPose(np.r_[rave.quatFromAxisAngle(aa), xyz])

def mat_to_xyzaa(hmat):
  return np.r_[hmat[:3,3], rave.axisAngleFromRotationMatrix(hmat[:3,:3])]
