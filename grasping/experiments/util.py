import numpy as np
import openravepy as rave

def transform_point(hmat, pt):
  return hmat.dot(np.r_[pt, 1])[:3]

def transform_points(hmat, points):
  return rave.poseTransformPoints(rave.poseFromMatrix(hmat), points)

def transform_normals(hmat, normals):
  return normals.dot(np.linalg.inv(hmat[:3,:3]))
