import openravepy as rave
from trajoptpy import make_kinbodies as mk
import numpy as np

PR2_LARM_SIDE_POSTURE = [  1.832,  -0.332,   1.011,  -1.437,   1.1  ,  -2.106,  3.074]

def mirror_arm_joints(x):
    "mirror image of joints (r->l or l->r)"
    return np.r_[-x[0],x[1],-x[2],x[3],-x[4],x[5],-x[6]]

def create_bare():
  env = rave.Environment()
  env.StopSimulation()
  env.Load('../data/pr2_table.env.xml')
  pr2 = env.GetRobot('pr2')
  pr2.SetDOFValues(PR2_LARM_SIDE_POSTURE, pr2.GetManipulator('leftarm').GetArmIndices())
  pr2.SetDOFValues(mirror_arm_joints(PR2_LARM_SIDE_POSTURE), pr2.GetManipulator('rightarm').GetArmIndices())
  return env

def create():
  env = create_bare()

  table = env.GetKinBody('table')

  table_aabb = table.ComputeAABB()
  table_top_z = table_aabb.pos()[2] + table_aabb.extents()[2]
  table_mid = table_aabb.pos()[:2]

  box_center = table_mid - [.4, 0]
  box_lwh = [0.1, 0.4, 0.2]
  mk.create_box_from_bounds(env, [-box_lwh[0]/2., box_lwh[0]/2., -box_lwh[1]/2., box_lwh[1]/2., -box_lwh[2]/2., box_lwh[2]/2.], name='box_0')
  box = env.GetKinBody('box_0')
  box.SetTransform(rave.matrixFromPose([1, 0, 0, 0, box_center[0], box_center[1], table_top_z+box_lwh[2]/2.]))

  cyl_height = .2
  cyl_radius = .05
  cyl_center = table_mid - [.25, 0]
  mk.create_cylinder(env, [0, 0, 0], cyl_radius, cyl_height, name='cyl_0')
  cyl = env.GetKinBody('cyl_0')
  cyl.SetTransform(rave.matrixFromPose([1, 0, 0, 0, cyl_center[0], cyl_center[1], table_top_z+cyl_height/2.]))

  return env

def create_topple_env():
  env = create_bare()

  table = env.GetKinBody('table')

  table_aabb = table.ComputeAABB()
  table_top_z = table_aabb.pos()[2] + table_aabb.extents()[2]
  table_mid = table_aabb.pos()[:2]

  box_center = table_mid + [-.4, -.1]
  box_lwh = [0.1, 0.1, 0.5]
  mk.create_box_from_bounds(env, [-box_lwh[0]/2., box_lwh[0]/2., -box_lwh[1]/2., box_lwh[1]/2., -box_lwh[2]/2., box_lwh[2]/2.], name='box_0')
  box = env.GetKinBody('box_0')
  box.SetTransform(rave.matrixFromPose([1, 0, 0, 0, box_center[0], box_center[1], table_top_z+box_lwh[2]/2.]))

  cyl_height = .5
  cyl_radius = .05
  cyl_center = table_mid + [-.25, 0]
  mk.create_cylinder(env, [0, 0, 0], cyl_radius, cyl_height, name='cyl_0')
  cyl = env.GetKinBody('cyl_0')
  cyl.SetTransform(rave.matrixFromPose([1, 0, 0, 0, cyl_center[0], cyl_center[1], table_top_z+cyl_height/2.]))

  return env
