import bulletsimpy
import openravepy as rave
import trajoptpy.make_kinbodies as mk
import numpy as np

from CGAL.CGAL_Kernel import Point_3, Triangle_3, Ray_3
from CGAL.CGAL_AABB_tree import AABB_tree_Triangle_3_soup

BT_STEP_PARAMS = (0.01, 100, 0.01) # dt, max substeps, internal dt
FAR_POINT = [-100, -100, -100]
SPHERE_RADIUS = .005 # TODO: set based on input mesh size?
PUSH_DIST = .01 # TODO: same as above

def normalize(v):
  return v / np.linalg.norm(v)

def draw_aabb(env, aabb, linewidth=1.0):
  pos, ext = aabb.pos(), aabb.extents()

  def genpts(axis, m=1):
    assert axis in range(3)
    other_axes = [i for i in range(3) if i != axis]
    x, y = other_axes[0], other_axes[1]
    out = np.empty((5, 3))
    out[:4,axis] = pos[axis] + m*ext[axis]
    out[:4,x] = [pos[x]-ext[x], pos[x]-ext[x], pos[x]+ext[x], pos[x]+ext[x]]
    out[:4,y] = [pos[y]-ext[y], pos[y]+ext[y], pos[y]+ext[y], pos[y]-ext[y]]
    out[4,:] = out[0,:]
    return out

  handles = []
  for axis in range(3):
    handles.append(env.drawlinestrip(points=genpts(axis,  1), linewidth=linewidth))
    handles.append(env.drawlinestrip(points=genpts(axis, -1), linewidth=linewidth))
  return handles


def create_grid_on_aabb(aabb, num, edge_padding=.005):
  aabb_pos, aabb_extents = aabb.pos(), aabb.extents()

  points = np.empty((num*num*6, 3))
  for fixed_axis in range(3):
    vals = []
    varying_axes = [i for i in range(3) if i != fixed_axis]
    for varying_axis in varying_axes:
      lo = aabb_pos[varying_axis] - aabb_extents[varying_axis] + edge_padding
      hi = aabb_pos[varying_axis] + aabb_extents[varying_axis] - edge_padding
      vals.append(np.linspace(lo, hi, num))
    assert len(vals) == 2
    xys = np.concatenate(np.asarray(np.meshgrid(vals[0], vals[1])).T)

    off = fixed_axis * 2*num*num
    points[off:off+num*num,fixed_axis] = aabb_pos[fixed_axis] - aabb_extents[fixed_axis]
    points[off:off+num*num,varying_axes] = xys

    points[off+num*num:off+2*num*num,fixed_axis] = aabb_pos[fixed_axis] + aabb_extents[fixed_axis]
    points[off+num*num:off+2*num*num,varying_axes] = xys

  return points

def sample_mesh_points(geom, mesh):
  # easy way: sample points on AABB, then project onto surface
  # done in the mesh's local coordinate system
  triangles = []
  for inds in mesh.indices:
    pts = [Point_3(*v) for v in mesh.vertices[inds]]
    assert len(pts) == 3
    triangles.append(Triangle_3(*pts))
  tree = AABB_tree_Triangle_3_soup(triangles)

  aabb = geom.ComputeAABB(np.eye(4))
  aabb_pts = create_grid_on_aabb(aabb, 10)
  center = aabb.pos()
  rays = [Ray_3(Point_3(*center), Point_3(*p)) for p in aabb_pts]

  surface_points, normals = [], []
  for ray in rays:
    intersections = []
    tree.all_intersections(ray, intersections)
    if len(intersections) == 0:
      continue
    farthest_dist, farthest_intersection = -1, None
    for i in intersections:
      o = i[0]
      if not o.is_Point_3():
        continue
      pt = o.get_Point_3()
      npt = np.array([pt.x(), pt.y(), pt.z()])
      dist = np.linalg.norm(npt - center)
      if dist > farthest_dist:
        farthest_dist = dist
        farthest_intersection = i
    if farthest_intersection is not None:
      pt = farthest_intersection[0].get_Point_3()
      surface_points.append(np.array([pt.x(), pt.y(), pt.z()]))
      tri = mesh.vertices[mesh.indices[farthest_intersection[1]]]
      normals.append(normalize(np.cross(tri[1] - tri[0], tri[2] - tri[0])))

  return np.asarray(surface_points), np.asarray(normals)


#def run_single_push(bt_env, bt_obj, ):
#  pass

def transform_points(hmat, points):
  return rave.poseTransformPoints(rave.poseFromMatrix(hmat), points)

def transform_normals(hmat, normals):
  return np.linalg.inv(hmat[:3,:3]).T.dot(normals.T).T

def run(env, obj_name):
  kinbody = env.GetKinBody(obj_name)
  assert len(kinbody.GetLinks()) == 1

  # extract mesh info to get pushing directions
  link0 = kinbody.GetLinks()[0]
  assert len(link0.GetGeometries()) == 1
  geom0 = link0.GetGeometries()[0]
  mesh = geom0.GetCollisionMesh()

  sample_points, sample_normals = sample_mesh_points(geom0, mesh)
  sample_trans = link0.GetTransform().dot(geom0.GetTransform())

  env.SetViewer('qtcoin')
  handles = []
  handles += draw_aabb(env, geom0.ComputeAABB(sample_trans))
  handles.append(env.plot3(points=transform_points(sample_trans, sample_points), pointsize=5.0))
  handles.append(env.plot3(points=create_grid_on_aabb(geom0.ComputeAABB(sample_trans), 10), pointsize=1.0))
  for pt, n in zip(transform_points(sample_trans, sample_points), transform_normals(sample_trans, sample_normals)):
    handles.append(env.drawarrow(p1=pt, p2=pt+.01*n, linewidth=.001))
  raw_input('hi')

  # the pushing sphere
  sphere = mk.create_spheres(env, [FAR_POINT], .01, '_sphere_')

  # setup physics sim
  bt_env = bulletsimpy.BulletEnvironment(env, [obj_name])
  bt_obj = bt_env.GetObjectByName(obj_name)
  bt_sphere = bt_env.GetObjectByName('_sphere_')
  # TODO: set sim params, scaling?

  # run to stabilize first
  for i in range(20):
    bt_env.Step(*BT_STEP_PARAMS)

  init_obj_trans = bt_obj.GetTransform()

def setup_testing_env():
  # object with a table under it
  pass


if __name__ == '__main__':
  print create_grid_on_aabb(rave.AABB([0, 0, 0], [1, 1, 1]), 2)

  env = rave.Environment()
  #env.Load('../../data/box.xml')
  env.Load('data/mug1.kinbody.xml')

  run(env, 'mug')
