import numpy as np
import scipy.spatial as ss

from CGAL.CGAL_Kernel import Point_3, Triangle_3, Ray_3
from CGAL.CGAL_AABB_tree import AABB_tree_Triangle_3_soup

def normalize(v):
  return v / np.linalg.norm(v)

def fit_linear(X_Nn, Y_Nm):
  N, n = X_Nn.shape
  m = Y_Nm.shape[1]
  assert Y_Nm.shape[0] == N

  Z = np.zeros((N, m, n*m))
  for i in range(N):
    Z[i,:,:] = block_diag(*np.tile(X_Nn[i][:,None], m).T)
  Z.shape = (N*m, n*m)

  a, residuals, rank, s = np.linalg.lstsq(Z, Y_Nm.reshape((N*m, 1)))
  return a.reshape((m, n))

def create_grid_on_aabb(aabb, density=.01, edge_padding=.005, ignore_z=True):
  aabb_pos, aabb_extents = aabb.pos(), aabb.extents()

  #points = np.empty((num*num*2*(2 if IGNORE_Z else 3), 3))
  points = []
  for fixed_axis in range(2 if ignore_z else 3):
    vals = []
    varying_axes = [i for i in range(3) if i != fixed_axis]
    for varying_axis in varying_axes:
      lo = aabb_pos[varying_axis] - aabb_extents[varying_axis] + edge_padding
      hi = aabb_pos[varying_axis] + aabb_extents[varying_axis] - edge_padding
      num = np.ceil((hi - lo)/density)
      vals.append(np.linspace(lo, hi, num))
    assert len(vals) == 2
    xys = np.concatenate(np.asarray(np.meshgrid(vals[0], vals[1])).T)

    curr_pts = np.empty((len(xys)*2, 3))
    curr_pts[:len(xys),fixed_axis] = aabb_pos[fixed_axis] - aabb_extents[fixed_axis]
    curr_pts[:len(xys),varying_axes] = xys
    curr_pts[len(xys):,fixed_axis] = aabb_pos[fixed_axis] + aabb_extents[fixed_axis]
    curr_pts[len(xys):,varying_axes] = xys
    points.extend(curr_pts)

  return np.asarray(points)

def sample_mesh_points(geom, ignore_z=True):
  # easy way: sample points on AABB, then project onto surface
  # done in the mesh's local coordinate system
  mesh = geom.GetCollisionMesh()
  triangles = []
  for inds in mesh.indices:
    pts = [Point_3(*v) for v in mesh.vertices[inds]]
    assert len(pts) == 3
    triangles.append(Triangle_3(*pts))
  tree = AABB_tree_Triangle_3_soup(triangles)

  aabb = geom.ComputeAABB(np.eye(4))
  aabb_pts = create_grid_on_aabb(aabb, ignore_z=ignore_z)
  center = aabb.pos()
  rays = [Ray_3(Point_3(*center), Point_3(*p)) for p in aabb_pts]

  # HACK: get id of first primitive (seems to be a global somewhere)
  first_tri_center = mesh.vertices[mesh.indices[0]].mean(axis=0)
  first_id = tree.closest_point_and_primitive(Point_3(*first_tri_center))[1]

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
      tri = mesh.vertices[mesh.indices[farthest_intersection[1] - first_id]]
      normals.append(normalize(np.cross(tri[1] - tri[0], tri[2] - tri[0])))

  return np.asarray(surface_points), np.asarray(normals)


def refine_mesh(vertices, indices, depth=1):
  '''refine triangle mesh by adding vertices at centroids and making 3 triangles for each original triangle'''
  if depth <= 0: return vertices, indices
  centroids = vertices[indices].mean(axis=1)

  #tripts = vertices[indices]
  #triv1 = tripts[:,1,:] - tripts[:,0,:]
  #triv2 = tripts[:,2,:] - tripts[:,0,:]
  #weights = np.random.rand(len(indices), 2)
  #flip_inds = np.nonzero(weights.sum(axis=1) > 1)[0]
  #flipped_weights = weights.copy()
  #flipped_weights[flip_inds,0] = 1 - weights[flip_inds,1]
  #flipped_weights[flip_inds,1] = 1 - weights[flip_inds,0]
  #centroids = tripts[:,0,:] + flipped_weights[:,0,None]*triv1 + flipped_weights[:,1,None]*triv2

  out_vertices = np.r_[vertices, centroids]
  centroid_inds = np.arange(len(vertices), len(out_vertices))
  out_indices = np.r_[
    np.c_[centroid_inds, indices[:,0], indices[:,1]],
    np.c_[centroid_inds, indices[:,1], indices[:,2]],
    np.c_[centroid_inds, indices[:,2], indices[:,0]]
  ]
  return refine_mesh(out_vertices, out_indices, depth=depth-1)

class FuncOnMesh(object):
  def __init__(self, geom, default_val=0):
    self.sample_points = sample_mesh_points(geom, ignore_z=False)[0]
    self.kdtree = ss.KDTree(self.sample_points)
    self.point_vals = np.ones(len(self.sample_points))*default_val
    self.distmat = ss.distance.squareform(ss.distance.pdist(self.sample_points))
    self.mindist = (self.distmat + 10*np.eye(len(self.distmat))*self.distmat.max()).min()

  def set_from_samples(self, pts, vals):
    assert len(pts) == len(vals)
    pts_kdtree = ss.KDTree(pts)
    dists, inds = pts_kdtree.query(self.sample_points, distance_upper_bound=.01)
    ok_inds = np.isfinite(dists)
    self.point_vals[ok_inds] = vals[inds[ok_inds]]

  def _get_nearby_points(self, p, radius):
    # TODO: only look at plane where p lies
    return self.kdtree.query_ball_point(p, radius)

  def smooth_by_euclidean(self, radius=.05, iters=1):
    # for each vertex, sample vertices at most SAMPLE_DIST away
    radius = max(radius, self.mindist)
    for _ in range(iters):
      vals = np.empty_like(self.point_vals)
      # TODO: only look at plane where the current point lies
      for i in range(len(self.sample_points)):
        inds = np.nonzero(self.distmat[i,:] < radius)[0]
        if len(inds) == 0: continue
        vals[i] = (self.point_vals[inds]).sum() # / (1+distmat[i,inds])).sum()
        vals[i] /= len(inds) #(1+distmat[i,inds]).sum()
      self.point_vals = vals

  def value(self, pt):
    # TODO: interpolation
    return self.point_vals[self.kdtree.query(pt)[1]]

  def gradient(self, pt):
    near_inds = self._get_nearby_points(pt, self.mindist*5)
    if len(near_inds) < 3:
      raise RuntimeError('no way im gonna fit a plane')
    points, vals = self.sample_points[near_inds], self.point_vals[near_inds]
    return fit_linear(points - pt[None,:], vals - self.value(pt))

  def gradient_nd(self, pt):
    raise NotImplementedError

  def plot_vertices(self, env, vert_trans=lambda vs: vs):
    # blue is 0, red is 1
    colors = np.zeros((len(self.sample_points), 3))
    colors[:,0] = self.point_vals
    colors[:,2] = 1. - colors[:,0]
    return env.plot3(points=vert_trans(self.sample_points), pointsize=5.0, colors=colors)

#class FuncOnMesh(object):
#  def __init__(self, vertices, indices, refine_depth=0, default_val=0):
#    # there can't be any unused vertices!
#    self.orig_vertices, self.orig_indices = vertices, indices
#    self.vertices, self.indices = refine_mesh(vertices, indices, depth=refine_depth)
#
#    self.kdtree = ss.cKDTree(self.vertices)
#    self.vertex_vals = np.ones(len(self.vertices))*default_val
#
#  def set_from_samples(self, pts, vals):
#    assert len(pts) == len(vals)
#    pts_kdtree = ss.cKDTree(pts)
#    dists, inds = pts_kdtree.query(self.vertices, distance_upper_bound=.01)
#    self.vertex_vals[np.isfinite(dists)] = vals[inds[np.isfinite(dists)]]
#
#  def smooth_by_euclidean(self, radius=.01, iters=1):
#    # for each vertex, sample vertices at most SAMPLE_DIST away
#    dists = ss.distance.squareform(ss.distance.pdist(self.vertices))
#    for _ in range(iters):
#      new_vertex_vals = np.empty_like(self.vertex_vals)
#      for i in range(len(self.vertices)):
#        inds = np.nonzero(dists[i,:] < radius)[0]
#        if len(inds) == 0: continue
#        new_vertex_vals[i] = (self.vertex_vals[inds]).sum() # / (1+dists[i,inds])).sum()
#        new_vertex_vals[i] /= len(inds) #(1+dists[i,inds]).sum()
#      self.vertex_vals = new_vertex_vals
#
#  def value(self, pt):
#    return self.vertex_vals[self.kdtree.query(pt)[1]]
#
#  def jacobian(self, pt):
#    pass
#
#  def jacobian_nd(self, pt):
#    pass
#
#  def plot_vertices(self, env, vert_trans=lambda vs: vs):
#    # blue is 0, red is 1
#    colors = np.zeros((len(self.vertices), 3))
#    colors[:,0] = self.vertex_vals
#    colors[:,2] = 1. - colors[:,0]
#    return env.plot3(points=vert_trans(self.vertices), pointsize=5.0, colors=colors)

def compute_normals(vertices, indices):
  normals = np.empty((len(indices), 3))
  for i in range(len(normals)):
    tri = vertices[indices[i]]
    normals[i,:] = normalize(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
  return normals

def remove_redundant_vertices(vertices, indices):
  uniq_inds = np.unique(indices)
  out_vertices = vertices[uniq_inds,:]

  ind_orig_to_new = len(vertices)*np.ones(len(vertices), dtype=int)
  ind_orig_to_new[uniq_inds] = np.arange(0, len(uniq_inds))
  out_indices = ind_orig_to_new[indices]

  return out_vertices, out_indices

def delete_facing(vertices, indices, direction=np.array([0, 0, -1]), tol=.01):
  normals = compute_normals(vertices, indices)
  dots = normals.dot(direction)
  mask = np.logical_or(dots <= 0, dots <= np.cos(tol))
  print indices[mask,:]
  return remove_redundant_vertices(vertices, indices[mask,:])

def print_off_data(vertices, indices):
  print 'OFF'
  print len(vertices), len(indices), 3*len(indices)
  for v in vertices:
    print v[0], v[1], v[2]
  for i in indices:
    print '3', i[0], i[1], i[2]
