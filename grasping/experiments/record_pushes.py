import bulletsimpy
import openravepy as rave
import trajoptpy.make_kinbodies as mk
import numpy as np
import meshes

#import argparse
#parser = argparse.ArgumentParser()

BT_STEP_PARAMS = (0.01, 100, 0.01) # dt, max substeps, internal dt
FAR_POINT = [-100, -100, -100]
SPHERE_RADIUS = .005 # TODO: set based on input mesh size?
SPHERE_NAME = '_sphere_'
PUSH_END_DIST = .01 # TODO: same as above
PUSH_TIMESTEPS = 20
PUSH_START_DIST = .02
EXPLODE_THRESH = 2
PLOTTING = True
PLOT_EACH_PUSH = False

def transform_point(hmat, pt):
  return hmat.dot(np.r_[pt, 1])[:3]

def transform_points(hmat, points):
  return rave.poseTransformPoints(rave.poseFromMatrix(hmat), points)

def transform_normals(hmat, normals):
  return normals.dot(np.linalg.inv(hmat[:3,:3]))

def exploded(hmat, orig_hmat):
  return np.linalg.norm(hmat[:3,3] - orig_hmat[:3,3]) > EXPLODE_THRESH

def toppled(hmat, orig_hmat, thresh=.02):
  R, R_orig = hmat[:3,:3], orig_hmat[:3,:3]
  z = np.array([0, 0, 1])
  print z.dot(R.dot(R_orig.T).dot(z)) 
  return np.arccos(z.dot(R.dot(R_orig.T).dot(z))) > thresh

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


#def run_single_push(bt_env, bt_obj, ):
#  pass

def run(env, obj_name):
  kinbody = env.GetKinBody(obj_name)
  assert len(kinbody.GetLinks()) == 1

  # extract mesh info to get pushing directions
  link0 = kinbody.GetLinks()[0]
  assert len(link0.GetGeometries()) == 1
  geom0 = link0.GetGeometries()[0]

  sample_points, sample_normals = meshes.sample_mesh_points(geom0)
  sample_trans = link0.GetTransform().dot(geom0.GetTransform())


  # the pushing sphere
  sphere_xml = '''
  <Environment>
    <KinBody name="%s">
      <Body type="static">
        <Geom type="sphere">
          <Translation>0 0 0</Translation>
          <Radius>%f</Radius>
        </Geom>
      </Body>
    </KinBody>
  </Environment>
  ''' % (SPHERE_NAME, SPHERE_RADIUS)
  env.LoadData(sphere_xml)
  sphere = env.GetKinBody(SPHERE_NAME)
  sphere.SetTransform(np.eye(4))

  # setup physics sim
  bt_env = bulletsimpy.BulletEnvironment(env, [obj_name])
  bt_obj = bt_env.GetObjectByName(obj_name)
  bt_sphere = bt_env.GetObjectByName(SPHERE_NAME)
  # TODO: set sim params, scaling?

  # run to stabilize first
  for _ in range(300):
    bt_env.Step(*BT_STEP_PARAMS)
  bt_obj.UpdateRave()
  env.UpdatePublishedBodies()

  init_obj_trans = bt_obj.GetTransform()
  print 'init obj trans', init_obj_trans


  result_obj_trans = np.empty((len(sample_points), 4, 4))
  result_exploded = np.empty(len(sample_points), dtype=bool)
  result_toppled = np.empty(len(sample_points), dtype=bool)
  for curr_iter, (sample_pt, sample_n) in enumerate(zip(sample_points, sample_normals)):
    print curr_iter, len(sample_points)
    curr_sample_to_world = link0.GetTransform().dot(geom0.GetTransform())
    if PLOTTING:
      handles = []
      if PLOT_EACH_PUSH:
        handles += draw_aabb(env, geom0.ComputeAABB(curr_sample_to_world))
        handles.append(env.plot3(points=transform_points(curr_sample_to_world, sample_points), pointsize=2.0))
        handles.append(env.plot3(points=meshes.create_grid_on_aabb(geom0.ComputeAABB(curr_sample_to_world), 10), pointsize=1.0))
        for pt, n in zip(transform_points(curr_sample_to_world, sample_points), transform_normals(curr_sample_to_world, sample_normals)):
          handles.append(env.drawarrow(p1=pt, p2=pt+.01*n, linewidth=.001))

    pt = transform_point(curr_sample_to_world, sample_pt)
    normal = transform_normals(curr_sample_to_world, sample_n)

    # execute push
    line_start_pt, line_end_pt = pt + PUSH_START_DIST*normal, pt - PUSH_END_DIST*normal
    for i in range(PUSH_TIMESTEPS):
      frac = float(i)/float(PUSH_TIMESTEPS-1)
      curr_pt = (1.-frac)*line_start_pt + frac*line_end_pt
      bt_sphere.SetTransform(rave.matrixFromPose(np.r_[[1, 0, 0, 0], curr_pt]))
      bt_env.Step(*BT_STEP_PARAMS)
    if PLOTTING and PLOT_EACH_PUSH:
      bt_sphere.UpdateRave()
      bt_obj.UpdateRave()
      print curr_iter, curr_pt, bt_obj.GetTransform()[:3,3]
      env.UpdatePublishedBodies()
      raw_input('asdfx')

    result_obj_trans[curr_iter,:,:] = bt_obj.GetTransform()
    result_exploded[curr_iter] = exploded(result_obj_trans[curr_iter,:,:], init_obj_trans)
    result_toppled[curr_iter] = not result_exploded[curr_iter] and toppled(result_obj_trans[curr_iter,:,:], init_obj_trans)

    # reset state to prepare for next push
    bt_sphere.SetTransform(rave.matrixFromPose(np.r_[[1, 0, 0, 0], FAR_POINT]))
    bt_sphere.UpdateRave()
    bt_obj.SetLinearVelocity([0, 0, 0])
    bt_obj.SetAngularVelocity([0, 0, 0])
    bt_obj.SetTransform(init_obj_trans)
    bt_obj.UpdateRave()
    # stabilize
    for i in range(5): bt_env.Step(*BT_STEP_PARAMS)

  print 'exploded:', np.count_nonzero(result_exploded), 'out of', len(result_obj_trans)
  print 'toppled:', np.count_nonzero(result_toppled), 'out of', len(result_obj_trans)

  if PLOTTING:
    curr_sample_to_world = link0.GetTransform().dot(geom0.GetTransform())
    toppled_inds = np.nonzero(result_toppled)[0]
    if toppled_inds.any():
      pass
      #handles.append(env.plot3(points=transform_points(curr_sample_to_world, sample_points[toppled_inds]), pointsize=5.0))
    env.UpdatePublishedBodies()
    raw_input('done')

    mesh = geom0.GetCollisionMesh()
    meshes.print_off_data(*meshes.delete_facing(mesh.vertices, mesh.indices))
    #fn = meshes.FuncOnMesh(mesh.vertices, mesh.indices, refine_depth=6)
    fn = meshes.FuncOnMesh(geom0)
    fn.set_from_samples(sample_points[toppled_inds], np.repeat(1, len(toppled_inds)))

    for i in range(10):
      print i
      fn.smooth_by_euclidean(radius=.05, iters=5)
      h = fn.plot_vertices(env, vert_trans=lambda vs: transform_points(curr_sample_to_world, vs))
      raw_input('asdf')

    #handles.append(env.drawtrimesh(fn.vertices, fn.indices))
    raw_input('asdf')

  return sample_points, sample_normals, toppled, exploded

def setup_testing_env():
  env = rave.Environment()
  table_xml = '''
  <Environment>
    <KinBody name="table">
      <Body type="static" name="table_link">
        <Geom type="box">
          <Translation>0 0 -.025</Translation>
          <Extents>1 1 .05</Extents>
        </Geom>
      </Body>
    </KinBody>
  </Environment>
  '''
  env.LoadData(table_xml)
  #env.Load('data/mug1.kinbody.xml')

  table_mid = np.array([0, 0])
  table_top_z = 0
  box_center = [0, 0]
  box_lwh = [0.1, 0.2, 0.5]
  mk.create_box_from_bounds(env, [-box_lwh[0]/2., box_lwh[0]/2., -box_lwh[1]/2., box_lwh[1]/2., -box_lwh[2]/2., box_lwh[2]/2.], name='box')
  box = env.GetKinBody('box')
  box.SetTransform(rave.matrixFromPose([1, 0, 0, 0, box_center[0], box_center[1], table_top_z+box_lwh[2]/2.]))


  env.StopSimulation()
  #env.Load('../../data/box.xml')
  if PLOTTING:
    env.SetViewer('qtcoin')
    raw_input("set up done")
  return env


if __name__ == '__main__':
  print meshes.create_grid_on_aabb(rave.AABB([0, 0, 0], [1, 1, 1]), 2)

  env = setup_testing_env()
  print env.GetKinBody('box').GetTransform()

# clone = env.CloneSelf(rave.CloningOptions.Bodies)
# clone.StopSimulation()

  points, normals, toppled, exploded = run(env, 'box')

  if not PLOTTING:
    env.SetViewer('qtcoin')

