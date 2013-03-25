import trajoptpy
import trajoptpy.kin_utils as ku
import trajoptpy.math_utils as mu
import openravepy as rave
import bulletsimpy
import physics
import optimization
import costs
import simple_env
import trajectories
from pprint import pprint
import numpy as np
import json

env = simple_env.create_topple_env()
#env.SetViewer('qtcoin')
pr2 = env.GetRobot('pr2')

dyn_obj_names = ['box_0']
static_obj_names = ['table'] # cyl_0 not included since we'll collide with it in the final pose

bullet_env = bulletsimpy.BulletEnvironment(env, dyn_obj_names)
bullet_env.SetGravity([0, 0, -9.8])
dyn_objs = [bullet_env.GetObjectByName(n) for n in dyn_obj_names]
bullet_box0 = bullet_env.GetObjectByName('box_0')
bullet_cyl0 = bullet_env.GetObjectByName('cyl_0')

# simulate for a few steps first to stabilize
for i in range(20):
  bullet_env.Step(0.01, 100, 0.01)
for o in dyn_objs:
  env.GetKinBody(o.GetName()).SetTransform(o.GetTransform())
env.UpdatePublishedBodies()


N_STEPS = 20
MANIP_NAME = "rightarm"
quat_target = list(physics.get_bulletobj_state(bullet_cyl0)['wxyz'])
xyz_target = list(physics.get_bulletobj_state(bullet_cyl0)['xyz'])
print 'target:', quat_target, xyz_target
hmat_target = rave.matrixFromPose( np.r_[quat_target, xyz_target] )
# BEGIN ik
manip = pr2.GetManipulator("rightarm")
init_joint_target = ku.ik_for_link(hmat_target, manip, "r_gripper_tool_frame")#, filter_options = rave.IkFilterOptions.CheckEnvCollisions)
# END ik

p = optimization.OptParams()
p.traj_time = 2
p.dt = p.internal_dt = 0.05
p.dynamic_obj_names = dyn_obj_names
p.max_iter = 2
p.dof_inds = pr2.GetManipulator(MANIP_NAME).GetArmIndices()
p.affine_dofs = 0

init_joint_start = pr2.GetDOFValues(p.dof_inds)
traj = mu.linspace2d(init_joint_start, init_joint_target, N_STEPS)
optimization.display_traj(p, env, traj)

print 'initial cost:'
#tmp_env, tmp_bullet = optimization.clone_with_bullet(p, env)
#rec = physics.record_sim_with_traj(p.robot_name, p.dof_inds, p.affine_dofs, traj, p.traj_time, tmp_bullet, [tmp_bullet.GetObjectByName(n) for n in p.dynamic_obj_names], update_rave_env=False, dt=p.dt, max_substeps=p.max_substeps, internal_dt=p.internal_dt)
#print costs.scene_rec_cost(rec, p.dynamic_obj_names)
print optimization.eval_scene_cost(p, env, traj)

print 'optimizing...'
traj2 = optimization.anneal(p, env, traj)
optimization.display_traj(p, env, traj2)
#basis = trajectories.make_perturbation_basis(manip, traj)
#coeffs = np.array([1.99879242,  0.74942991, -1.10029305,  0.92905564, -0.83804528,  1.41086009])
#traj2 = traj + basis.T.dot(coeffs).T
print 'final cost:', optimization.eval_scene_cost(p, env, traj2)



e, bullet_env = optimization.clone_with_bullet(p, env)
bullet_dynamic_objs = [bullet_env.GetObjectByName(n) for n in p.dynamic_obj_names]
e.SetViewer('qtcoin')
physics.record_sim_with_traj(p.robot_name, p.dof_inds, p.affine_dofs, traj2, p.traj_time, bullet_env, bullet_dynamic_objs, extra_steps=10, update_rave_env=True, dt=p.dt, max_substeps=p.max_substeps, internal_dt=p.internal_dt, pause_per_iter=True)
