import trajoptpy
import openravepy as rave
import bulletsimpy
import physics
from pprint import pprint

env = rave.Environment()
env.StopSimulation()
env.Load("robots/pr2-beta-static.zae")
env.Load("../data/table.xml")

env.SetViewer('qtcoin')

dyn_obj_names = ['table']

bullet_env = bulletsimpy.LoadFromRave(env, dyn_obj_names)
bullet_env.SetGravity([0, 0, -9.8])

dyn_objs = [bullet_env.GetObjectByName(name) for name in dyn_obj_names]

# objects to record
rec_obj_names = ['table']
rec_objs = [bullet_env.GetObjectByName(name) for name in rec_obj_names]

# simulate for a few steps first to stabilize
for i in range(20):
  bullet_env.Step(0.01, 100, 0.01)
for o in dyn_objs:
  env.GetKinBody(o.GetName()).SetTransform(o.GetTransform())
env.UpdatePublishedBodies()


rec = physics.record_sim(bullet_env, rec_objs, n_timesteps=50, update_rave_env=True, pause_per_iter=True)
pprint(rec)

