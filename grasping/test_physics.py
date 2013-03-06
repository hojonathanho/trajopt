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

#bullet_env = bulletsimpy.LoadFromRave(env, dyn_obj_names)
bullet_env = bulletsimpy.BulletEnvironment(env, dyn_obj_names)
#bullet_env.SetGravity([0, 0, -9.8])

dyn_objs = [bullet_env.GetObjectByName(name) for name in dyn_obj_names]

for o in dyn_objs:
  o.UpdateRave()
env.UpdatePublishedBodies()

# objects to record
rec_obj_names = ['table']
rec_objs = [bullet_env.GetObjectByName(name) for name in rec_obj_names]

rec = physics.record_sim(bullet_env, rec_objs, n_timesteps=50, update_rave_env=True, pause_per_iter=True)
pprint(rec)
