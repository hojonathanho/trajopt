.. _tutorial:


Tutorial
===========

This tutorial will describe how to set up and solve optimization problems in python.

Move arm to joint target
-------------------------
Full source code:  :file:`python_examples/arm_to_joint_target.py`

The following script will plan to get the right arm to a joint-space target.
It is assumed that the robot's current state is the start state, so make sure you set the robots DOFs appropriately.

The optimization will pause at every iteration and plot the current trajectory. Press 'p' to un-pause. To disable plotting and pausing, change the statement to :code:`trajoptpy.SetInteractive(False)`.

.. literalinclude:: ../python_examples/arm_to_joint_target.py

Every problem description contains four sections: basic_info, costs, constraints, and init_info. (TODO: describe the syntax).

Here, there are two cost components: joint-space velocity, and the collision penalty. There is one constraint: the final joint state. (TODO: explain the parameters)

Let's inspect the terminal output after running the script. The following lines should appear at the end of the output.

::

  [INFO optimizers.cpp:225] iteration 9
  [INFO optimizers.cpp:294] 
                  |   oldexact |    dapprox |     dexact |      ratio
            COSTS | -------------------------------------------------
        joint_vel |  1.776e+00 |  2.249e-05 |  2.249e-05 |  1.000e+00
   cont_collision |  0.000e+00 |  0.000e+00 |  0.000e+00 | (     nan)
   cont_collision |  0.000e+00 |  0.000e+00 |  0.000e+00 | (     nan)
   cont_collision |  0.000e+00 |  0.000e+00 |  0.000e+00 | (     nan)
   cont_collision |  0.000e+00 | -3.796e-10 |  0.000e+00 | (-0.000e+00)
   cont_collision |  0.000e+00 | -3.801e-10 |  0.000e+00 | (-0.000e+00)
   cont_collision |  0.000e+00 | -3.947e-10 | -5.022e-04 | (1.272e+06)
   cont_collision |  0.000e+00 | -7.592e-10 |  0.000e+00 | (-0.000e+00)
   cont_collision |  0.000e+00 | -1.139e-09 |  0.000e+00 | (-0.000e+00)
   cont_collision |  0.000e+00 | -7.592e-10 |  0.000e+00 | (-0.000e+00)
            TOTAL |  1.776e+00 |  2.249e-05 | -4.797e-04 | -2.133e+01
  [INFO optimizers.cpp:305] converged because improvement was small (2.249e-05 < 1.000e-04)
  [INFO optimizers.cpp:343] woo-hoo! constraints are satisfied (to tolerance 1.00e-04)

Here, the optimization converged in 9 iterations. Each line in the table shows a cost or constraint term and it's value and improvement in the latest iteration.
The "joint_vel" line refers to the joint velocity cost.
The "cont_collision" lines each refer to the collision cost for a time interval [t,t+1].
As you can see, all of the collision costs are zero, meaning the trajectory is safely out of collision, with a safety margin of `dist_pen` meters.
Note that there's no term for the "joint" constraint. That's because this constraint is a linear constraint in the optimization problem, and the table only shows nonlinear constraints which must be approximated at each iteration.
As you can see, there's not a 1-1 correspondence between the "cost" and "constraint" items in the json file and the costs and constraints that the optimization algorithm sees.
(If you're digging into the C++ API, note that the table rows correspond to `Cost <../../dox_build/classsco_1_1_cost.html>`_ and `Constraint <../../dox_build/classsco_1_1_constraint.html>`_ objects).

.. note:: Why do we use a collision cost, rather than a constraint? In the sequential convex optimization procedure, constraints get converted into costs--specifically, :math:`\ell_1` penalties (see the paper). So the difference between a constraint and an :math:`\ell_1` cost is that for a constraint, the optimizer checks to see if it is satisfied (to some tolerance), and if not, it jacks up the penalty coefficient. The thing about collisions is that it's often necessary to violate the safety margin, e.g. when you need to contact an object. So you're best off handling the logic yourself of adjusting penalties and safety margins, based on your specific problem.



Move arm to pose target
-------------------------
Full source code:  :file:`python_examples/arm_to_cart_target.py`


Next, let's solve the same problem, but instead of specifying the target joint position, we'll specify the target pose.

Now we use a pose constraint instead of a joint constraint:

.. literalinclude:: ../python_examples/arm_to_cart_target.py
  :start-after: BEGIN pose_constraint
  :end-before: END pose_constraint

Initialize using collision-free IK:

.. literalinclude:: ../python_examples/arm_to_cart_target.py
  :start-after: BEGIN ik
  :end-before: END ik

...

.. literalinclude:: ../python_examples/arm_to_cart_target.py
  :start-after: BEGIN init
  :end-before: END init

Now you'll notice an extra couple lines of the optimizer output::

                |   oldexact |    dapprox |     dexact |      ratio
          COSTS | -------------------------------------------------
    ...
    CONSTRAINTS | -------------------------------------------------
           pose |  4.326e-05 |  4.326e-05 |  3.953e-05 |  9.138e-01

As you can see, the pose constraint is satisfied at convergence

Exploring other costs and constraints
-----------------------------------------

The best way to learn about the other costs and constraints that are available is to look at the Doxygen documentation for the subclasses of `CostInfo <http://rll.berkeley.edu/trajopt/doc/dox_build/structtrajopt_1_1_cost_info.html>`_ and `CntInfo <http://rll.berkeley.edu/trajopt/doc/dox_build/structtrajopt_1_1_cnt_info.html>`_ because each item under "costs" or "constraints" is first converted to one of these structs when the JSON file is read.

More examples coming soon

.. TODO

Optimizing the base position and torso height
-----------------------------------------------

This next example shows how to jointly optimize over the base and torso degrees of freedom of a mobile robot, as well as the arms. This program considers a trajectory with one timestep, and ``start_fixed=False``, so it is really just doing collision-aware numerical IK. You can optimize over a trajectory containing all of these degrees of freedom by setting ``n_steps``. ``basic_info`` is set as follows:

.. literalinclude:: ../python_examples/position_base.py
  :start-after: BEGIN basic_info
  :end-before: END basic_info

Note that ``manip="active"``, which makes the selects the degrees of freedom of the optimization problem based on the degrees of freedom of the robot, which we set earlier:

.. literalinclude:: ../python_examples/position_base.py
  :start-after: BEGIN set_active
  :end-before: END set_active

Numerical IK is very unreliable since the forward kinematics function is so nonlinear. Therefore, we do a series of random initializations:

.. literalinclude:: ../python_examples/position_base.py
  :start-after: BEGIN random_init
  :end-before: END random_init

Walking with humanoid robot
----------------------------

See :file:`python_examples/drc_walk.py` for a rather simple example of planning footsteps with the Atlas robot model (for DARPA robotics challenge).


Using point cloud data
-------------------------

This section will discuss how load various types of perception data into the collision world in a way that is favorable for speed, success rate, and safety.

(TODO)