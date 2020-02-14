Aggregate Simulation
====================

Simulate spherical aggregates with a defined population mix.

Aggregates are simulated as a set of hard spheres of defined radius, uniformly randomly packed into a larger sphere. Those spheres are then randomly allocated to either ``green`` or ``red`` populations according to one of several sampling schemes based on radial position. Finally, coordinates files are written out compatible with the ``double_green`` reader in ``analyze_3d_aggregate_mixing.py``.

The main simulation tool is ``./simulate_3d_aggregate_mixing.py``, which can be called with the following syntax

.. code-block:: bash

    $ ./simulate_3d_aggregate_mixing.py \
        --distribution uniform \
        --num-red 400 \
        --num-green 127 \
        --aggregate-radius 75.3 \
        --neighbor-radius 20 \
        --same-cell-radius 5 \
        --num-batches 16 \
        ../data/sim_uniform_pos

Where the options are:

* ``distribution``: Which distribution to use when allocating ``red`` vs ``green`` cells
* ``num_red``: Number of "red" (mKate) cells to generate
* ``num_green``: Number of "green" (GFP) cells to generate
* ``aggregate_radius``: um - Radius of the spherical aggregate
* ``neighbor_radius``: um - Cells this close or closer are "neighbors"
* ``same_cell_radius``: um - Cells this close or closer are "the same cell"
* ``num_batches``: Number of aggregates to simulate (to match the number of empirical samples)

See the command-line help with ``./simulate_3d_aggregate_mixing.py -h``

Although only the ``uniform`` distribution is used by the paper, other distributions are available. The distribution name always refers to the bias applied to the ``red`` population.

* ``uniform``: ``red`` and ``green`` cells are allocated at random, regardless of radius
* ``left_triangle``: ``red`` cells prefer the inside of the aggregate, ``green`` the outside
* ``right_triangle``: ``red`` cells prefer the outside of the aggregate, ``green`` the inside
* ``inside``: ``red`` cells are forced to the inside of the aggregate, ``green`` the outside
* ``outside``: ``red`` cells are forced to the outside of the aggregate, ``green`` the inside

Additional distributions can be added by modifying the :py:func:`~cm_microtissue_struct.simulation.split_red_green` function.

Sphere packing is accomplished using rejection sampling in the :py:func:`~cm_microtissue_struct.simulation.simulate_spheres_in_sphere` function. Rejection sampling often fails to pack spheres above a certain density, even when such a packing should be possible. Increasing cell number, increasing cell radius, or decreasing aggregate radius too much may result in failed simulations. The parameters given above were never observed to result in failed packings as they are quite sparse compared to the theoretical limit.

API Reference
-------------

.. automodule:: cm_microtissue_struct.simulation
   :members:
   :undoc-members:
