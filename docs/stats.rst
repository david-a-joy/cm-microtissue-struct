Aggregate Statistics
====================

Statistics for 3d aggregate mixing

Using the results of ``analyze_3d_aggregate_mixing`` and ``simulate_3d_aggregate_mixing``, generate curves that compare empirical and theoretical distributions of cells in 3D.

To generate the comparison to the paper:

.. code-block:: bash

    $ ./stats_3d_aggregate_mixing.py \
        ../data/empirical_pos \
        ../data/sim_uniform_pos

Where ``../data/empirical_pos`` is the path to the empirical data and ``../data/sim_uniform_pos`` is the path to the simulated data.

Final plots will be written out under ``../data/sim_uniform_pos``

API Reference
-------------

.. automodule:: cm_microtissue_struct.stats
   :members:
   :undoc-members:
