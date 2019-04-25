Background ions
===============

LCODE 3D currently simulates only the movement of plasma electrons.
The ions are modelled as a constant charge density distribution component
that is calculated from the inital electron placement during
the :doc:`initialization <../technicalities/initialization>`.

.. autofunction:: lcode.initial_deposition

   For this initial deposition invocation,
   the ion density argument is specified as ``0``.

The result is stored as ``const.ro_initial``
and passed to every consequtive :func:`lcode.deposit` invocation.
