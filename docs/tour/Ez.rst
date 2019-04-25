Ez
==

Equations
---------
We want to solve

.. math::

   \Delta_\perp E_z = \frac{\partial j_x}{\partial x} - \frac{\partial j_y}{\partial y}

with Dirichlet (zero) boundary conditions.


Method
------
The algorithm can be succinctly written as ``iDST2D(dirichlet_matrix * DST2D(RHS))``,
where ``DST2D`` and ``iDST2D`` are
Type-1 Forward and Inverse Discrete Sine 2D Trasforms respectively,
``RHS`` is the right-hand side of the equiation above,
and ``dirichlet_matrix`` is a 'magical' matrix that does all the work.


.. autofunction:: lcode.dirichlet_matrix

  In addition to the magic values, it also hosts the DST normalization multiplier.

.. todo:: DOCS: expand with method description (Kargapolov, Shalimova)


.. autofunction:: lcode.calculate_Ez

  Note that the outer cells do not participate in the calculations,
  and the result is simply padded with zeroes at the end.


DST2D
-----

.. autofunction:: lcode.dst2d

   As ``cupy`` currently ships no readily available function for calculating
   the DST2D on the GPU, we roll out our own FFT-based implementation.

   We don't need to make a separate iDST2D function
   as (for Type-1) it matches DST2D up to the normalization multiplier,
   which is taken into account in :func:`dirichlet_matrix`.
