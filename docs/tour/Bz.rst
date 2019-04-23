Bz
==

Equations
---------
We want to solve

.. math::

   \Delta_\perp B_z = \frac{\partial j_x}{\partial y} - \frac{\partial j_y}{\partial x}

with Neumann boundary conditions (derivative = 0).


Method
------
The algorithm can be succinctly written as ``iDCT2D(neumann_matrix * DCT2D(RHS))``,
where ``DCT2D`` and ``iDCT2D`` are
Type-1 Forward and Inverse Discrete Sine 2D Trasforms respectively,
``RHS`` is the right-hand side of the equiation above,
and ``neumann_matrix`` is a 'magical' matrix that does all the work.


.. autofunction:: lcode.neumann_matrix

  In addition to the magic values, it also hosts the DCT normalization multiplier.

.. todo:: DOCS: expand with method description (Kargapolov, Shalimova)


.. autofunction:: lcode.calculate_Bz

  Note that this time the outer cells do not participate in the calculations,
  so the RHS derivatives are padded with zeroes in the beginning.


DCT2D
-----

.. autofunction:: lcode.dct2d

   As ``cupy`` currently ships no readily available function for calculating
   the DCT2D on the GPU, we roll out our own FFT-based implementation.

   We don't need to make a separate iDCT2D function
   as (for Type-1) it matches DCT2D up to the normalization multiplier,
   which is taken into account in ``neumann_matrix``.
