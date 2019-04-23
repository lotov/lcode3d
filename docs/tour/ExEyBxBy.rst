Ex, Ey, Bx, By
==============

Theory
------
We want to solve

.. math::

   \Delta_\perp E_x = \frac{\partial \rho}{\partial x} - \frac{\partial j_x}{\partial \xi}

   \Delta_\perp E_y = \frac{\partial \rho}{\partial y} - \frac{\partial j_y}{\partial \xi}

   \Delta_\perp B_x = \frac{\partial j_y}{\partial \xi} - \frac{\partial j_z}{\partial y}

   \Delta_\perp B_y = \frac{\partial j_z}{\partial x} - \frac{\partial j_x}{\partial \xi}


with mixed boundary conditions.

.. todo:: DOCS: specify the boundary conditions.

Unfortunately, what we actually solve is no less than three steps away from these.


.. _helmholtz:

Helmholtz equations
-------------------
The harsh reality of numerical stability forces us to solve
these Helmholtz equations instead:

.. math::

   \Delta_\perp E_x - E_x = \frac{\partial \rho}{\partial x} - \frac{\partial j_x}{\partial \xi} - E_x

   \Delta_\perp E_y - E_y = \frac{\partial \rho}{\partial y} - \frac{\partial j_y}{\partial \xi} - E_y

   \Delta_\perp B_x - B_x = \frac{\partial j_y}{\partial \xi} - \frac{\partial j_z}{\partial y} - B_x

   \Delta_\perp B_y - B_y = \frac{\partial j_z}{\partial x} - \frac{\partial j_x}{\partial \xi} - B_y


.. note::
   The behaviour is actually configurable with
   ``config.field_solver_subtraction_trick`` (what a mouthful).
   ``0`` or ``False`` corresponds to Laplace equation,
   and while any floating-point values or whole matrices of them should be accepted,
   it's recommended to simply use ``1`` or ``True`` instead.


Method
------
The algorithm can be succinctly written as ``iMIX2D(mixed_matrix * MIX2D(RHS))``,
where ``MIX2D`` and ``iMIX2D`` are
Type-1 Forward and Inverse Discrete Trasforms, Sine in one direction and Cosine in the other.
``RHS`` is the right-hand side of the equiation above,
and ``dirichlet_matrix`` is a 'magical' matrix that does all the work.


.. autofunction:: lcode.mixed_matrix

.. todo:: DOCS: expand with method description (Kargapolov, Shalimova)


.. autofunction:: lcode.calculate_Ex_Ey_Bx_By

  Note that some outer cells do not participate in the calculations,
  and the result is simply padded with zeroes at the end.
  We don't define separate functions for separate boundary condition types
  and simply transpose the input and output data.


DST-DCT Hybrid
--------------

.. autofunction:: lcode.mix2d

   As ``cupy`` currently ships no readily available function for calculating
   even 1D DST/DCT on the GPU,
   we, once again, roll out our own FFT-based implementation.


.. _variant_a:

Variant B
---------
But wait, the complications don't stop here.

While we do have a succesfully implemented :math:`(\Delta_\perp - 1)` inverse operator,
there's still an open question of supplying an unknown value to the RHS.

The naive version (internally known as 'Variant B')
is to pass the best known substitute to date, i.e.
previous layer fields at the predictor phase
and the averaged fields at the corrector phase.
:math:`xi` derivatives are taken at half-steps,
transverse derivatives are averaged at half-steps
or taken from the previous layer if not available.

.. math::

   (\Delta_\perp - 1) E_x^{next} = \frac{\partial \rho^{prev}}{\partial x} - \frac{\partial j_x^{avg}}{\partial \xi} - E_x^{avg}

   (\Delta_\perp - 1) E_y^{next} = \frac{\partial \rho^{prev}}{\partial y} - \frac{\partial j_y^{avg}}{\partial \xi} - E_y^{avg}

   (\Delta_\perp - 1) B_x^{next} = \frac{\partial j_y^{avg}}{\partial \xi} - \frac{\partial j_z^{prev}}{\partial y} - B_x^{avg}

   (\Delta_\perp - 1) B_y^{next} = \frac{\partial j_z^{prev}}{\partial x} - \frac{\partial j_x^{avg}}{\partial \xi} - B_y^{avg}


Variant A
---------

The more correct version (known as 'Variant A')
mutates the equations once again to take everything at half-steps:

.. math::

   (\Delta_\perp - 1) (2 E_x^{avg} - E_x^{avg}) = \frac{\partial \rho^{prev}}{\partial x} - \frac{\partial j_x^{avg}}{\partial \xi} - E_x^{avg}

   (\Delta_\perp - 1) (2 E_y^{avg} - E_y^{avg}) = \frac{\partial \rho^{prev}}{\partial y} - \frac{\partial j_y^{avg}}{\partial \xi} - E_y^{avg}

   (\Delta_\perp - 1) (2 B_x^{avg} - B_x^{avg}) = \frac{\partial j_y^{avg}}{\partial \xi} - \frac{\partial j_z^{prev}}{\partial y} - B_x^{avg}

   (\Delta_\perp - 1) (2 B_y^{avg} - B_y^{avg}) = \frac{\partial j_z^{prev}}{\partial x} - \frac{\partial j_x^{avg}}{\partial \xi} - B_y^{avg}

and calculates the fields at next step in the following fashion: :math:`E_x^{next} = 2 E_x^{avg} - E_x^{prev})`, e.t.c.

Solving these is equivalent to solving Variant B equations
with averaged :math:`\rho` and :math:`j_z` and applying the above transformation to the result.
See `step(...)` function for the wrapping code that does that.

.. autodata:: config_example.field_solver_variant_A
   :annotation: =True

