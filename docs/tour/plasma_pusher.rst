Plasma pusher
=============

Without fields
--------------

The coordinate-evolving equations of motion are as follows:

.. math::
  \frac{d x}{d \xi} &= -\frac{v_x}{1-v_z}

  \frac{d y}{d \xi} &= -\frac{v_y}{1-v_z}

  \vec{v} &= \frac{\vec{p}}{\sqrt{M^2+p^2}}

.. autofunction:: lcode.move_estimate_wo_fields

   This is used at the beginning of the :doc:`xi step <xi_step>`
   to roughly estimate the half-step positions of the particles.

   The reflection here flips the coordinate, but not the momenta components.


With fields
-----------

The coordinate-evolving equation of motion is as follows:

.. math::

  \frac{d \vec{p}}{d \xi} = -\frac{q}{1-v_z} \left( \vec{E} + \left[ \vec{v} \times \vec{B} \right]\right)

As the particle momentum is present at both sides of the equation
(as :math:`p` and :math:`v` respectively),
an iterative predictor-corrector scheme is employed.

The alternative is to use a symplectic solver that solves the resulting matrix equation
(not mainlined at the moment, look for an alternative branch in ``t184256``'s fork).


.. autofunction:: lcode.move_smart_kernel

   The function serves as a single coarse particle loop,
   fusing together midpoint calculation,
   field interpolation with :func:`interp9` and
   particle movement
   for performance reasons.

   The equations for half-step momentum are solved twice,
   with more precise momentum for the second time.

   The particles coordinates are advanced using half-step momentum,
   and afterwards the momentum is advanced to the next step.

   The reflection is more involved this time, affecting both the coordinates and the momenta.

   Note that the reflected particle offsets are mixed with the positions,
   resulting in a possible float precision loss [:doc:`../technicalities/offsets`].
   This effect probably negligible at this point, as the particle had to travel
   at least several cell sizes at this point.
   The only place where the separation really matters is the (final) coordinate addition
   (``x_offt += ...`` and ``y_offt += ...``).

   The strange incantation at the top and
   the need to modify the output arrays instead of returning them
   is dictated by the fact that
   ihis is actually not a function, but a CUDA kernel
   (for more info, refer to :doc:`../technicalities/gpu`).
   It is launched in parallel for each coarse particle, determines its 1D index ``k``,
   interpolates the fields at its position and proceeds to move and reflect it.


.. autofunction:: lcode.move_smart

   This function allocates the output arrays,
   unpacks the arguments from ``config``
   calculates the kernel dispatch parameters
   (for more info, refer to :doc:`../technicalities/gpu`),
   flattens the input and output array of particle characteristics
   (as the pusher does not care about the particle 2D indices)
   and launches the kernel.
