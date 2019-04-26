Offset-coordinate separation
============================

Float precision loss
--------------------
When floating point numbers of different magnitudes get added up,
there is an inherent precision loss that grows with the magnitude disparity.

If a particle has a large coordinate (think ``5.223426``),
but moves for a small distance (think ``7.139152e-4``) due to low ``xi_step_size``
and small momentum projection, calculating the sum of these numbers
suffers from the precision loss due to the finite significand size:

An oversimplified illustration in decimal notation::

    5.223426
   +0.0007139152
   =5.224139

We have not conducted extensive research on how detrimental this round-off accumulation
is to LCODE 3D numerical stability in :math:`\xi`.
Currently the transverse noise dominates,
but in order to make our implementation a bit more future-proof,
we store the plasma particle coordinates separated into two floats:
initial position (``x_init``, ``y_init``) and accumulated offset (``x_offt``, ``y_offt``)
and do not mix them.


Mixing them all the time...
---------------------------
OK, we do mix them. Each and every function involving them adds them up at some point
and even has the code like this:

.. code-block:: python

   x = x_init + x_offt
   ...
   x_offt = x - x_init

to reconstruct ``x_offt`` from the 'dirty' sum values ``x``.

We do that because we're fine with singular round-off errors until they don't propagate
to the next step, accumulating
for millions of :math:`\xi`-steps ('Test 1' simulations were conducted for up to 1.5 million steps).


... but not where it really matters
-----------------------------------

This way the only places where the separation should be preserved
is the path from ``prev.x_offt`` to ``new_state.x_offt``.
Several ``x_offt`` additions are performed and rolled back
at each :math:`\xi`-step,
but only two kinds of them persist, all of them residing in :func:`move_smart`:

1. ``x_offt += px / (gamma_m - pz) * xi_step_size`` does no mixing with the coordinate values, and

2. ``x = +2 * reflect_boundary - x`` and the similar one for the left boundary
   only happen during particle reflection,
   which presumably happens rarely and only affects the particles that have already deviated
   at least several cells away from the initial position.

This way most particles won't experience this kind of rounding issues with their coordinates.
On the flip side, splitting the coordinates makes working with them quite unwieldy.
