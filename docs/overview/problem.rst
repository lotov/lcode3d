Problem
=======

Objective
---------
LCODE 3D calculates the plasma response to an ultrarelativistic charged beam.

Simulating particle beams is definitely planned for the future.


.. _geometry:

Geometry
--------
Quasistatic approximation is employed, with time-space coordinate :math:`\xi = z - ct`.

From the perspective of the beam, :math:`\xi` is a space coordinate.
The head of the beam corresponds to :math:`\xi = 0`,
with its tail extending into *lower, negative* values of :math:`\xi`.

From the perspective of a plasma layer, penetrated by the beam,
:math:`\xi` is a time coordinate.
At :math:`\xi = 0` the plasma layer is unperturbed;
as the beam passes through it, :math:`\xi` values decrease.

The remaining two coordinates :math:`x, y` are way more boring
[:doc:`../overview/window_and_grids`].

The problem geometry is thus :math:`\xi, x, y`.


Beam
----
The beam is currently simulated as a charge density function :math:`\rho_b(\xi, x, y)`,
and not with particles
[:doc:`../tour/beam`].


Plasma
------
Only the electron motion is simulated,
the ions are represented with a static backround charge density
[:doc:`../tour/background_ions`].

.. math::
  \frac{d \vec{p}}{d \xi} &= -\frac{q}{1-v_z} \left( \vec{E} + \left[ \vec{v} \times \vec{B} \right]\right)

  \frac{d x}{d \xi} &= -\frac{v_x}{1-v_z}

  \frac{d y}{d \xi} &= -\frac{v_y}{1-v_z}

  \vec{v} &= \frac{\vec{p}}{\sqrt{M^2+p^2}}

The plasma is simulated using a PIC method with an optional twist:
only a 'coarse' grid of plasma (think 1 particle per 9 cells) is stored and evolved,
while 'fine' particles (think 4 per cell) are bilinearly interpolated from it during the deposition
[:doc:`../tour/coarse_and_fine_plasma`].
The plasma is effectively made not from independent particles,
but from a fabric of 'fine' TSC-2D shaped particles.


Fields
------
Both the plasma movement and the 'external' beam contribute to the charge density/currents
:math:`\rho, j_x, j_y, j_z`
[:doc:`../tour/deposition`].

The fields are calculated from their derivatives. Theoretically, the equations are

.. math::

   \Delta_\perp E_z &= \frac{\partial j_x}{\partial x} - \frac{\partial j_y}{\partial y}

   \Delta_\perp B_z &= \frac{\partial j_x}{\partial y} - \frac{\partial j_y}{\partial x}

   \Delta_\perp E_x &= \frac{\partial \rho}{\partial x} - \frac{\partial j_x}{\partial \xi}

   \Delta_\perp E_y &= \frac{\partial \rho}{\partial y} - \frac{\partial j_y}{\partial \xi}

   \Delta_\perp B_x &= \frac{\partial j_y}{\partial \xi} - \frac{\partial j_z}{\partial y}

   \Delta_\perp B_y &= \frac{\partial j_z}{\partial x} - \frac{\partial j_x}{\partial \xi}

   \Delta_\perp &= \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}

   \rho &= \rho_e + \rho_i + \rho_b

   j &= j_e + j_i + j_b

where indices :math:`e, i, b` represent electrons, ions and beam respectively.

.. note::

   In reality, things are not that simple.

   :math:`E_z` and :math:`B_z` calculations is relatively straightforward and
   boils down to solving
   the Laplace and Neumann equation with Dirichlet boundary conditions
   respectively.

   The transverse fields are actually obtained
   by solving the Helmholtz equation with mixed boundary conditions
   (so refer to :doc:`../tour/Ez`, :doc:`../tour/ExEyBxBy` and :doc:`../tour/Bz`
   for the equations that we *really* solve).


Step
----
The :math:`\xi`-cycle idea consists of looping these three actions:

* depositing plasma particles (and adding the beam density/current),
* calculating the new fields and
* moving plasma particles,

executed several times for each step in a predictor-corrector scheme
[:doc:`../tour/xi_step`].
