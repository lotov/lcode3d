Plasma
======

Characteristics
---------------

A plasma particle has these characteristics according to our model:

* Coordinates :math:`x` and :math:`y`, stored as ``x_init + x_offt`` and ``y_init + y_offt``
  [:doc:`../technicalities/offsets`].
* Momenta :math:`p_x`, :math:`p_y` and :math:`p_z`, stored as ``px``, ``py`` and ``pz``.
* Charge :math:`q`, stored as ``q``.
* Mass :math:`m`, stored as ``m``.


.. _plasma_particle_shape:

Shape
-----

From the interpolation/deposition perspective, a plasma particle represents not a point in space,
but a 2D Triangular-Shaped Cloud (TSC2D).

These clouds always (partially) cover an area the size of :math:`3x3` cells:
the one where their center and eight neighouring ones.

.. todo:: DOCS: WRITE: write a nicer formula for the weights of each cell.

.. autofunction:: lcode.weights

The same coefficients are used for both deposition of the particle characterictics onto the grid
[:doc:`deposition`]
and interpolation of the fields in the particle center positions
[:doc:`plasma_pusher`].

.. autofunction:: lcode.deposit9

.. autofunction:: lcode.interp9

The concept is orthogonal to the coarse plasma particle shape
[:doc:`coarse_and_fine_plasma`].
While a coarse particle may be treated as an elastic cloud of fine particles,
each individial fine particle sports the same TSC2D shape.
