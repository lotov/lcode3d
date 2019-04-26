Initialization
==============

.. autofunction:: lcode.init

   This function performs quite a boring sequence of actions, outlined here for interlinking purposes:

   * validates the oddity of ``config.grid_steps``
     [:ref:`fields_and_densities_grid`],
   * validates that ``config.reflect_padding_steps`` is large enough
     [:ref:`reflect_and_plasma_boundaries`],
   * calculates the ``reflect_boundary`` and monkey-patches it back into
     ``config``,
   * initializes the ``x`` and ``y`` arrays for use in :func:`config_example.beam`,
   * calculates the plasma placement boundary,
   * immediately passes it to :func:`make_plasma`, leaving it oblivious to the padding concerns,
   * performs the initial electrion deposition to obtain the background ions charge density
     [:doc:`../tour/background_ions`],
   * groups the constant arrays into a :class:`GPUArray` instance ``const``, and
   * groups the evolving arrays into a :class:`GPUArray` instance ``state``.
