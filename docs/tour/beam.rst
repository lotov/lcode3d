Beam
====

The beam is currently simulated as a charge density function :math:`\rho_b(\xi, x, y)`,
and not with particles.

In the future, there will certaintly be a way to define a beam
with particles and simulate beam-plasma interaction both ways,
but for now only simulating a plasma response to a rigid beam is possible.

.. autofunction:: config_example.beam

   The user should specify the beam charge density as a function in the configuration file.

   ``xi_i`` is not the value of the :math:`\xi` coordinate, but the step index.
   Please use something in the lines of ``xi = -xi_i * xi_step_size + some_offset``,
   according to where exactly in :math:`\xi` do you define the beam density slices.

   ``x`` and ``y`` are ``numpy`` arrays, so one should use vectorized numpy operations
   to calculate the desired beam charge density, like ``numpy.exp(-numpy.sqrt(x**2 + y**2))``.

   The function should ultimately return an array with the same shape as ``x`` and ``y``.

.. todo:: CODE: Simulate the beam with particles and evolve it according to the plasma response.
