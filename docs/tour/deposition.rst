Deposition
==========

Deposition operates on fine particles.
Once the :doc:`coarse-to-fine interpolation <../tour/coarse_and_fine_plasma>` is out of the picture,
there isn't much left to discuss.

.. autofunction:: lcode.deposit_kernel

   First, the fine particle characteristics are interpolated from the coarse ones.
   Then the total contribution of the particles to the density and the currents
   is calculated and, finally,
   deposited on a grid in a 3x3 cell square with ``i``, ``j`` as its center
   according to the weights calculated by :func:`weights`.
   Finally, the `ion background density <../tour/background_ions>`
   is added to the resulting array.

   The strange incantation at the top and
   the need to modify the output arrays instead of returning them
   is dictated by the fact that
   ihis is actually not a function, but a CUDA kernel
   (for more info, refer to :doc:`../technicalities/gpu`).
   It is launched in parallel for each fine particle, determines its index,
   interpolates its characteristics from coarse particles and proceeds to deposit it.
   It determines the fine particle indices.

.. autofunction:: lcode.deposit

   This function allocates the output arrays,
   unpacks the arguments from ``config`` and ``virt_params``,
   calculates the kernel dispatch parameters
   (for more info, refer to :doc:`../technicalities/gpu`),
   and launches the kernel.

.. todo:: DOCS: explain deposition contribution formula (Lotov)
