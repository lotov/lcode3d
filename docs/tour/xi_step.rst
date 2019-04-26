Looping in xi
=============

Finally, here's the function that binds it all together,
and currently makes up half of LCODE 3D API.

In short it: moves, deposits, estimates fields, moves, deposits, recalculates fields, moves and deposits.


.. autofunction:: lcode.step


Input parameters
----------------
Beam density array `\rho_b` (``beam_ro``) is copied to the GPU with ``cupy.asarray``,
as it is calculated with ``numpy`` in config-residing :func:`beam`.

All the other arrays come packed in ``GPUArrays`` objects [:ref:`array_conversion`],
which ensures that they reside in the GPU memory.
These objects are:

* ``const`` and ``virt_params``,
  which are constant at least for the :math:`\xi`-step duration
  and defined during the :doc:`initialization <../technicalities/initialization>`, and

* ``prev``,
  which is usually obtained as the return value of the previous :func:`step` invocation,
  except for the very first step.


Initial half-step estimation
----------------------------
1. The particles are advanced according to their current momenta only
   (:func:`lcode.move_estimate_wo_fields`).


Field prediction
----------------

While we don't know the fields on the next step:

2. The particles are advanced with the fields from **the previous step**
   using the coordinates **estimated at 1.** to calculate the half-step positions
   where the **previous step** fields should be interpolated at
   (:func:`lcode.move_smart`).
3. The particles from **2.** are deposited onto the charge/current density grids
   (:func:`lcode.deposit`).
4. The fields at the next step are calculated using densities from **3.**
   (:func:`lcode.calculate_Ez`, :func:`lcode.calculate_Ex_Ey_Bx_By`, :func:`lcode.calculate_Bz`)
   and averaged with the previous fields.

This phase gives us an estimation of the fields at half-step,
and the coordinate estimation at next step,
while all other intermediate results are ultimately ignored.


Field correction
----------------
5. The particles are advanced with the **averaged** fields from **4.**,
   using the coordinates **from 2.** to calculate the half-step positions
   where the **averaged** fields from *4.* should be interpolated at
   (:func:`lcode.move_smart`).
6. The particles from **5.** are deposited onto the charge/current density grids
   (:func:`lcode.deposit`).
7. The fields at the next step are calculated using densities from **6.**
   (:func:`lcode.calculate_Ez`, :func:`lcode.calculate_Ex_Ey_Bx_By`, :func:`lcode.calculate_Bz`)
   and averaged with the previous fields.

The resulting fields are far more precise than the ones from the prediction phase,
but the coordinates and momenta are still pretty low-quality until we recalculate them
using the new fields.
Iterating the algorithm more times improves the stability,
but it currently doesn't bring much to the table as the transverse noise dominates.


Final plasma evolution and deposition
-------------------------------------
8. The particles are advanced with the **averaged** fields from **7.**,
   using the coordinates **from 5.** to calculate the half-step positions
   where the **averaged** fields from **7.** should be interpolated at
   (:func:`lcode.move_smart`).
9. The particles from **8.** are deposited onto the charge/current density grids
   (:func:`lcode.deposit`).


The result, or the 'new prev'
-----------------------------
The fields from 7., coordinates and momenta from 8., and densities from 9.
make up the new ``GPUArrays`` collection that would be passed as ``prev``
to the next iteration of :func:`step()`.
