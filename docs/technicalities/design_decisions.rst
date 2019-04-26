Design decisions
================

Codebase complexity
-------------------
The code strives to be readable and vaguely understandable
by a freshman student with only some basic background
in plasma physics and numerical simulations,
to the point of being comfortable with modifying it.

Given the complexity of the physical problem behind it,
this goal is, sadly, unattainable,
but the authors employ several avenues to get as close as possible:

1. Abstaining from using advanced programming concepts.
   Cool things like aspect-oriented programming are neat,
   but keeping the code well under 1000 SLOC is even neater.
   The two classes we currently have is two classes over the ideal amount of them.
2. Preferring less code over extensibility.
3. Picking simpler code over performance tweaking.
4. Choosing malleability over user convenience.
5. Creating external modules or branches over exhaustive featureset.
6. Not shying away from external dependencies or unpopular technologies,
   even if this means sacrificing the portability.
7. Appointing a physics student with modest programming background
   as the maintainer and primary code reviewer.


Codebase size
-------------
LCODE 3D wasn't always around 500 SLOC.
In fact, even as it got rewritten from C to Cython to numba to numba.cuda to cupy,
it peaked at around 5000 SLOC, twice.
And we don't even count its Fortran days.

In order to objectively curb the complexity, scope and malleability of the codebase,
its size is limited to 1000 SLOC.

David Wheeler's SLOCCount is used for obtaining the metric.
Empty lines, docstrings and comments don't count towards that limit.


.. _zero_special_boundary_treatment:

Zero special boundary treatment
-------------------------------

[:ref:`Not allowing the particles to reach the outer cells of the simulation window <reflect_and_plasma_boundaries>`]
slightly modifies the physical problem itself, but, in return
blesses us with the ability to forego special boundary checks during deposition, interpolation and field calculation,
simplifying the code and boosting the performance.


.. _memory_considerations:

Memory considerations
---------------------

LCODE 3D is observed to consume roughly the same amount of host and GPU RAM,
hovering around 500 MiB for a 641x641 grid, coarseness=3 and fineness=2.

The size of the arrays processed by LCODE 3D depends on these parameters.

Let's label the field/densities grid size in a single direction as :math:`N`,
coarse plasma grid size as :math:`N_c \approx \frac{N}{\text{coarseness}}` and
fine plasma grid size as :math:`N_f \approx N * \text{fineness}`.

With about the same amount of arrays in scope for each of these three sizes,
it is clear that the :math:`N_f^2`-sized arrays would dominate the memory consumption.
Fortunately, the arrays that contain fine plasma characteristics would be transient and only used during the deposition,
while the interpolation indices and coefficients grouped under ``virt_params``
can be reduced to 1D arrays by exploiting the :math:`x/y` symmetry of the coarse/fine plasma grids.

This way LCODE 3D stores only :math:`N_c^2`- and :math:`N^2`-sized arrays,
with `N_f`-sized ones barely taking up any space thanks to the being 1D.

Also, all previous attempts to micromanaged the GPU memory allocations have been scraped
in favor of blindly trusting the ``cupy`` on-demand allocation.
Not only it is extremely convenient, it's even more performant than our own solutions.


.. _integer_xi_steps:

Integer xi steps
----------------

:math:`\xi`-steps are integer for the purpose of bypassing float precision-based errors.
The task of converting it into the :math:`\xi`-coordinate is placed within the usage context.
