GPU calculations pecularities
=============================

LCODE 3D performs most of the calculations on GPU using a mix of two approaches.


.. _cuda_kernels:

CUDA kernels with numba.cuda
----------------------------
One can use CUDA from Python more or less directly by writing and launching CUDA kernels with
``numba.cuda``.

An example would be:

.. code-block:: python

   @numba.cuda.jit
   def add_two_arrays_kernel(arr1, arr2, result):
       i = numba.cuda.grid(1)
       if i >= arr1.shape[0]:
           return
       result[i] = arr1[i] + arr2[i]

This function represents a loop body, launched in parallel with many threads at once.
Each of them starts with obtaining the array index it is 'responsible' for with `cuda.grid(1)`
and then proceeds to do the required calculation.
As it is optimal to launch them in 32-threaded 'warps', one also has to handle the case
of having more threads than needed by making them skip the calculation.

No fancy Python operations are supported inside CUDA kernels,
it basically a way to write C-like bodies for hot loops
without having to write actual C/CUDA code.
You can only use simple types for kernel arguments
and you cannot return anything from them.

To rub it in, this isn't even a directly callable function yet.
To conceal the limitations and the calling complexity,
it is convenient to write a wrapper for it.

.. code-block:: python

   def add_two_arrays(arr1, arr2):
       result = cp.zeros_like(arr1)  # uses cupy, see below
       warp_count = int(ceil(arr1.size / WARP_SIZE))
       add_two_arrays_kernel[warp_count, warp_size](arr1, arr2, result)
       return result

A pair of numbers (``warp_count``, ``WARP_SIZE``) is required to launch the kernel.
``warp_count`` is chosen this way so that ``warp_count * WARP_SIZE`` would be
larger than the problem size.


.. autodata:: lcode.WARP_SIZE


Array-wise operations with cupy
-------------------------------
``cupy`` is a GPU array library that aims to implement a ``numpy-like`` interface to GPU arrays.
It allows one to, e.g., add up two GPU arrays with a simple and terse ``a + b``.
Most of the functions in LCODE use vectorized operations and ``cupy``.
All memory management is done with ``cupy`` for consistency.

It's hard to underestimate the convenience of this approach, but sometimes expressing algorithms
in vectorized notation is too hard or suboptimal.
The only two times we're actually going for writing CUDA kernels are
:func:`deposit` (our fine particle loop) and
:func:`move_smart` (our coarse particle loop).


Copying is expensive
--------------------
If the arrays were copied between GPU RAM and host RAM, the PCI-E bandwidth would become a bottleneck.
The two most useful strategies to minimize excessive copying are

1. churning for several consecutive :math:`\xi`-steps
   with no copying and no CPU-side data processing
   (with a notable exception of :func:`beam` and the resulting `beam_ro`); and

2. copying only the subset of the arrays that the outer diagnostics code needs.


.. _array_conversion:

GPU array conversion
--------------------
In order for ``a + b`` to work in ``cupy``,
both arrays have to be copied to GPU (``cupy.asarray(a)``) and,
in case you want the results back as ``numpy`` arrays, you have to explicitly copy them back
(``gpu_array.get()``).

While for the LCODE 3D itself it's easier and quicker to stick to using GPU arrays exclusively,
this means the only time when we want to do the conversion to ``numpy`` is when we are returning
the results back to the external code.

There are two classes that assist in copying the arrays back and forth and conveniently as possible.
The implementation looks a bit nightmarish, but using them is simple.

.. autoclass:: lcode.GPUArrays

.. autoclass:: lcode.GPUArraysView

This way we can wrap everything we need in GPUArrays with, e.g.,
``const = GPUArrays(x_init=x_init, y_init=y_init, ...)``
and then access them as ``const.x_init`` from GPU-heavy code.
For the outer code that does not care about GPU arrays at all,
we can return a wrapped ``const_view = GPUArraysView(const)``
and access the arrays as ``const_view.x_init``.

Copying will happen on-demand during the attribute access,
intercepted by our ``__getattr__`` implementation,
but beware!

.. note::
   Repeatedly accessing ``const_view.x_init`` will needlessly perform the copying again,
   so one should bind it to the variable name (``x_init = const_view.x_init``) once
   and reuse the resulting ``numpy`` array.

.. todo:: CODE: wrap the returned array with GPUArraysView by default


Selecting GPU
-------------

.. autodata:: config_example.gpu_index

LCODE 3D currently does not support utilizing several GPUs for one simulation,
but once we switch to beam evolution calculation,
processing several consecutive :math:`t`-steps in a pipeline of several GPUs
should be a low hanging fruit.
