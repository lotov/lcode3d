Installation
============

Common
------
LCODE 3D requires an NVIDIA GPU with CUDA support.
CUDA Compute Capability 6+ is strongly recommended
for accelerated atomic operations support.

On the Python front, it needs Python 3.6+ and the packages listed in ``requirements.txt``:

.. literalinclude:: ../../requirements.txt

Most of them are extremely popular
and the only one that may be slightly problematic to obtain due to its 'teen age' is ``cupy``.


Linux, distribution Python
--------------------------
All the dependencies, except for, probably, ``cupy``,
should be easily installable with your package manager.

Install NVIDIA drivers and CUDA packages according to your distribution documentation.

Install ``cupy`` according to the
`official installation guide <https://docs-cupy.chainer.org/en/stable/install.html>`_,
unless ``5.1`` or newer is already packaged by your distribution.


Linux, Anaconda
---------------
All dependencies, including ``cupy``, are available from the official conda channels.

.. code-block:: bash

   conda install cupy

or, if you are a miniconda user (or you just like to be thorough),

.. code-block:: bash

   while read req; do conda install --yes $req; done < requirements.txt

You probably still need to install NVIDIA drivers and CUDA packages,
follow your distribution documentation.


Linux, NixOS
---------------
.. code-block:: bash

   nix-shell

In case it's not enough, consider adding

.. code-block:: nix

   boot.kernelPackages = pkgs.linuxPackages;  # explicitly require stable kernel
   boot.kernelModules = [ "nvidia-uvm" ];

to ``/etc/nixos/configuration.nix`` and rebuilding the system.


Linux, locked-down environment
------------------------------
If want to, e.g., run LCODE 3D on a cluster without permissions
to install software the proper way, please contact the administrator first
and refer them to this page.

If you are sure about CUDA support and you absolutely want to install the dependencies yourself,
then make sure you have Python 3.6+ and
try to install ``cupy`` using the
official installation guide.
If you succeed, install all the other missing requirements with ``pip``'s
'User Installs' feature.
You mileage may vary. You're responsible for the cleanup.


Windows, Anaconda
-----------------
If ``cupy`` ``5.1`` or newer has already hit the channels, you're in luck.
Just ``conda install cupy``, and you should be good to go.

If https://anaconda.org/anaconda/cupy still shows 'win-64' at ``v4.1.0``,
please accept our condolences and proceed to the next subsection.


Windows, the hard way
---------------------
* Ensure that you have Python 3.6+.
* Free up some 10 GiB of disk space or more.
* Verify that you're on good terms with the deity of your choice.
* Install Visual Studio (Community Edition is fine) with C++ support.
* Install NVIDIA CUDA Toolkit.
* Follow the ``cupy`` installation guide.
* Prefer installing precompiled packages,
  but you might also try installing from source.
* Verify that it works by executing ``import cupy; (cupy.asarray([2])**2).get()``
  in Python shell.
* Install the other dependencies.


Known to work
-------------
As of early 2019, LCODE 3D is developed and known to work under:

* NixOS 19.03 "Koi"
* Debian 10 "Buster" + Anaconda 2019.03
* Windows 10 1903 + Anaconda 2019.03
