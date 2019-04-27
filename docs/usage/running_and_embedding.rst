Running and embedding
=====================

Only two files are required
---------------------------
LCODE 3D is a single-file module and you only need two files to execute it:
``lcode.py`` and ``config.py``.

Installing LCODE into `PYTHONPATH` with the likes of `pip install .` is possible,
but is not officially supported.


Configuration
-------------
LCODE 3D is configured by placing a file ``config.py`` into the current working directory.
An example is provided as ``config_example.py''.

The file gets imported by the standard Python importing mechanism,
the resulting module is passed around internally as ``config``.

One can use all the features of Python inside the configuration file,
from arithmetic expressions and functions to other modules and metaprogramming.


Execution
---------
``python3 lcode.py``, ``python lcode.py`` or ``./lcode.py``


.. todo:: CODE: embedding
