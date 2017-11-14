from setuptools.dist import Distribution
dist = Distribution()
dist.parse_config_files()
dist.finalize_options()
cython_extensions = [e.replace('extension=', '') for e in dist.command_options
                     if e.startswith('extension=')]


import glob
import os
pyfiles = glob.glob('lcode/**/*.py', recursive=True)
pyfiles = [f for f in pyfiles if not '__' in f]
pymodules = [f.replace(os.path.sep, '.')[:-3] for f in pyfiles]

imports = cython_extensions + pymodules + [
    'h5py._proxy',
    'h5py.defs',
    'h5py.h5ac',
    'h5py.utils',
    'numpy',
    'scipy.ndimage',
    'scipy._lib.messagestream',
]


# https://github.com/pyinstaller/pyinstaller/issues/1773
from PyInstaller.utils.hooks import collect_submodules
imports += collect_submodules('pkg_resources._vendor')

excludes = [
    'IPython',
    'PIL',
    'matplotlib',  # Temporary measure to keep the binary size adequate
]

a = Analysis(['__main__.py'], hiddenimports=imports, excludes=excludes)
pyz = PYZ(a.pure, a.zipped_data)
exe = EXE(pyz, a.scripts, a.binaries, a.zipfiles, a.datas, name='lcode')
