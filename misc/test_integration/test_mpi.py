# Copyright (c) 2016-2017 LCODE team <team@lcode.info>.

# LCODE is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# LCODE is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with LCODE.  If not, see <http://www.gnu.org/licenses/>.


import os
import subprocess
import shutil
import sys
import unittest

import numpy as np
import h5py

import lcode.configuration
import lcode.main
import lcode.util

CONFIG = '''
import lcode.beam_construction

hacks = [
    'lcode.diagnostics.print_beam_info:PrintBeamInfo',
    'lcode.diagnostics.print_Ez_max:PrintEzMax',
    'lcode.beam.mpi:MPIPowers',
]

def particle():
    yield lcode.beam_particle.BeamParticle(x=1, m=1000, q=1, p_xi=1e8)
beam = {
    'beam1': lcode.beam_construction.some_particle_beam,
    'beam2': lcode.beam_construction.some_particle_beam,
    'particle1': particle,
    'particle2': particle,
}

xi_steps = 3
time_step_size = 10
time_steps = 4
'''


@lcode.util.in_temp_dir()
def test_integration_mpi():
    try:
        import mpi4py  # noqa # pylint: disable=unused-import
    except ImportError:
        raise unittest.SkipTest('mpi4py not found')
    if sys.platform != 'linux':
        raise unittest.SkipTest('Linux only for now')
    if not shutil.which('mpirun'):
        raise unittest.SkipTest('mpirun not in PATH')
    if os.geteuid() == 0:
        raise unittest.SkipTest('running as root')

    import lcode
    lcode_module_path = lcode.__path__[0]
    lcode_parent_dir = os.path.dirname(lcode_module_path)
    current_python_path = sys.executable

    with open('cfg.py', 'w') as cf:
        cf.write(CONFIG)

    cmd = '{current_python_path} {lcode_parent_dir} cfg.py'.format(**locals())
    r = subprocess.run('mpirun -n 3 ' + cmd,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       shell=True, check=True)

    with h5py.File('00004.h5', 'r') as f:
        assert np.array_equal(f['beam/beam1'], f['beam/beam2'])
        assert np.array_equal(f['beam/particle1'], f['beam/particle2'])
