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

import numpy as np
import h5py

import hacks

import lcode.configuration
import lcode.main
import lcode.util


MAX_ERR_RMS = 0.008
MAX_ERR_MAX = 0.018
MAX_ERR_AVG = 0.007
START_ERR_ALLOWED = 0.001

CONFIG = '''
from numpy import sqrt, exp, cos, pi

import lcode.plasma_construction

def beam(xi, x, y):
    if xi < -2 * sqrt(2 * pi):
        return 0
    r = sqrt(x**2 + y**2)
    return .05 * exp(-.5 * r**2) * (1 - cos(xi * sqrt(pi / 2)))

window_width = 12.85  # window size
grid_steps = 2**6 + 1
plasma_solver_eps = 0.0000001
plasma_solver_B_0 = 0
plasma_solver_corrector_passes = 1
plasma_solver_corrector_transverse_passes = 7
plasma_solver_particle_mover_corrector = 2
xi_step_size = .05 * 4
xi_steps = 1400 // 4
print_every_xi_steps = 140 // 4
openmp_limit_threads = 0
plasma_solver_fields_interpolation_order = -1

plasma = lcode.plasma_construction.UniformPlasma(window_width,
                                                 grid_steps,
                                                 substep=2)

from lcode.diagnostics.main import EachXi
diagnostics_enabled = True
diagnostics = [
    EachXi('Ez_00', lambda Ez: Ez[grid_steps // 2, grid_steps // 2]),
    EachXi('Bz_00', lambda Bz: Bz[grid_steps // 2, grid_steps // 2]),
]
'''


@lcode.util.in_temp_dir()
def test_against_higher_precision():
    config = lcode.configuration.get(CONFIG)

    testdir = os.path.dirname(os.path.abspath(__file__))
    HIGH = os.path.join(testdir, 'Ez_00_high.npy')
    Ez_00_high = np.load(HIGH)[:-1:4]

    with hacks.use(*config.hacks):
        lcode.main.simulation_time_step(config)

    with h5py.File('diags_00001.h5', 'r') as diags:
        Ez_00 = np.array(diags['Ez_00'])
        Bz_00 = np.array(diags['Bz_00'])

    err = np.absolute(Ez_00_high - Ez_00)
    rms = np.sqrt(np.mean(err**2))
    max_ = np.max(err)
    mean = np.mean(err)
    print('rms err =', rms, 'OK' if rms < MAX_ERR_RMS else 'FAIL')
    print('max err =', max_, 'OK' if max_ < MAX_ERR_MAX else 'FAIL')
    print('avg err =', mean, 'OK' if mean < MAX_ERR_AVG else 'FAIL')
    assert rms < MAX_ERR_RMS
    assert max_ < MAX_ERR_MAX
    assert mean < MAX_ERR_AVG

    error_limit_trend = np.linspace(START_ERR_ALLOWED, MAX_ERR_MAX, len(err))
    print(np.max(error_limit_trend - err))
    assert np.all(err < error_limit_trend)

    assert np.allclose(Bz_00, 0)
