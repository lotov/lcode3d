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


from nose.plugins.skip import SkipTest

import numpy as np

import hacks

import lcode.beam_depositor
import lcode.configuration
import lcode.main


CONFIG_RO_FUNC = '''
grid_steps = 2**3 + 1

beam_deposit_nearest = True

def beam(xi, x, y):
    COMPRESS, BOOST, S, SHIFT = 1, 7, 1, 0
    if xi < -2 * sqrt(2 * pi) / COMPRESS:
        return 0
    r = sqrt(x**2 + (y - SHIFT)**2)
    A = .05 * BOOST
    return A * exp(-.5 * (r/S)**2) * (1 - cos(xi * COMPRESS * sqrt(pi / 2)))
'''


CONFIG_PARTICLES = CONFIG_RO_FUNC + '''
import numpy as np

from lcode.default_config import window_width
from lcode.default_config import xi_steps, xi_step_size

import lcode.beam_construction

ro_beam_func = beam

def beam():
    for xi_i in range(xi_steps):
        xi = -xi_i * xi_step_size
        c = ((np.arange(grid_steps) + .5) * window_width / grid_steps -
             window_width / 2)
        x = c[:, None]
        y = c[None, :]
        ro = ro_beam_func(xi, x, y)
        xi_midlayer = xi - xi_step_size / 2
        sl = lcode.beam_construction.PreciselyWeighted(window_width,
                                                       grid_steps,
                                                       ro, xi=xi_midlayer)
        yield from sl
'''


@SkipTest
def test_compare_depositions():
    config_ro_func = lcode.configuration.get(CONFIG_RO_FUNC)
    config_particles = lcode.configuration.get(CONFIG_PARTICLES)

    ro_r = np.zeros((config_particles.grid_steps,)*2)
    ro_p = np.zeros_like(ro_r)

    with hacks.use(*config_particles.hacks):
        source_particles = lcode.main.choose_beam_source(config_particles)
    with hacks.use(*config_ro_func.hacks):
        source_ro_func = lcode.main.choose_beam_source(config_ro_func)

    with source_particles, source_ro_func:
        for xi_i, beam_layer in enumerate(source_particles):
            weights = np.ones(beam_layer.shape)
            ro_r[...] = lcode.beam_depositor.deposit_beam(config_particles,
                                                          beam_layer,
                                                          weights=weights)

            lcode.beam.ro_function.BeamRoFunction.beam_ro_from_function_kludge(
                config_ro_func, source_ro_func, xi_i, ro_p
            )

            assert np.array_equal(ro_r, ro_p)
