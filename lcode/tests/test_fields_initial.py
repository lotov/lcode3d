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


import numpy as np

import hacks

import lcode.configuration
import lcode.fields_initial


CONFIG_TIME_AND_XY_DEPENDENCE = '''
time_step_size = 1
<% t = t_i * 1 %>

A = 4

% for z in range(0, 30 * 1000, 1000):  # generate 30 quadrupole pairs

# ${z} ${t} ${t_i}
% if 0 + z <= t < 200 + z:  # first orientation from 0 to 200 each 1000
Bx = lambda x, y: A * y
By = lambda x, y: A * x
% elif 500 + z <= t < 700 + z:  # second orientation from 500 to 700
Bx = lambda x, y: -A * x
By = lambda x, y: A * y
% endif

% endfor


# t-function style w/o code generation

def Bz(t):
    if t < 700:
        return lambda x, y: x
    return 0
'''


def test_fields_t_xy_dependence():
    @hacks.friendly('simulation_time_step')
    def fake_simulation_time_step(config, t_i=0):
        c = lcode.configuration.get(config, t_i=t_i)
        v = np.linspace(-c.window_width / 2, c.window_width / 2, c.grid_steps)
        x, y = np.meshgrid(v, v, indexing='ij')
        if t_i == 50:
            assert np.array_equal(c.Bx, 4 * y)
            assert np.array_equal(c.By, 4 * x)
            assert np.array_equal(c.By[:, 0], 4 * v)
            assert np.array_equal(c.Bz, x)
        elif t_i == 800:
            assert c.Bx == 0
            assert c.By == 0
            assert c.Bz == 0
            assert c.Bx == c.By == c.Bz == 0
        elif t_i == 1600:
            assert np.array_equal(c.Bx, -4 * x)
            assert np.array_equal(c.By, 4 * y)
            assert np.array_equiv(c.Bz, 0)

    with hacks.use(lcode.fields_initial.FieldsXYDependence,
                   lcode.configuration.TimeDependence):
        for t_i in 50, 800, 1600:
            fake_simulation_time_step(CONFIG_TIME_AND_XY_DEPENDENCE, t_i)
