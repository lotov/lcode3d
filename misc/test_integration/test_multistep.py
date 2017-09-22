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
import h5py

import lcode.configuration
import lcode.main
import lcode.util

CONFIG_COMMON = '''
import lcode.beam_construction

xi_steps = 2
time_step_size = 1
'''


CONFIG_4_STEPS = CONFIG_COMMON + '''
time_steps = 4
beam = lcode.beam_construction.some_particle_beam
beam_save = '4steps_{time_i}.h5'
'''


CONFIG_STEP_1 = CONFIG_COMMON + '''
time_steps = 1
beam = lcode.beam_construction.some_particle_beam
beam_save = 'step_1.h5'
'''


CONFIG_STEP_2 = CONFIG_COMMON + '''
time_start = 1
beam = 'step_1.h5'
beam_save = 'step_2.h5'
time_steps = 1
'''


CONFIG_STEPS_34 = CONFIG_COMMON + '''
time_start = 2
beam = 'step_2.h5'
beam_save = 'steps_34_{time_i}.h5'
time_steps = 2
'''


@lcode.util.in_temp_dir()
def test_multistep():
    lcode.main.run(CONFIG_4_STEPS)

    lcode.main.run(CONFIG_STEP_1)
    lcode.main.run(CONFIG_STEP_2)
    lcode.main.run(CONFIG_STEPS_34)

    with h5py.File('4steps_00001.h5', 'r') as f_a1, \
         h5py.File('4steps_00002.h5', 'r') as f_a2, \
         h5py.File('4steps_00004.h5', 'r') as f_a4, \
         h5py.File('step_1.h5', 'r') as f_s1, \
         h5py.File('step_2.h5', 'r') as f_s2, \
         h5py.File('steps_34_00002.h5', 'r') as f_s4:  # noqa: E127

        assert np.array_equal(f_a1['beam'], f_s1['beam'])
        assert np.array_equal(f_a2['beam'], f_s2['beam'])
        assert np.array_equal(f_a4['beam'], f_s4['beam'])

        assert not np.array_equal(f_s4['beam'], f_s1['beam'])
        assert not np.array_equal(f_s4['beam'], f_s2['beam'])
        assert not np.array_equal(f_s2['beam'], f_s1['beam'])
