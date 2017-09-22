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


CONFIG_1 = '''
import lcode.beam_construction

xi_steps = 3
beam = lcode.beam_construction.some_particle_beam
beam_save = 'out.h5'
'''


CONFIG_2 = '''
time_step_size = 1
beam = 'out.h5'
beam_save = 'out_evolved_more.h5'
xi_steps = 3
'''


CONFIG_3 = '''
time_step_size = 0
beam = 'out.h5'
beam_save = 'out_not_evolved.h5'
xi_steps = 3
'''


@lcode.util.in_temp_dir()
def test_files():
    lcode.main.run(CONFIG_1)
    lcode.main.run(CONFIG_2)
    lcode.main.run(CONFIG_3)

    with h5py.File('out.h5', 'r') as f1, \
         h5py.File('out_evolved_more.h5', 'r') as f2, \
         h5py.File('out_not_evolved.h5', 'r') as f3:  # noqa: E127
        # dt == 0, x must not change
        assert np.array_equal(f1['beam'], f3['beam'])
        # dt != 0, x must change
        assert not np.array_equal(f1['beam'], f2['beam'])
