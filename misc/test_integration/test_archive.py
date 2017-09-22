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


CONFIG = '''
import lcode.beam_construction

time_steps = 3
time_step_size = 1
xi_steps = 3
beam = lcode.beam_construction.some_particle_beam

hacks = [
    'lcode.beam.archive:BeamArchive'
]
def archive(t, t_i):
    return t_i == 2
'''


@lcode.util.in_temp_dir()
def test_archive():
    lcode.main.run(CONFIG)

    with h5py.File('00002.h5', 'r') as f2, \
         h5py.File('00003.h5', 'r') as f3, \
         h5py.File('archive_00003.h5', 'r') as fa:  # noqa: E127
        assert np.array_equal(f3['beam'], fa['beam'])
        assert not np.array_equal(f2['beam'], fa['beam'])
