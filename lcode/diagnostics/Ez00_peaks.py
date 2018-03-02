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
import scipy.signal

import hacks

import lcode.configuration


class Ez00Peaks:
    @hacks.before('simulation_time_step')
    def before(self, simulation_time_step, config=None, t_i=0):
        self.config = lcode.configuration.get(config, t_i=t_i)
        self.Ez_00 = np.zeros(self.config.xi_steps)

    @hacks.into('each_xi')
    @hacks.stealing
    def plot(self, *a, xi=hacks.steal, xi_i=hacks.steal, Ez=hacks.steal):
        self.Ez_00[xi_i] = Ez[Ez.shape[0] // 2, Ez.shape[1] // 2]

    @hacks.into('print_extra')
    @hacks.stealing
    def print(self, *a, xi_i=hacks.steal, Ez=hacks.steal):
        Ez_00 = self.Ez_00[:xi_i + 1]
        peaks = scipy.signal.argrelmax(Ez_00)[0]
        peak_values = Ez_00[peaks]
        zero_crossings = np.where(np.diff(np.sign(Ez_00)))[0]
        if len(zero_crossings) >= 4:
            # Linear interpolation to enhance precision past one xi step
            third_crossing = zero_crossings[2]
            fix = (Ez_00[third_crossing] /
                   (Ez_00[third_crossing] - Ez_00[third_crossing - 1]))
            better_third_crossing = third_crossing + fix
            last_crossing = zero_crossings[-1]
            fix = (Ez_00[last_crossing] /
                   (Ez_00[last_crossing] - Ez_00[last_crossing - 1]))
            better_last_crossing = last_crossing + fix
            zero_period = ((better_last_crossing - better_third_crossing) * 2 *
                          self.config.xi_step_size / (len(zero_crossings) - 3))
            zero_msg = 'T=%0.5f' % zero_period
        else:
            zero_msg = 'zeroC:%d/4' % len(zero_crossings)
        if len(peaks) >= 2:
            rel_deviations_perc = 100 * (peak_values / peak_values.mean() - 1)
            peak_period = ((peaks[-1] - peaks[0]) *
                           self.config.xi_step_size / (len(peaks) - 1))
            peak_msg = ' '.join([
                '%d' % len(peaks),
                '%0.3e' % peak_values[-1],
                '%+0.2f%%' % rel_deviations_perc[-1],
                '±%0.2f%%' % (np.ptp(rel_deviations_perc) / 2),
                'Tp=%0.5f' % peak_period,
            ])
        elif len(peaks) == 1:
            peak_msg = '1 %0.3e' % peak_values[0]
        else:
            peak_msg = 'no peaks detected yet'

        return '%+0.3e|%s|%s' % (Ez_00[-1], zero_msg, peak_msg)
