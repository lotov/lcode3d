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


class PrintEzMax:
    @hacks.before('simulation_time_step')
    def before(self, simulation_time_step, config=None, t_i=0):
        config = lcode.configuration.get(config, t_i=0)
        self.Ez_00 = np.zeros(config.xi_steps)
        self.Ez_max = np.zeros(config.xi_steps)
        self.badness = np.zeros(config.xi_steps)
        self.trend = np.zeros(config.xi_steps)

    @hacks.into('each_xi')
    @hacks.stealing
    def plot(self, *a, xi=hacks.steal, xi_i=hacks.steal, Ez=hacks.steal):
        self.Ez_00[xi_i] = Ez[Ez.shape[0] // 2, Ez.shape[1] // 2]
        self.Ez_max[xi_i] = np.max(np.absolute(self.Ez_00))
        self.trend[xi_i] = -self.Ez_max[xi_i] / xi if xi else 0
        self.badness[xi_i] = (self.trend[xi_i] / self.trend.max()
                              if self.trend.max() else 0)

    @hacks.into('print_extra')
    @hacks.stealing
    def print(self, *a, xi_i=hacks.steal, Ez=hacks.steal):
        peaks = scipy.signal.argrelmax(self.Ez_00[:xi_i + 1])
        inclination = (self.Ez_00[peaks[-1]] / self.Ez_00[peaks[0]]
                       if len(peaks) > 1 else 0)
        return ' '.join((
            '(%0.3e)' % self.Ez_00[xi_i],
            '|%0.3e|' % self.Ez_max[xi_i],
            '<%.1f%%>' % (self.badness[xi_i] * 100),
            '[%.1f%%]' % (inclination * 100)
        ))
