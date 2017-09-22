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


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import hacks  # noqa: E402

import lcode.util  # noqa: E402


class BeamPortrait:
    @hacks.before('simulation_time_step')
    def init(self, simulation_time_step, config=None, t_i=0):
        self._xs = []
        self._ys = []
        self.t_i = t_i

    @hacks.into('diagnose_layer')
    def store_coordinates(self, layer):
        self._xs += list(layer['r'][:, 1])
        self._ys += list(layer['r'][:, 2])

    @hacks.after('simulation_time_step')
    def save_plot(self, retval, *a, **kwa):
        filename = lcode.util.fmt_time(self.t_i) + '_%6f_beam_portrait.png' % 0

        plt.scatter(self._xs, self._ys)
        plt.savefig(filename)
        plt.close()
