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

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import hacks


class QuickDensity:
    @staticmethod
    @hacks.before('simulation_time_step')
    def before(func, *a, **kwa):
        if not os.path.isdir('transverse'):
            os.mkdir('transverse')

    # pylint: disable=too-many-arguments
    @staticmethod
    @hacks.into('each_xi')
    @hacks.stealing
    def plot(config, t_i, xi_i, roj, plasma,
             Ex, Ey, Ez, Bx, By, Bz):
        xi = -config.xi_step_size * xi_i

        if not xi_i or not config.transverse_peek_enabled(xi, xi_i):
            return

        ro = roj['ro'].repeat(3, axis=0).repeat(3, axis=1)
        plt.imsave(os.path.join('transverse', 'ro_%09.2f.png' % xi),
                   ro.T, origin='lower',
                   vmin=-0.1, vmax=0.1,
                   cmap=cm.bwr)
