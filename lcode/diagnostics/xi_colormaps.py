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

import hacks

import lcode.configuration
import lcode.util


class XiColormaps:
    @hacks.before('simulation_time_step')
    def before(self, simulation_time_step, config=None, t_i=0):
        config = lcode.configuration.get(config, t_i=t_i)
        if 'xi_colormaps_x_cut_position' not in dir(config):
            config.xi_colormaps_x_cut_position = 0

        self.Ez_colormap = np.zeros(
            (config.xi_steps, config.grid_steps)
        )
        self.Ez_Phi_colormap = np.zeros_like(self.Ez_colormap)
        self.Bz_colormap = np.zeros_like(self.Ez_colormap)
        self.ro_colormap = np.zeros_like(self.Ez_colormap)
        self.t_i = t_i

    # pylint: disable=too-many-arguments
    @hacks.into('each_xi')
    def collect_data(self, config, t_i, xi_i, roj, plasma,
                     Ex, Ey, Ez, Bx, By, Bz):
        # (x / width + .5) * grid_steps
        cut_pos_i = round(
            (config.xi_colormaps_x_cut_position / config.window_width + .5) *
            config.grid_steps
        )

        self.Ez_colormap[xi_i, :] = Ez[cut_pos_i, :]
        self.Ez_Phi_colormap[xi_i, :] = (
            np.sum(self.Ez_colormap[:xi_i + 1, :], axis=0) /
            config.xi_step_size
        )
        self.Bz_colormap[xi_i, :] = Bz[cut_pos_i, :]
        self.ro_colormap[xi_i, :] = roj['ro'][cut_pos_i, :]

    @hacks.after('simulation_time_step')
    def after(self, func, ret, *a, **kwa):
        if not os.path.isdir('xi_colormaps'):
            os.mkdir('xi_colormaps')
        dirname = os.path.join('xi_colormaps', lcode.util.fmt_time(self.t_i))
        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        def fn(name):
            return os.path.join(dirname, name + '.npy')

        np.save(fn('Ez'), self.Ez_colormap)
        np.save(fn('Ez_Phi'), self.Ez_Phi_colormap)
        np.save(fn('Bz'), self.Bz_colormap)
        np.save(fn('ro'), self.ro_colormap)
