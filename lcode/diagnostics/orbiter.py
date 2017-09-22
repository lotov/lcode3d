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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import hacks  # noqa: E402


class Orbiter:
    # WARNING: SLOW!
    @hacks.before('simulation_time_step')
    def before(self, func, *a, **kwa):
        if not os.path.isdir('force_cut'):
            os.mkdir('force_cut')
        self.x = []
        self.y = []
        self.p = []

    # pylint: disable=too-many-arguments, too-many-locals
    @hacks.into('each_xi')
    @hacks.stealing
    def plot(self, config, t_i, xi_i, roj, plasma,
             Ex, Ey, Ez, Bx, By, Bz,
             plasma_solver_config=hacks.steal):
        # tracked = config.track_plasma_particles(plasma)
        tracked = plasma

        x = tracked['x'][0]
        y = tracked['y'][0]
        p = np.sqrt(tracked['p'][0, 1]**2 + tracked['p'][0, 2]**2)
        self.x.append(x)
        self.y.append(y)
        self.p.append(p)

        probes = tracked[0]

        Exs, Eys, Ezs, Bxs, Bys, Bzs = [np.zeros(len(probes))
                                        for i in range(6)]
        Ext, Eyt, Ezt, Bxt, Byt, Bzt = [np.zeros(1) for i in range(6)]

        cells = np.linspace(-config.window_width / 2, +config.window_width / 2,
                            config.grid_steps + 1)
        assert len(cells) == config.grid_steps + 1

        # if config.plasma_solver_fields_interpolation_order == -1:
        #     interpolate_fields = plasma_solver.interpolate_fields_fs
        # elif config.plasma_solver_fields_interpolation_order == -2:
        #     interpolate_fields = plasma_solver.interpolate_fields_fs9
        # else:
        #     interpolate_fields = plasma_solver.interpolate_fields_sl

        xi = -config.xi_step_size * xi_i

        if 'force_cut_enabled' not in dir(config):
            return
        if not config.force_cut_enabled(xi, xi_i):
            return

        plt.figure(figsize=(10, 10), dpi=100)

        plt.title(
            r'Ion and electron (orange), small initial kick, '
            r'$x=%.3e$, $p=%.3e$, $\xi=%.3f$' % (x, p, xi)
        )
        plt.xlabel('x')
        plt.ylabel('y')

        # Cell borders
        for c in cells:
            plt.axvline(c, color='lightgray')
            plt.axhline(c, color='lightgray')

        # The original particle
        plt.scatter(tracked['x'][0], tracked['y'][0], s=15, color='orange')
        plt.scatter(tracked['x'][1], tracked['y'][1], s=15, color='red')

        plt.plot(self.x, self.y, color='orange')
        plt.plot(self.x, np.array(self.p) * 10, color='cyan')

        plt.legend()

        plt.savefig(os.path.join('force_cut', '%09.2f.png' % xi))
        plt.close()
