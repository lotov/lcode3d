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

from ..plasma import solver_v2_monolithic as plasma_solver  # noqa: E402


class ForceCut:
    # WARNING: SLOW!
    @hacks.before('simulation_time_step')
    def before(self, func, *a, **kwa):
        if not os.path.isdir('force_cut'):
            os.mkdir('force_cut')
        self.x = []
        self.p = []
        self.dp_dxi = []
        self.Ex = []

    # pylint: disable=too-many-arguments, too-many-locals, too-many-statements
    @hacks.into('each_xi')
    @hacks.stealing
    def plot(self, config, t_i, xi_i, roj, plasma,
             Ex, Ey, Ez, Bx, By, Bz,
             plasma_solver_config=hacks.steal):
        tracked = config.track_plasma_particles(plasma)
        #tracked = plasma

        x = tracked['x'][0]
        p = tracked['p'][0, 1]
        p_prev = self.p[-1] if self.p else 0
        dp_dxi = p - p_prev / config.xi_step_size
        self.x.append(x)
        self.p.append(p)
        self.dp_dxi.append(dp_dxi)

        SUBCELL = 30
        G = 6  # 2x gap from the edge
        grid_step = config.window_width / config.grid_steps
        less_half_width = (config.window_width - G * grid_step) / 2
        less_grid_steps = config.grid_steps - G
        xs = np.linspace(-less_half_width, less_half_width,
                         less_grid_steps * SUBCELL)
        cells = np.linspace(-less_half_width, less_half_width,
                            less_grid_steps + 1)
        cell_centers = (cells[:-1] +
                        .5 * grid_step)
        assert len(cell_centers) == Ex.shape[0] - 6
        probes = tracked.repeat(less_grid_steps * SUBCELL)
        probes['x'] = xs

        Exs, Eys, Ezs, Bxs, Bys, Bzs = [np.zeros(len(probes))
                                        for i in range(6)]
        Ext, Eyt, Ezt, Bxt, Byt, Bzt = [np.zeros(1) for i in range(6)]

        interpolate_fields = plasma_solver.interpolate_fields_fs9

        for x_scan in np.linspace(-config.window_width / 2,
                                  +config.window_width / 2,
                                  config.grid_steps * 20):

            interpolate_fields(plasma_solver_config,
                               probes['x'],
                               probes['y'],
                               Ex, Ey, Ez, Bx, By, Bz,
                               Exs, Eys, Ezs, Bxs, Bys, Bzs,
                               np.zeros((config.grid_steps,) * 2))

            interpolate_fields(plasma_solver_config,
                               tracked['x'],
                               tracked['y'],
                               Ex, Ey, Ez, Bx, By, Bz,
                               Ext, Eyt, Ezt, Bxt, Byt, Bzt,
                               np.zeros((config.grid_steps,) * 2))

        self.Ex.append(Ext)

        xi = -config.xi_step_size * xi_i

        if 'force_cut_enabled' not in dir(config):
            return
        if not config.force_cut_enabled(xi, xi_i):
            return

        # FullHD, 1920x1080, 16:9
        plt.figure(figsize=(19.2, 10.8), dpi=100)

        plt.title(
            r'Single electron (orange), small initial kick, '
            r'$x=%.3e$, $p_x=%.3e$, $\xi=%.3f$' % (tracked['x'][0],
                                                   tracked['p'][0][1],
                                                   xi)
        )
        plt.xlabel('x')
        plt.ylabel('Ex')

        plt.axvline(tracked['x'][0])

        # Cell borders
        for c in cells:
            plt.axvline(c, color='lightgray')
        # zero level
        plt.axhline(0, color='lightgray')

        # The original particle (line hint)
        plt.axvline(tracked['x'][0], color='orange')

        # Ex from a particle as seen by probe particles, interpolated
        plt.ylim(-0.07, 0.07)
        plt.plot(xs, Exs, '-', color='blue', label='Ex, interpolated')

        # Ex from a particle, not interpolated
        tracked_y_cell = int(
            config.grid_steps *
            (tracked['y'][0] + config.window_width / 2) / config.window_width
        )
        plt.plot(cell_centers, Ex[:, tracked_y_cell][3:-3], '*', markersize=5,
                 color='black', label='Ex, not interpolated')

        # The original particle
        plt.axvline(tracked['x'][0], color='orange')
        plt.scatter(tracked['x'], Ext, s=15, color='orange')

        if len(self.dp_dxi) > 1:
            plt.plot(self.x[1:], np.array(self.dp_dxi[1:]) * 10, color='pink', label='self-acceleration x 10')
            plt.plot(np.array(self.p[1:]) * 10, np.array(self.Ex[1:]) * 10, color='lightgreen', label='Ex(dp_dxi) x10 both')
        plt.plot(self.x, self.p, color='cyan', label='p')
        plt.plot(self.x, self.Ex, color='orange', label='self-Ex')
        plt.plot(self.x, np.array(self.Ex) * 10, color='yellow', label='self-Ex x10')

        plt.legend()

        plt.savefig(os.path.join('force_cut', '%09.2f.png' % xi))
        plt.close()
