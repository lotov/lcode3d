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
import matplotlib.cm as cm  # noqa: E402

import hacks  # noqa: E402

from .. import plasma_solver  # noqa: E402


class TransversePeek:
    # WARNING: SLOW!
    @staticmethod
    @hacks.before('simulation_time_step')
    def before(func, *a, **kwa):
        if not os.path.isdir('transverse'):
            os.mkdir('transverse')

    # pylint: disable=too-many-arguments, too-many-locals
    @staticmethod
    @hacks.into('each_xi')
    @hacks.stealing
    def plot(config, t_i, xi_i, roj, plasma,
             Ex, Ey, Ez, Bx, By, Bz,
             plasma_solver_config=hacks.steal):
        xi = -config.xi_step_size * xi_i

        if not xi_i:
            return
        if 'transverse_peek_enabled' not in dir(config):
            return
        if not config.transverse_peek_enabled(xi, xi_i):
            return

        l = np.linspace(-config.window_width / 2,
                        +config.window_width / 2,
                        config.grid_steps)

        tracked = config.track_plasma_particles(plasma)

        # FullHD, 1920x1080, 16:9, 3x2
        fig, ax = plt.subplots(2, 3, figsize=(19.2, 10.8), dpi=100)
        for a in ax.flatten():
            a.axis([-config.window_width / 2, +config.window_width / 2] * 2)
        (ax_Ez, ax_Bz, ax_ro), (ax_Exy, ax_Bxy, ax_jxy) = ax

        ax_Ez.set_title('Ez')
        im_Ez = ax_Ez.pcolormesh(l, l, Ez.T, vmin=-.15, vmax=.15, cmap=cm.bwr)
        fig.colorbar(im_Ez, ax=ax_Ez)

        ax_Bz.set_title('Bz')
        im_Bz = ax_Bz.pcolormesh(l, l, Bz.T, vmin=-.05, vmax=.05, cmap=cm.bwr)
        fig.colorbar(im_Bz, ax=ax_Bz)

        ax_ro.set_title('ro, E (orange), B (green), speed (black)')
        im_ro = ax_ro.pcolormesh(l, l, roj['ro'].T,
                                 vmin=-.5, vmax=.5, cmap=cm.bwr)
        fig.colorbar(im_ro, ax=ax_ro)

        # Tracked particles Ex, green arrows
        tEx, tEy, tEz, tBx, tBy, tBz = (np.empty_like(tracked['x'])
                                        for i in range(6))
        plasma_solver.interpolate_fields_fs(plasma_solver_config,
                                            tracked['x'], tracked['y'],
                                            Ex, Ey, Ez, Bx, By, Bz,
                                            tEx, tEy, tEz, tBx, tBy, tBz)
        if (np.any(tEx) or np.any(tEy)):
            ax_ro.quiver(tracked['x'], tracked['y'],
                         tEx, tEy, scale=1,
                         width=.004, alpha=.5, color='green')
        # Tracked particles Bx, orange arrows
        if (np.any(tBx) or np.any(tBy)):
            ax_ro.quiver(tracked['x'], tracked['y'],
                         tBx, tBy, scale=1,
                         width=.004, alpha=.5, color='orange')
        # Tracked particles speed, black arrows
        if (np.any(tracked['v'][:, 1:])):
            ax_ro.quiver(tracked['x'], tracked['y'],
                         tracked['v'][:, 1], tracked['v'][:, 2], scale=1,
                         width=.004, alpha=.5, color='black')
        # Tracked particles, circles
        ax_ro.scatter(tracked['x'], tracked['y'], alpha=.5, s=8)

        vnorm = matplotlib.colors.Normalize(vmin=0, vmax=.2)

        if (np.any(Ex) or np.any(Ey)):
            ax_Exy.set_title('Ex, Ey')
            Et = np.sqrt(Ex**2 + Ey**2)
            im_Exy = ax_Exy.streamplot(l, l, Ex.T, Ey.T,
                                       cmap=cm.cool, norm=vnorm,
                                       linewidth=0.125 + 5 * Et.T,
                                       color=np.sqrt(Ex**2 + Ey**2).T)
            fig.colorbar(im_Exy.lines, ax=ax_Exy)

        if (np.any(Bx) or np.any(By)):
            ax_Bxy.set_title('Bx, By')
            Bt = np.sqrt(Bx**2 + By**2)
            im_Bxy = ax_Bxy.streamplot(l, l, Bx.T, By.T,
                                       cmap=cm.cool, norm=vnorm,
                                       linewidth=0.125 + 5 * Bt.T,
                                       color=np.sqrt(Bx**2 + By**2).T)
            fig.colorbar(im_Bxy.lines, ax=ax_Bxy)

        jx, jy = roj['jx'], roj['jy']
        jt = np.sqrt(jx**2 + jy**2).T
        if min(np.max(np.abs(jx)),
               np.max(np.abs(jy)),
               np.max(np.abs(jt))) > 1e-10:
            ax_jxy.set_title('jx, jy')
            im_jxy = ax_jxy.streamplot(l, l, jx.T, jy.T,
                                       cmap=cm.cool, norm=vnorm,
                                       linewidth=0.125 + 5 * jt,
                                       color=50*jt)
            fig.colorbar(im_jxy.lines, ax=ax_jxy)

        plt.savefig(os.path.join('transverse', '%09.2f.png' % xi))
        plt.close()
