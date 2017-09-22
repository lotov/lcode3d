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


import inspect

import hacks
import h5py
import numpy as np

import lcode.util


class EachXi:  # pylint: disable=too-few-public-methods
    def __init__(self, name, func):
        self.name = name
        self.func = func
        self.func_argnames = list(inspect.signature(func).parameters)

    @hacks.friendly('EachXi.collect', 'diagnostics.collect')
    def collect(self, config, h5diagfile, megadict, xi_i):
        argvals = [megadict[argname] for argname in self.func_argnames]

        func_result = self.func(*argvals)
        func_result = np.array(func_result)

        if self.name not in h5diagfile:
            # Dataset not created yet. Create it:
            h5diagfile.create_dataset(self.name,
                                      (config.xi_steps,),
                                      func_result.dtype)

        h5diagfile[self.name][xi_i] = func_result

    def plot_init(self, plt, config):  # pylint: disable=no-self-use
        points = plt.gca().plot([], [])[0]
        plt.xlim(0, config.xi_steps)
        ax2 = plt.gca().twiny()
        # ax2.setxlim(config.xi_steps * config.xi_step_size)
        return plt, points, ax2, config

    # pylint: disable=too-many-arguments
    def plot_update(self, plt, points, ax2, config, xi_i, data):
        plt.ylim(data.min(), data.max())
        points.set_data(np.arange(xi_i + 1), data)
        tick_locations = np.linspace(0, config.xi_steps, 5)
        ax2.set_xticks(tick_locations)
        ax2.set_xticklabels([x * config.xi_step_size for x in tick_locations])
        plt.tight_layout()
        plt.title('%s %d' % (self.name, xi_i))


class EachXi2D:  # pylint: disable=too-few-public-methods
    def __init__(self, name, func, vmin=-1, vmax=1):
        self.name = name
        self.func = func
        self.vmin, self.vmax = vmin, vmax
        self.func_argnames = list(inspect.signature(func).parameters)

    @hacks.friendly('EachXi.collect', 'diagnostics.collect')
    def collect(self, config, h5diagfile, megadict, xi_i):
        argvals = [megadict[argname] for argname in self.func_argnames]

        func_result = self.func(*argvals)
        func_result = np.array(func_result)
        self.shape = func_result.shape

        if self.name not in h5diagfile:
            # Dataset not created yet. Create it:
            h5diagfile.create_dataset(self.name,
                                      (config.xi_steps,) + func_result.shape,
                                      func_result.dtype)

        h5diagfile[self.name][xi_i] = func_result

    def plot_init(self, plt, config):
        l = np.linspace(-config.window_width / 2,
                        +config.window_width / 2,
                        config.grid_steps + 1)
        empty = np.zeros((config.grid_steps, config.grid_steps), np.float)
        cmesh = plt.pcolormesh(l, l, empty, vmin=self.vmin, vmax=self.vmax,
                               cmap='bwr')
        plt.colorbar(cmesh)
        return plt, cmesh, config

    # pylint: disable=too-many-arguments
    def plot_update(self, plt, cmesh, config, xi_i, data):
        data = data[-1]
        cmesh.set_array(data.T.flatten())
        plt.title('%s %d 2D' % (self.name, xi_i))
        plt.plot()


class DiagnosticsSliceAccessor:  # pylint: disable=too-few-public-methods
    def __init__(self, h5file, xi_i):
        self.h5file = h5file
        self.xi_i = xi_i

    def __getitem__(self, s):
        return self.h5file[s][self.xi_i]


class Diagnostics:
    @hacks.before('simulation_time_step')
    def before(self, func, config=None, t_i=0):
        # TODO: flexibility
        # TODO: diagnostics_filename
        self.enabled = False
        if 'diagnostics_enabled' in dir(config) and config.diagnostics_enabled:
            filename = lcode.util.h5_filename(t_i + 1, 'diags_{time_i}.h5')
            self.file = h5py.File(filename, 'w')
            self.enabled = True

    # pylint: disable=too-many-arguments
    @hacks.into('each_xi')
    @hacks.stealing
    def each_xi(self, config, t_i, xi_i, roj, plasma, Ex, Ey, Ez, Bx, By, Bz,
                Phi_Ezs: hacks.steal, Sz: hacks.steal):
        if not self.enabled:
            return

        # TODO: optimize!..
        context_megadict = {}
        context_megadict.update(config.__dict__)
        context_megadict.update(locals())

        context_megadict['xi'] = -config.xi_step_size * xi_i
        context_megadict['t'] = config.time_start + config.time_step_size * t_i
        context_megadict['diagnostics'] = self.file
        context_megadict['current'] = DiagnosticsSliceAccessor(self.file, xi_i)

        for x in config.diagnostics:
            context_megadict['self'] = x  # Just in case...
            x.collect(config, self.file, context_megadict, xi_i)

    @hacks.after('simulation_time_step')
    def after(self, ret, *a, **kwa):
        if self.enabled:
            self.file.close()
