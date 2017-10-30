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


import logging

import matplotlib.pyplot as plt  # noqa: E402

import hacks  # noqa: E402

import lcode.configuration


class LiveDiagnostics:  # pylint: disable=too-many-instance-attributes
    # WARNING: SLOW!
    @hacks.before('simulation_time_step')
    def before(self, func, config, *a, t_i=0, **kwa):
        config = lcode.configuration.get(config, t_i=t_i)
        logger = logging.getLogger(__name__)
        logger.debug('Live diagnostics backend: %s', plt.get_backend())
        assert config.diagnostics
        self.quitting = False
        self.config = config
        self.diagnostics = config.diagnostics
        self.diagnostic_names = [d.name for d in self.diagnostics]
        self.data = {d: [] for d in self.diagnostics}

        self.h5diagfile, self.xi_i = None, 0

        plt.show(False)
        self.select_diagnostic(0)

        plt.connect('key_press_event', self.on_key_press)

    def select_diagnostic(self, i):
        self.active_diagnostic_i = i
        self.active_diagnostic = self.diagnostics[self.active_diagnostic_i]
        diagnostic = self.active_diagnostic
        plt.clf()
        dargs = diagnostic.plot_init(plt, self.config)
        self.data[diagnostic] = dargs
        self.replot()

    # pylint: disable=too-many-arguments, too-many-locals
    @hacks.after('diagnostics.collect')
    def after_collect(self, retval, diag, config, h5diagfile, megadict, xi_i):
        if self.active_diagnostic.name != diag.name:
            return
        self.h5diagfile = h5diagfile
        self.xi_i = xi_i
        if not self.quitting:
            self.replot()
        else:
            import sys
            sys.exit(1)

    def replot(self):
        if not self.h5diagfile:
            return
        data = self.h5diagfile[self.active_diagnostic.name][:self.xi_i + 1]
        dargs = self.data[self.active_diagnostic]
        self.active_diagnostic.plot_update(*dargs, self.xi_i, data)
        plt.draw()
        plt.pause(1e-5)

    def on_key_press(self, event):
        if event.key in ('up', 'down'):
            if event.key == 'up':
                self.active_diagnostic_i -= 1
            elif event.key == 'down':
                self.active_diagnostic_i += 1
            self.active_diagnostic_i %= len(self.diagnostics)
            self.select_diagnostic(self.active_diagnostic_i)
        if event.key == 'q':
            self.quitting = True
