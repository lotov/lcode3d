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


import imp
import inspect

import numpy as np

import hacks

import lcode.configuration


class FieldsXYDependence:  # pylint: disable=too-few-public-methods
    @staticmethod
    @hacks.around('simulation_time_step')
    def expand_fields(simulation_time_step):
        def EB_expanded_simulation_time_step(config=None, t_i=0):
            '''Call simulation_time_step expanding Ez(x, y) and friends.'''
            config = lcode.configuration.get(config, t_i=t_i)
            v = np.linspace(-config.window_width / 2, config.window_width / 2,
                            config.grid_steps)
            x, y = np.meshgrid(v, v, indexing='ij')
            # a really shallow copy of the config
            expanded_config = imp.new_module(config.__name__)
            expanded_config.__dict__.update(config.__dict__)
            for component_name in 'Ez', 'Ex', 'Ey', 'Bz', 'Bx', 'By':
                if inspect.isroutine(getattr(config, component_name)):
                    func = getattr(config, component_name)
                    params = list(inspect.signature(func).parameters)
                    if params == ['x', 'y']:
                        setattr(expanded_config, component_name, func(x, y))
            return simulation_time_step(expanded_config, t_i)
        return EB_expanded_simulation_time_step
