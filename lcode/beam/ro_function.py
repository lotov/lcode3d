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

import numpy as np

import hacks

import lcode.beam_construction
import lcode.beam_particle


class BeamRoFunctionFakeSource:  # pylint: disable=too-few-public-methods
    def __init__(self, config, ro_func):
        self.ro_func = ro_func
        c = ((np.arange(config.grid_steps) + .5) *
             config.window_width / config.grid_steps -
             config.window_width / 2)
        self.x = c[:, None]
        self.y = c[None, :]

    def __enter__(self):
        return self

    def __next__(self):  # pylint: disable=no-self-use
        return {}

    def __iter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __repr__(self):
        return '<' + self.__class__.__name__ + ':' + repr(self.ro_func) + '>'


class BeamRoFunction:
    '''
    When config.beam is a 'def beam(xi, x, y):' there is really no beam.
    1) The source should be set to BeamRoFunctionFakeSource, yielding nothing
    2) A kludge is hacked into to inject beam_ro values
    3) A fake sink is used so that nonexistent beam particles are not saved
    All in all combined it allows to 'use a beam defined as charge density',
    witout beaing represented by particles at all.
    '''

    @staticmethod
    @hacks.before('lcode.main.choose_beam_source')
    def hijack_choose_beam_source(choose_beam_source, config, t_i=0):
        if BeamRoFunction.looks_like_ro_func(config.beam):
            source = BeamRoFunctionFakeSource(config, config.beam)
            if t_i > 1:
                raise RuntimeWarning(
                    'Beam as a ro function + multiple time steps = why?'
                )
            return hacks.FakeResult(source)

    @staticmethod
    @hacks.before('lcode.main.choose_beam_sink')
    def hijack_choose_beam_sink(choose_beam_sink, config, t_i=0):
        if BeamRoFunction.looks_like_ro_func(config.beam):
            sink = lcode.beam_construction.BeamFakeSink()
            return hacks.FakeResult(sink)

    @staticmethod
    def looks_like_ro_func(func):
        return (inspect.isroutine(func) and
                list(inspect.signature(func).parameters) == ['xi', 'x', 'y'])

    @staticmethod
    @hacks.into('beam_ro_from_function_kludge')
    def beam_ro_from_function_kludge(config, beam_source, xi_i, beam_ro):
        if not isinstance(beam_source, BeamRoFunctionFakeSource):
            return
        xi = -config.xi_step_size * xi_i
        beam_ro[...] = beam_source.ro_func(xi, beam_source.x, beam_source.y)
