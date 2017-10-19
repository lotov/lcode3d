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


import numpy as np
import h5py

from . import beam_particle


def layer_slicer_generator(particle_generator, length):
    length = abs(length)
    layer, xi = [], 0
    for p in particle_generator:
        if p['r'][0] <= xi - length:
            # Next layer particle encountered, time to yield this layer
            yield beam_particle.BeamParticleArray(layer)
            layer, xi = [], xi - length
        layer.append(p)
    yield beam_particle.BeamParticleArray(layer)


# pylint: disable=too-many-arguments
def PreciselyWeighted(window_width, grid_steps, weights, xi=0, m=1, q=1):
    if np.all(weights == 0):
        raise StopIteration
    # TODO: use gamma, convert to p_z/p_xi
    ss = window_width / grid_steps
    n = 0
    xs = ys = np.arange(-window_width / 2, window_width / 2, ss) + ss / 2
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            w = weights[i, j]
            if not w:
                continue
            yield beam_particle.BeamParticle(m=m, q=q, W=(w / q), N=n,
                                             xi=xi, x=x, y=y,
                                             p_xi=0, p_x=0, p_y=0)
            n += 1


class BeamFileSink:  # pylint: disable=too-few-public-methods
    def __init__(self, config, filename='out.h5'):
        self.filename = filename
        self.config = config

    def __enter__(self):
        self.f = h5py.File(self.filename, 'w')
        self.beam = self.f.create_dataset('beam', (0,), maxshape=(None,),
                                          dtype=beam_particle.dtype,
                                          chunks=True)
        return self

    def put(self, layer):
        l = len(self.beam)
        self.beam.resize((l + len(layer),))
        self.beam[l:] = layer

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.f.close()

    def __repr__(self):
        return '<' + self.__class__.__name__ + ': ' + self.filename + '>'


class BeamFileSource:  # pylint: disable=too-few-public-methods
    def __init__(self, config, filename='in.h5'):
        self.config = config
        self.filename = filename

    def __enter__(self):
        self.f = h5py.File(self.filename, 'r')
        self.beam = self.f['beam']
        self.xis = self.beam['r'][:, 0]
        self.xi_i = 0
        self.start_i = 0
        return self

    def __next__(self):
        # TODO: needs a speedup
        i = self.start_i
        xi_stop = (-self.xi_i - 1) * self.config.xi_step_size
        while i < len(self.beam) and self.xis[i] > xi_stop:
            i += 1
        arr = np.array(self.beam[self.start_i:i])
        assert np.all(arr['r'][:, 0] > xi_stop)
        assert np.all(arr['r'][:, 0] <= xi_stop + self.config.xi_step_size)
        self.start_i = i
        self.xi_i += 1
        return arr

    def __iter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.f.close()

    def __repr__(self):
        return '<' + self.__class__.__name__ + ': ' + self.filename + '>'


class BeamConstructionSource:  # pylint: disable=too-few-public-methods
    def __init__(self, config, particle_generator_function):
        self.particle_generator_function = particle_generator_function
        self.xi_step_size = config.xi_step_size

    def __enter__(self):
        self.gen = layer_slicer_generator(self.particle_generator_function(),
                                          self.xi_step_size)
        return self

    def __next__(self):
        return next(self.gen)

    def __iter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __repr__(self):
        return ('<' + self.__class__.__name__ + ': ' +
                repr(self.particle_generator_function) + '>')


class BeamFakeSink:  # pylint: disable=too-few-public-methods
    def __enter__(self):
        return self

    def put(self, layer):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __repr__(self):
        return '<' + self.__class__.__name__ + '>'


def some_particle_beam():
    import random
    from . import beam_particle
    xi = 0
    random.seed(0)
    for i in range(1000):
        xi -= 0.001
        x = random.normalvariate(0, sigma=1)
        y = random.normalvariate(0, sigma=1)
        p_x = random.normalvariate(0, sigma=2)
        p_y = random.normalvariate(0, sigma=2)
        yield beam_particle.BeamParticle(m=1, q=-1,
                                         xi=xi, x=x, y=y,
                                         p_xi=32, p_x=p_x, p_y=p_y)
