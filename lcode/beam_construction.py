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


def try_next(it):
    try:
        return next(it)
    except StopIteration:
        return beam_particle.BeamParticleArray([])


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


def layer_slicer_random_access(particle_data, length):
    length = abs(length)
    xi, start_i, i, data_len = 0, 0, 0, len(particle_data)
    while i < data_len:
        if particle_data[i]['r'][0] <= xi - length:
            arr = np.array(particle_data[start_i:i])
            assert np.all(arr['r'][:, 0] <= xi)
            assert np.all(arr['r'][:, 0] > xi - length)
            yield arr
            start_i = i
            xi -= length
        i += 1
    yield np.array(particle_data[start_i:])


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
        self.b = self.f.create_group('beam')
        return self

    def put(self, layer):
        for sp_name in layer:
            try:
                ds = self.b[sp_name]
            except KeyError:
                self.b.create_dataset(sp_name, (0,), maxshape=(None,),
                                      dtype=beam_particle.dtype,
                                      chunks=True)
                ds = self.b[sp_name]
            sp_particles = layer[sp_name]
            l = len(ds)
            ds.resize((l + len(sp_particles),))
            ds[l:] = sp_particles
        self.f.flush()

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
        self.species_names = list(self.beam.keys())
        species_data = [self.beam[sp] for sp in self.species_names]
        self.slicers = [layer_slicer_random_access(d, self.config.xi_step_size)
                        for d in species_data]
        return self

    def __next__(self):
        return {sp_name: try_next(slicer)
                for sp_name, slicer in zip(self.species_names, self.slicers)}

    def __iter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.f.close()

    def __repr__(self):
        return '<' + self.__class__.__name__ + ': ' + self.filename + '>'


class BeamConstructionSource:  # pylint: disable=too-few-public-methods
    def __init__(self, config, particle_generator_function_dict):
        if not isinstance(particle_generator_function_dict, dict):
            particle_generator_function_dict = {
                'particles': particle_generator_function_dict
            }
        self.genfuncs = particle_generator_function_dict
        self.xi_step_size = config.xi_step_size

    def __enter__(self):
        self.gens = {sp_name: genfunc()
                     for sp_name, genfunc in self.genfuncs.items()}
        self.slicers = {sp_name: layer_slicer_generator(gen, self.xi_step_size)
                        for sp_name, gen in self.gens.items()}
        return self

    def __next__(self):
        return {sp_name: try_next(slicer)
                for sp_name, slicer in self.slicers.items()}

    def __iter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __repr__(self):
        return ('<' + self.__class__.__name__ + ': ' +
                repr(self.genfuncs) + '>')


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
    external_state = random.getstate()
    for i in range(1000):
        xi -= 0.001
        x = random.normalvariate(0, sigma=1)
        y = random.normalvariate(0, sigma=1)
        p_x = random.normalvariate(0, sigma=2)
        p_y = random.normalvariate(0, sigma=2)
        p_xi = random.uniform(0, 1e3)
        internal_state = random.getstate()
        random.setstate(external_state)
        yield beam_particle.BeamParticle(m=1, q=-1,
                                         xi=xi, x=x, y=y,
                                         p_xi=p_xi, p_x=p_x, p_y=p_y)
        external_state = random.getstate()
        random.setstate(internal_state)
