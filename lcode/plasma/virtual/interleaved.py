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


# HACK: only allows two interleaved sorts of plasma particles
# i.e. even ions, odd electrons
# HACK: each sort must have same mass and charge


import numpy as np
import scipy.interpolate

from ... import plasma_construction
from ... import plasma_particle


def devirtualize_plasma(real, virt, weights, indices):
    plasma = virt.copy()
    plasma['x'] = devirtualize_values(real['x'], weights, indices)
    plasma['y'] = devirtualize_values(real['y'], weights, indices)
    # plasma['m'] = devirtualize_values(real['m'], weights, indices)
    # plasma['q'] = devirtualize_values(real['q'], weights, indices)
    plasma['v'][:, 0] = devirtualize_values(real['v'][:, 0], weights, indices)
    plasma['v'][:, 1] = devirtualize_values(real['v'][:, 1], weights, indices)
    plasma['v'][:, 2] = devirtualize_values(real['v'][:, 2], weights, indices)
    plasma['p'][:, 0] = plasma['v'][:, 0] * plasma['m']
    plasma['p'][:, 1] = plasma['v'][:, 1] * plasma['m']
    plasma['p'][:, 2] = plasma['v'][:, 2] * plasma['m']
    return plasma


def devirtualize_values(values, weights, indices):
    return np.sum(weights * values[indices], axis=1)


def CoarsePlasma_(window_width, grid_steps, coarseness=3):
    # TODO: return: ions, electrons, virtualizer
    m_i = plasma_particle.USUAL_ION_MASS * coarseness**2
    m_e = plasma_particle.USUAL_ELECTRON_MASS * coarseness**2
    q_i = plasma_particle.USUAL_ION_CHARGE * coarseness**2
    q_e = plasma_particle.USUAL_ELECTRON_CHARGE * coarseness**2
    cell_size = window_width / grid_steps
    half = np.arange(coarseness * cell_size / 2,
                     window_width / 2 - 0 * cell_size,  # !!!
                     cell_size * coarseness)
    xs = ys = np.concatenate([-half[::-1], half])
    n = 0
    for x in xs:
        for y in ys:
            yield plasma_particle.Ion(x=x, y=y, q=q_i, m=m_i, N=n)
            n += 1
            yield plasma_particle.Electron(x=x, y=y, q=q_e, m=m_e, N=n)
            n += 1


def CoarsePlasma(*a, **kwa):
    return plasma_particle.PlasmaParticleArray(CoarsePlasma_(*a, **kwa))


def FinePlasma(window_width, steps, substep):
    return plasma_particle.PlasmaParticleArray(
        plasma_construction.UniformPlasma(window_width, steps, substep)
    )


def virtualize_plasma(coarse, coarse_initial, fine_initial):
    def interp(coarse_component):
        # HACK, TODO: move to closure inside CoarsePlasma or whatever
        Nc = np.sqrt(coarse_initial.shape[0])
        assert Nc % 1 == 0
        Nc = int(Nc)
        coarse_initial_ = coarse_initial.reshape(Nc, Nc)

        Nf = np.sqrt(fine_initial.shape[0])
        assert Nf % 1 == 0
        Nf = int(Nf)

        coarse_initial_ = coarse_initial.reshape(Nc, Nc)
        coarse_component = coarse_component.reshape(Nc, Nc)
        fine_initial_ = fine_initial.reshape(Nf, Nf)
        xs_coarse_initial = coarse_initial_['x'][:, 0]
        ys_coarse_initial = coarse_initial_['y'][0, :]
        xs_fine_initial = fine_initial_['x'][:, 0]
        ys_fine_initial = fine_initial_['y'][0, :]
        f = scipy.interpolate.RectBivariateSpline(xs_coarse_initial,
                                                  ys_coarse_initial,
                                                  coarse_component,
                                                  kx=1, ky=1)
        v = f(xs_fine_initial, ys_fine_initial)
        return v.flatten()  # TODO: 'F'? 'C'? .T?

    fine = fine_initial.copy()

    coarse_x_offsets = coarse['x'] - coarse_initial['x']
    fine_x_offsets = interp(coarse_x_offsets)
    fine['x'] = fine_x_offsets + fine_initial['x']

    fine['y'] = interp(coarse['y'] - coarse_initial['y']) + fine_initial['y']
    fine['p'][:, 0] = (interp(coarse['p'][:, 0] - coarse_initial['p'][:, 0]) +
                       fine_initial['p'][:, 0])
    fine['p'][:, 1] = (interp(coarse['p'][:, 1] - coarse_initial['p'][:, 1]) +
                       fine_initial['p'][:, 1])
    fine['p'][:, 2] = (interp(coarse['p'][:, 2] - coarse_initial['p'][:, 2]) +
                       fine_initial['p'][:, 2])
    fine['v'][:, 0] = (interp(coarse['v'][:, 0] - coarse_initial['v'][:, 0]) +
                       fine_initial['v'][:, 0])
    fine['v'][:, 1] = (interp(coarse['v'][:, 1] - coarse_initial['v'][:, 1]) +
                       fine_initial['v'][:, 1])
    fine['v'][:, 2] = (interp(coarse['v'][:, 2] - coarse_initial['v'][:, 2]) +
                       fine_initial['v'][:, 2])
    return fine


def virtualizer(coarse_, fine_):
    # Say hello to a closure abuse: storing copies in a closure
    coarse_initial = coarse_.copy()
    coarse_initial.setflags(write=False)
    fine_initial = fine_.copy()
    fine_initial.setflags(write=False)

    def virtualize(coarse):
        coarse_ions = coarse[::2]
        coarse_electrons = coarse[1::2]
        coarse_ions_initial = coarse_initial[::2]
        coarse_electrons_initial = coarse_initial[1::2]
        fine_ions_initial = fine_initial[::2]
        fine_electrons_initial = fine_initial[1::2]
        fine = fine_initial.copy()
        fine[::2] = virtualize_plasma(coarse_ions, coarse_ions_initial,
                                      fine_ions_initial)
        fine[1::2] = virtualize_plasma(coarse_electrons,
                                       coarse_electrons_initial,
                                       fine_electrons_initial)
        return fine

    return virtualize


def make(window_width, steps, coarseness=2, fineness=2):
    coarse = CoarsePlasma(window_width, steps, coarseness)
    fine = FinePlasma(window_width, steps, fineness)
    return coarse, virtualizer(coarse, fine)
