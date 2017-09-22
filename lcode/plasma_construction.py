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

from . import plasma_particle


def construct(something, shape=1):
    plasma = plasma_particle.PlasmaParticleArray(something)
    # TODO: plasma macrosity
    plasma['m'] *= shape
    plasma['q'] *= shape
    return plasma


def UniformPlasma(window_width, grid_steps, substep=6):
    m_i = plasma_particle.USUAL_ION_MASS / substep**2
    m_e = plasma_particle.USUAL_ELECTRON_MASS / substep**2
    q_i = plasma_particle.USUAL_ION_CHARGE / substep**2
    q_e = plasma_particle.USUAL_ELECTRON_CHARGE / substep**2
    substeps = grid_steps * substep
    ss = window_width / substeps
    n = 0
    for x in np.arange(-window_width / 2, window_width / 2, ss) + ss / 2:
        for y in np.arange(-window_width / 2, window_width / 2, ss) + ss / 2:
            yield plasma_particle.Ion(x=x, y=y, q=q_i, m=m_i, N=n)
            n += 1
            yield plasma_particle.Electron(x=x, y=y, q=q_e, m=m_e, N=n)
            n += 1


def UniformPlasmaCompatible(window_width, grid_steps, substep=6):
    # Same as UniformPlasma,
    # but the particle order is compatible with the C version
    h = window_width / grid_steps
    hc = h / substep
    m_i = plasma_particle.USUAL_ION_MASS / substep**2
    m_e = plasma_particle.USUAL_ELECTRON_MASS / substep**2
    q_i = plasma_particle.USUAL_ION_CHARGE / substep**2
    q_e = plasma_particle.USUAL_ELECTRON_CHARGE / substep**2
    n = 0
    for j in range(grid_steps):
        for i in range(grid_steps):
            for jj in range(substep):
                for ii in range(substep):
                    x = -window_width / 2 + h * i + hc * (ii + .5)
                    y = -window_width / 2 + h * j + hc * (jj + .5)
                    yield plasma_particle.Ion(x=x, y=y,
                                              q=q_i, m=m_i, N=n)
                    n += 1
                    yield plasma_particle.Electron(x=x, y=y,
                                                   q=q_e, m=m_e, N=n)
                    n += 1


def UniformPlasmaHex(window_width, grid_steps, substep=6):
    m_i = plasma_particle.USUAL_ION_MASS / substep**2
    m_e = plasma_particle.USUAL_ELECTRON_MASS / substep**2
    q_i = plasma_particle.USUAL_ION_CHARGE / substep**2
    q_e = plasma_particle.USUAL_ELECTRON_CHARGE / substep**2
    substeps = grid_steps * substep
    ss = window_width / substeps
    hs = ss * np.sqrt(3) / 2
    n = 0
    odd = False
    l = window_width / 2
    vert = np.arange(0, l - ss * 2, ss)
    vert = np.concatenate((-vert[::-1], vert[1:]))
    hor1 = np.arange(0, l - hs * 2 - hs / 2, hs)
    hor1 = np.concatenate((-hor1[::-1], hor1[1:]))
    hor2 = hor1[:-1] + hs / 2
    for y in vert:
        for x in hor1 if odd else hor2:
            yield plasma_particle.Ion(x=x, y=y, q=q_i, m=m_i, N=n)
            n += 1
            yield plasma_particle.Electron(x=x, y=y, q=q_e, m=m_e, N=n)
            n += 1
        odd = not odd


def UniformPlasmaRhombic(window_width, grid_steps, substep=6):
    m_i = plasma_particle.USUAL_ION_MASS / substep**2
    m_e = plasma_particle.USUAL_ELECTRON_MASS / substep**2
    q_i = plasma_particle.USUAL_ION_CHARGE / substep**2
    q_e = plasma_particle.USUAL_ELECTRON_CHARGE / substep**2
    substeps = grid_steps * substep
    ss = window_width / substeps
    l = window_width / 2
    vert = np.arange(0, l, ss)
    vert = np.concatenate((-vert[::-1], vert[1:]))
    hor1 = np.arange(0, l - ss / 2, ss)
    hor1 = np.concatenate((-hor1[::-1], hor1[1:]))
    hor2 = hor1[:-1] + ss / 2
    n = 0
    odd = False
    for y in vert:
        for x in hor1 if odd else hor2:
            yield plasma_particle.Ion(x=x, y=y, q=q_i, m=m_i, N=n)
            n += 1
            yield plasma_particle.Electron(x=x, y=y, q=q_e, m=m_e, N=n)
            n += 1
            odd = not odd
