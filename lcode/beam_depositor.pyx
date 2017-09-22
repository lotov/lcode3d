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

# cython: language_level=3, unraisable_tracebacks=True, profile=True


'''
Beam depositor for LCODE.
Deposits beam charge density on a grid.
Primary author: A. P. Sosedkin <A.P.Sosedkin@inp.nsk.su>
Secondary author: I. A. Shalimova <ias@osmf.sscc.ru>
'''

import numpy as np
cimport numpy as np

from . import beam_particle
from . cimport beam_particle


cpdef np.ndarray[double, ndim=2] deposit_beam(config,
                                              beam_particle.t[:] particles,
                                              double[:] weights,
                                              ):
    if config.beam_deposit_nearest:
        return deposit_beam_nearest(config, particles, weights)
    else:
        return deposit_beam_smooth(config, particles, weights)


cpdef np.ndarray[double, ndim=2] \
        deposit_beam_smooth(config,
                            beam_particle.t[:] particles,
                            double[:] weights,
                            ):
    cdef unsigned int i, j
    cdef long k
    cdef double x_loc, y_loc
    cdef double fx1, fy1, fx2, fy2, fx3, fy3
    cdef double dro
    cdef beam_particle.t p

    cdef double x_max = config.window_width / 2
    cdef double h = config.window_width / config.grid_steps
    cdef double n_0 = config.base_plasma_density

    # TODO: maybe optimize the allocation
    cdef np.ndarray[double, ndim=2] tmp = np.zeros(
        (config.grid_steps + 2, config.grid_steps + 2)
    )

    cdef double c = 3e10  # cm/s
    cdef double e = 4.8e-10  # statcoulomb
    cdef double m_e = 9.1e-28  # g
    cdef double K = 4 * np.pi * e**3 / (c**3 * m_e) * np.sqrt(4 * np.pi / m_e)
    K *= np.sqrt(n_0)
    K /= config.xi_step_size * (config.window_width / config.grid_steps) ** 2

    # indexed for performance
    for k in range(particles.shape[0]):
        p = particles[k]
        dro = p.q * p.W * K * weights[k]
        # dro = p.q / (1 - p.v[0]) was used for plasma

        # particle indices in ro array
        i = <unsigned int> ((x_max + p.r[1]) / h)
        j = <unsigned int> ((x_max + p.r[2]) / h)
        # assert 0 <= i < n_dim
        # assert 0 <= j < n_dim

        x_loc = x_max + p.r[1] - i * h - .5 * h
        y_loc = x_max + p.r[2] - j * h - .5 * h

        # convert to indices in tmp (n_dim + 2, n_dim + 2)
        i, j = i + 1, j + 1

        fx1 = .75 - x_loc**2 / h**2
        fy1 = .75 - y_loc**2 / h**2
        fx2 = .5 + x_loc / h
        fy2 = .5 + y_loc / h
        fx3 = .5 - x_loc / h
        fy3 = .5 - y_loc / h

        tmp[i + 0, j + 0] += dro * fx1 * fy1
        tmp[i + 1, j + 0] += dro * fx2**2 * fy1 / 2
        tmp[i + 0, j + 1] += dro * fy2**2 * fx1 / 2
        tmp[i + 1, j + 1] += dro * fx2**2 * fy2**2 / 4
        tmp[i - 1, j + 0] += dro * fx3**2 * fy1 / 2
        tmp[i + 0, j - 1] += dro * fy3**2 * fx1 / 2
        tmp[i - 1, j - 1] += dro * fx3**2 * fy3**2 / 4
        tmp[i - 1, j + 1] += dro * fx3**2 * fy2**2 / 4
        tmp[i + 1, j - 1] += dro * fx2**2 * fy3**2 / 4

    # Move the outer edges data that 'fell off' to a closer row/column
    # to simulate symmetrical boundary conditions for the inner cells
    # aka pick_up_tails
    tmp[1, :] += tmp[0, :]
    tmp[:, 1] += tmp[:, 0]
    tmp[-2, :] += tmp[-1, :]
    tmp[:, -2] += tmp[:, -1]

    return tmp[1:-1, 1:-1]


cpdef np.ndarray[double, ndim=2] \
        deposit_beam_nearest(config,
                             beam_particle.t[:] particles,
                             double[:] weights,
                             ):
    cdef unsigned int i, j
    cdef long k
    cdef double dro
    cdef beam_particle.t p

    cdef double x_max = config.window_width / 2
    cdef double h = config.window_width / config.grid_steps

    cdef double c = 3e10  # cm/s
    cdef double e = 4.8e-10  # statcoulomb
    cdef double m_e = 9.1e-28  # g
    cdef double n_0 = 7e14  # 1/cm3
    cdef double K = 4 * np.pi * e**3 / (c**3 * m_e) * np.sqrt(4 * np.pi / m_e)
    K *= np.sqrt(n_0)
    K /= config.xi_step_size * (config.window_width / config.grid_steps) ** 2

    # TODO: maybe optimize the allocation
    cdef np.ndarray[double, ndim=2] ro = np.zeros((config.grid_steps,) * 2)

    # indexed for performance
    for k in range(particles.shape[0]):
        p = particles[k]
        dro = p.q * p.W * K * weights[k]  # CLARIFY!
        # dro = p.q / (1 - p.v[0]) was used for plasma

        i = <unsigned int> ((x_max + p.r[1]) / h)
        j = <unsigned int> ((x_max + p.r[2]) / h)

        ro[i, j] += dro

    return ro
