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
3D plasma solver for LCODE.
Primary author: I. A. Shalimova <ias@osmf.sscc.ru>
Secondary author: A. P. Sosedkin <A.P.Sosedkin@inp.nsk.su>
'''


from libc.math cimport sin, cos, sqrt, fabs
from libc.math cimport exp  # ro_b_func only
from libc.math cimport M_PI as pi  # 3.141592653589793 on my machine

from cython.parallel import prange
cimport openmp

import numpy as np
cimport numpy as np
import scipy.ndimage

from . import plasma_particle
from . cimport plasma_particle
from . import plasma_construction

# STYLE TODO: replace .5 * x with x / 2 etc.

# Most of the functions should be declared nogil, but they aren't.
# That's done as a temporary measure to prevent inlining and aid debugging.
# Use a preprocessing trick?


# Contains both scalar charge density ro and vector current j
cdef packed struct RoJ_t:
    double ro
    double jz
    double jx
    double jy

RoJ_dtype = np.dtype([
    ('ro', np.double),
    ('jz', np.double),
    ('jx', np.double),
    ('jy', np.double),
], align=False)


cdef class PlasmaSolverConfig:
    """Wraps relevant config values in a classy C struct for faster access."""
    cdef public unsigned int npq, n_dim
    cdef public unsigned long Lq
    cdef public unsigned int iter_p, n_corr, n_corr_transverse
    cdef public int interpolation_order
    cdef public unsigned int print_every
    cdef public unsigned int limit_threads
    cdef public double x_max, h, h3, eps, B_0
    cdef public unsigned int reuse_EB, use_average_speed, zero_edges
    cdef public double boundary_suppression

    cdef public object virtualize

    def __init__(self, global_config):
        from math import log
        npq, unwanted = divmod(log(global_config.grid_steps - 1, 2), 1)
        if unwanted:
            raise RuntimeError('Grid step must be N**2 + 1')
        self.npq = npq

        self.n_dim = 2**self.npq + 1
        self.eps = global_config.plasma_solver_eps
        self.x_max = global_config.window_width / 2
        self.h = global_config.window_width / self.n_dim
        self.h3 = global_config.xi_step_size
        self.B_0 = global_config.plasma_solver_B_0
        self.n_corr = global_config.plasma_solver_corrector_passes
        self.n_corr_transverse = (
            global_config.plasma_solver_corrector_transverse_passes
        )
        self.iter_p = global_config.plasma_solver_particle_mover_corrector
        self.Lq = global_config.xi_steps
        self.print_every = global_config.print_every_xi_steps
        self.interpolation_order = (
            global_config.plasma_solver_fields_interpolation_order
        )
        self.limit_threads = global_config.openmp_limit_threads
        self.reuse_EB = global_config.plasma_solver_reuse_EB
        self.use_average_speed = (
            global_config.plasma_solver_use_average_speed
        )
        self.boundary_suppression = (
            global_config.plasma_solver_boundary_suppression
        )

        self.virtualize = global_config.virtualize
        self.zero_edges = global_config.plasma_solver_zero_edges


cdef void ro_and_j_ie_Vshivkov(PlasmaSolverConfig config,
                               plasma_particle.t[:] plasma_particles,
                               plasma_particle.t[:] plasma_particles_cor,
                               # n_dim, n_dim
                               np.ndarray[RoJ_t, ndim=2] roj,
                               # 4, 2, n_dim
                               np.ndarray[RoJ_t, ndim=3] roj_edges,
                               ):
    cdef unsigned int i, j
    cdef int q
    cdef long k
    cdef double x_loc, y_loc
    cdef double fx1, fy1, fx2, fy2, fx3, fy3
    cdef double dro, djx, djy, djz
    cdef plasma_particle.t p, p_cor

    cdef unsigned int n_dim = roj.shape[0]
    assert roj.shape[0] == roj.shape[1] == roj_edges.shape[2]
    assert plasma_particles.shape[0] == plasma_particles_cor.shape[0]

    roj[...] = 0
    roj_edges[...] = 0

    # for p in plasma_particles: indexed for performance
    for k in range(plasma_particles.shape[0]):
        p = plasma_particles[k]  # half-step behind
        # dro = p.q / (1 - p.v[0])
        # djz = dro * p.v[0]
        # djx = dro * p.v[1]
        # djy = dro * p.v[2]
        p_cor = plasma_particles_cor[k]  # half-step ahead
        dro = p.q / (1 - (p_cor.v[0] + p.v[0]) / 2)
        djz = dro * (p_cor.v[0] + p.v[0]) / 2
        djx = dro * (p_cor.v[1] + p.v[1]) / 2
        djy = dro * (p_cor.v[2] + p.v[2]) / 2

        # particle indices in roj and adge arrays
        i = <unsigned int> ((config.x_max + p.x) / config.h)
        j = <unsigned int> ((config.x_max + p.y) / config.h)
        # assert 0 <= i
        # assert i < n_dim
        # assert 0 <= j
        # assert j < n_dim

        # For particles from outer cells:
        # TODO:  What's in with the corners?
        # They seem to only be deposited to one, not two edge arrays
        # Maybe 'continue's below should be dropped?
        if i == 0:
            q = 0 if config.x_max + p.x < config.h * .5 else 1
            roj_edges[0, q, j].ro += dro
            roj_edges[0, q, j].jx += djx
            roj_edges[0, q, j].jy += djy
            roj_edges[0, q, j].jz += djz
            # We can't deposit it to roj right here
            # because of the tail picking
            # roj[0, j].ro += dro
            # But we'll do it later,
            # see pick_up_tails and delayed_edges_merging
            # roj[0, j].jx += djx  # same
            # roj[0, j].jy += djx  # story
            # roj[0, j].jz += djz  # here
            continue

        # ****************
        if i == n_dim - 1:
            q = 0 if p.x < config.x_max - config.h * .5 else 1
            # roj[-1, j].ro += dro  # -||-, and same for current

            # WARNING: mako templating magic
            # This code
            % for component in 'ro', 'jz', 'jx', 'jy':
            roj_edges[1, q, j].${component} += d${component}
            % endfor
            # gets expanded to:
            # roj_edges[0, q, j].ro += dro
            # roj_edges[0, q, j].jz += djz
            # roj_edges[0, q, j].jx += djx
            # roj_edges[0, q, j].jy += djy

            continue

        # #c******************
        if j == 0:
            q = 0 if config.x_max + p.y < config.h * .5 else 1
            # roj[i, 0].ro += dro  # -||-, and same for current
            % for component in 'ro', 'jz', 'jx', 'jy':
            roj_edges[2, q, i].${component} += d${component}
            % endfor
            continue

        # c******************
        if j == n_dim - 1:
            q = 0 if p.y < config.x_max - config.h * .5 else 1
            # roj[i, -1].ro += dro  # -||-, and same for current
            % for component in 'ro', 'jz', 'jx', 'jy':
            roj_edges[3, q, i].${component} += d${component}
            % endfor
            continue

        # Only for particles from inner cells
        # assert 0 < i < n_dim - 1 and 0 < j < n_dim - 1
        # Their density goes directly to roj['ro']

        x_loc = config.x_max + p.x - i * config.h - .5 * config.h
        y_loc = config.x_max + p.y - j * config.h - .5 * config.h
        fx1 = .75 - x_loc**2 / config.h**2
        fy1 = .75 - y_loc**2 / config.h**2
        fx2 = .5 + x_loc / config.h
        fy2 = .5 + y_loc / config.h
        fx3 = .5 - x_loc / config.h
        fy3 = .5 - y_loc / config.h

        % for comp in 'ro', 'jz', 'jx', 'jy':
        roj[i + 0, j + 0].${comp} += d${comp} * fx1 * fy1
        roj[i + 1, j + 0].${comp} += d${comp} * fx2**2 * fy1 / 2
        roj[i + 0, j + 1].${comp} += d${comp} * fy2**2 * fx1 / 2
        roj[i + 1, j + 1].${comp} += d${comp} * fx2**2 * fy2**2 / 4
        roj[i - 1, j + 0].${comp} += d${comp} * fx3**2 * fy1 / 2
        roj[i + 0, j - 1].${comp} += d${comp} * fy3**2 * fx1 / 2
        roj[i - 1, j - 1].${comp} += d${comp} * fx3**2 * fy3**2 / 4
        roj[i - 1, j + 1].${comp} += d${comp} * fx3**2 * fy2**2 / 4
        roj[i + 1, j - 1].${comp} += d${comp} * fx2**2 * fy3**2 / 4
        % endfor

    # 'Pick up the tails' and write proper edge values for density
    pick_up_tails(roj['ro'])
    delayed_edges_merging(roj['ro'], roj_edges['ro'])

    # Same for currents
    pick_up_tails(roj['jx'])
    pick_up_tails(roj['jy'])
    pick_up_tails(roj['jz'])
    delayed_edges_merging(roj['jx'], roj_edges['jx'])
    delayed_edges_merging(roj['jy'], roj_edges['jy'])
    delayed_edges_merging(roj['jz'], roj_edges['jz'])


cpdef pick_up_tails(arr):
    # Move the outer edges data to closes row
    # to simulate symmetrical boundary conditions for the inner cells
    arr[1, :] += arr[0, :]
    arr[:, 1] += arr[:, 0]
    arr[-2, :] += arr[-1, :]
    arr[:, -2] += arr[:, -1]
    arr[0, :] = arr[:, 0] = arr[-1, :] = arr[:, -1] = 0


cdef delayed_edges_merging(np.ndarray[double, ndim=2] arr,  # n_dim, n_dim
                           np.ndarray[double, ndim=3] arr_edges,  # 4, 2, n_dim
                           ):
    # Previously we didn't do
    # roj[0, j].ro += dro
    # so that we don't confuse it with the Vshivkov's tails.
    # But we did `roj_edges[0, q, j].ro += dro`
    # The tails are picked now, time to deposit roj_edges['ro'] back!
    arr[0, :] += arr_edges[0, 0, :]  # Copy all roj_edges[0, 0, j] to ro[0, j]
    arr[0, :] += arr_edges[0, 1, :]  # -||- for roj_edges[0, 1, j],
    # Separation into two inplace additions aims to avoid extraneous allocation
    arr[-1, :] += arr_edges[1, 0, :]
    arr[-1, :] += arr_edges[1, 1, :]
    arr[:, 0] += arr_edges[2, 0, :]
    arr[:, 0] += arr_edges[2, 1, :]
    arr[:, -1] += arr_edges[3, 0, :]
    arr[:, -1] += arr_edges[3, 1, :]


cdef void ro_and_j_ie_cor_Vshivkov(PlasmaSolverConfig config,
                                   plasma_particle.t[:] plasma_particles_cor,
                                   # n_dim, n_dim
                                   np.ndarray[RoJ_t, ndim=2] roj_cor,
                                   # n_dim + 2, n_dim + 2
                                   np.ndarray[RoJ_t, ndim=2] tmp,
                                   ):
    cdef unsigned int i, j
    cdef long k
    cdef double x_loc, y_loc
    cdef double fx1, fy1, fx2, fy2, fx3, fy3
    cdef double dro, djx, djy, djz
    cdef plasma_particle.t p

    roj_cor[...] = 0
    tmp[...] = 0

    # indexed for performance
    for k in range(plasma_particles_cor.shape[0]):
        p = plasma_particles_cor[k]
        dro = p.q / (1 - p.v[0])
        djz = dro * p.v[0]
        djx = dro * p.v[1]
        djy = dro * p.v[2]

        # particle indices in roj and adge arrays
        i = <unsigned int> ((config.x_max + p.x) / config.h)
        j = <unsigned int> ((config.x_max + p.y) / config.h)
        # assert 0 <= i < n_dim
        # assert 0 <= j < n_dim

        x_loc = config.x_max + p.x - i * config.h - .5 * config.h
        y_loc = config.x_max + p.y - j * config.h - .5 * config.h

        # convert to indices in tmp (n_dim + 2, n_dim + 2)
        i, j = i + 1, j + 1

        fx1 = .75 - x_loc**2 / config.h**2
        fy1 = .75 - y_loc**2 / config.h**2
        fx2 = .5 + x_loc / config.h
        fy2 = .5 + y_loc / config.h
        fx3 = .5 - x_loc / config.h
        fy3 = .5 - y_loc / config.h

        % for comp in 'ro', 'jz', 'jx', 'jy':
        tmp[i + 0, j + 0].${comp} += d${comp} * fx1 * fy1
        tmp[i + 1, j + 0].${comp} += d${comp} * fx2**2 * fy1 / 2
        tmp[i + 0, j + 1].${comp} += d${comp} * fy2**2 * fx1 / 2
        tmp[i + 1, j + 1].${comp} += d${comp} * fx2**2 * fy2**2 / 4
        tmp[i - 1, j + 0].${comp} += d${comp} * fx3**2 * fy1 / 2
        tmp[i + 0, j - 1].${comp} += d${comp} * fy3**2 * fx1 / 2
        tmp[i - 1, j - 1].${comp} += d${comp} * fx3**2 * fy3**2 / 4
        tmp[i - 1, j + 1].${comp} += d${comp} * fx3**2 * fy2**2 / 4
        tmp[i + 1, j - 1].${comp} += d${comp} * fx2**2 * fy3**2 / 4
        % endfor

    pick_up_tails(tmp['ro'])
    pick_up_tails(tmp['jx'])
    pick_up_tails(tmp['jy'])
    pick_up_tails(tmp['jz'])
    roj_cor[...] = tmp[1:-1, 1:-1]


cpdef void interpolate_fields_sl(PlasmaSolverConfig config,
                                 np.ndarray[double] xs_,
                                 np.ndarray[double] ys_,
                                 np.ndarray[double, ndim=2] Ex,
                                 np.ndarray[double, ndim=2] Ey,
                                 np.ndarray[double, ndim=2] Ez,
                                 np.ndarray[double, ndim=2] Bx,
                                 np.ndarray[double, ndim=2] By,
                                 np.ndarray[double, ndim=2] Bz,
                                 np.ndarray[double] Exs,
                                 np.ndarray[double] Eys,
                                 np.ndarray[double] Ezs,
                                 np.ndarray[double] Bxs,
                                 np.ndarray[double] Bys,
                                 np.ndarray[double] Bzs):
    """
    Calculates fields at particle positions.
    This one does interpolation with scipy.ndimage.map_coordinates.
    Higher spline orders are available with this function.
    See also interpolate_fields_fs.
    """
    cdef np.ndarray[double] xs = xs_.copy()
    cdef np.ndarray[double] ys = ys_.copy()
    xs += config.x_max
    xs *= config.n_dim / (config.x_max * 2)
    xs -= .5
    ys += config.x_max
    ys *= config.n_dim / (config.x_max * 2)
    ys -= .5

    % for f_arr in 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz':
    scipy.ndimage.map_coordinates(${f_arr},
                                  (xs, ys),
                                  order=config.interpolation_order,
                                  output=${f_arr}s)
    % endfor


cpdef void interpolate_fields_fs(PlasmaSolverConfig config,
                                 np.ndarray[double] xs,
                                 np.ndarray[double] ys,
                                 np.ndarray[double, ndim=2] Ex,
                                 np.ndarray[double, ndim=2] Ey,
                                 np.ndarray[double, ndim=2] Ez,
                                 np.ndarray[double, ndim=2] Bx,
                                 np.ndarray[double, ndim=2] By,
                                 np.ndarray[double, ndim=2] Bz,
                                 np.ndarray[double] Exs,
                                 np.ndarray[double] Eys,
                                 np.ndarray[double] Ezs,
                                 np.ndarray[double] Bxs,
                                 np.ndarray[double] Bys,
                                 np.ndarray[double] Bzs):
    """
    Calculates fields at particle positions.
    This one does linear interpolation manually, but fast.
    Higher spline orders are not available with this function.
    See also interpolate_fields_sl.
    """
    cdef long k
    cdef unsigned int i, j
    cdef double dxqk, dyqk

    cdef double h2_reciprocal = 1 / config.h**2

    # indexed for performance
    for k in prange(xs.shape[0], nogil=True, num_threads=config.limit_threads):
        i = <unsigned int> ((config.x_max + xs[k] + config.h * .5) / config.h)
        j = <unsigned int> ((config.x_max + ys[k] + config.h * .5) / config.h)
        # assert 0 <= i < n_dim
        # assert 0 <= j < n_dim

        dxqk = config.x_max + xs[k] - config.h * (i - .5)
        dyqk = config.x_max + ys[k] - config.h * (j - .5)

        if i == 0:
            if j == 0:
                % for Fl in 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz':
                ${Fl}s[k] = ${Fl}[0, 0]
                % endfor
                continue

            if j == Ex.shape[1]:
                % for Fl in 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz':
                ${Fl}s[k] = ${Fl}[0, -1]
                % endfor
                continue

            % for Fl in 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz':
            ${Fl}s[k] = (${Fl}[0, j - 1] +
                         (${Fl}[0, j] - ${Fl}[0, j - 1]) * dyqk / config.h)
            % endfor
            continue

        if i == Ex.shape[0]:
            if j == 0:
                % for Fl in 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz':
                ${Fl}s[k] = ${Fl}[-1, 0]
                % endfor
                continue

            if j == Ex.shape[1]:
                % for Fl in 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz':
                ${Fl}s[k] = ${Fl}[-1, -1]
                % endfor
                continue

            % for Fl in 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz':
            ${Fl}s[k] = (${Fl}[-1, j - 1] +
                         (${Fl}[-1, j] - ${Fl}[-1, j - 1]) * dyqk /
                         config.h)
            % endfor
            continue

        if j == 0:
            % for Fl in 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz':
            ${Fl}s[k] = (${Fl}[i - 1, 0] +
                         (${Fl}[i, 0] - ${Fl}[i - 1, 0]) * dxqk / config.h)
            % endfor
            continue

        if j == Ex.shape[1]:
            % for Fl in 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz':
            ${Fl}s[k] = (${Fl}[i - 1, -1] +
                         (${Fl}[i, -1] - ${Fl}[i - 1, -1]) * dxqk /
                         config.h)
            % endfor
            continue

        % for Fl in 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz':
        ${Fl}s[k] = h2_reciprocal * (  # noqa: E201
            ${Fl}[i + 0, j + 0] * dxqk * dyqk +  # noqa: E222
            ${Fl}[i - 1, j + 0] * (config.h - dxqk) * dyqk +  # noqa: E222
            ${Fl}[i + 0, j - 1] * dxqk * (config.h - dyqk) +  # noqa: E222
            ${Fl}[i - 1, j - 1] * (config.h - dxqk) * (config.h - dyqk)
        )
        % endfor
        continue


cpdef void interpolate_fields_fs9(PlasmaSolverConfig config,
                                  np.ndarray[double] xs,
                                  np.ndarray[double] ys,
                                  np.ndarray[double, ndim=2] Ex,
                                  np.ndarray[double, ndim=2] Ey,
                                  np.ndarray[double, ndim=2] Ez,
                                  np.ndarray[double, ndim=2] Bx,
                                  np.ndarray[double, ndim=2] By,
                                  np.ndarray[double, ndim=2] Bz,
                                  np.ndarray[double] Exs,
                                  np.ndarray[double] Eys,
                                  np.ndarray[double] Ezs,
                                  np.ndarray[double] Bxs,
                                  np.ndarray[double] Bys,
                                  np.ndarray[double] Bzs):
    """
    Calculates fields at particle positions.
    TODO: describe
    """
    cdef long k
    cdef unsigned int i, j, i1, j1
    cdef double dxqk, dyqk
    cdef double x_loc, y_loc
    cdef double fx1, fy1, fx2, fy2, fx3, fy3

    cdef double h2_reciprocal = 1 / config.h**2

    assert Ex.shape[0] == Ex.shape[1] == config.n_dim

    # indexed for performance
    for k in prange(xs.shape[0], nogil=True, num_threads=config.limit_threads):
        i = <unsigned int> ((config.x_max + xs[k] + config.h * .5) / config.h)
        j = <unsigned int> ((config.x_max + ys[k] + config.h * .5) / config.h)
        i1 = <unsigned int> ((config.x_max + xs[k]) / config.h)
        j1 = <unsigned int> ((config.x_max + ys[k]) / config.h)

        dxqk = config.x_max + xs[k] - config.h * (i - .5)
        dyqk = config.x_max + ys[k] - config.h * (j - .5)

        if i == 0:
            if j == 0:
                % for Fl in 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz':
                ${Fl}s[k] = ${Fl}[0, 0]
                % endfor
                continue

            if j == Ex.shape[1]:
                % for Fl in 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz':
                ${Fl}s[k] = ${Fl}[0, -1]
                % endfor
                continue

            % for Fl in 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz':
            ${Fl}s[k] = (${Fl}[0, j - 1] +
                         (${Fl}[0, j] - ${Fl}[0, j - 1]) * dyqk / config.h)
            % endfor
            continue

        if i == Ex.shape[0]:
            if j == 0:
                % for Fl in 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz':
                ${Fl}s[k] = ${Fl}[-1, 0]
                % endfor
                continue

            if j == Ex.shape[1]:
                % for Fl in 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz':
                ${Fl}s[k] = ${Fl}[-1, -1]
                % endfor
                continue

            % for Fl in 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz':
            ${Fl}s[k] = (${Fl}[-1, j - 1] +
                         (${Fl}[-1, j] - ${Fl}[-1, j - 1]) * dyqk /
                         config.h)
            % endfor
            continue

        if j == 0:
            % for Fl in 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz':
            ${Fl}s[k] = (${Fl}[i - 1, 0] +
                         (${Fl}[i, 0] - ${Fl}[i - 1, 0]) * dxqk / config.h)
            % endfor
            continue

        if j == Ex.shape[1]:
            % for Fl in 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz':
            ${Fl}s[k] = (${Fl}[i - 1, -1] +
                         (${Fl}[i, -1] - ${Fl}[i - 1, -1]) * dxqk /
                         config.h)
            % endfor
            continue

        if ((i1 == 0 or i1 == Ex.shape[0] - 1 or
             j1 == 0 or j1 == Ex.shape[1] - 1)):
            % for Fl in 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz':
            ${Fl}s[k] = h2_reciprocal * (  # noqa: E201
                ${Fl}[i + 0, j + 0] * dxqk * dyqk +  # noqa: E222
                ${Fl}[i - 1, j + 0] * (config.h - dxqk) * dyqk +  # noqa: E222
                ${Fl}[i + 0, j - 1] * dxqk * (config.h - dyqk) +  # noqa: E222
                ${Fl}[i - 1, j - 1] * (config.h - dxqk) * (config.h - dyqk)
            )
            % endfor
            continue

        x_loc = config.x_max + xs[k] - i1 * config.h - .5 * config.h
        y_loc = config.x_max + ys[k] - j1 * config.h - .5 * config.h

        fx1 = .75 - x_loc**2 / config.h**2
        fy1 = .75 - y_loc**2 / config.h**2
        fx2 = .5 + x_loc / config.h
        fy2 = .5 + y_loc / config.h
        fx3 = .5 - x_loc / config.h
        fy3 = .5 - y_loc / config.h

        % for Fl in 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz':
        ${Fl}s[k] = (  # noqa: E201
            ${Fl}[i1 + 0, j1 + 0] * fx1 * fy1 +            # noqa: E222
            ${Fl}[i1 + 1, j1 + 0] * fx2**2 * fy1 / 2 +     # noqa: E222
            ${Fl}[i1 + 0, j1 + 1] * fy2**2 * fx1 / 2 +     # noqa: E222
            ${Fl}[i1 + 1, j1 + 1] * fx2**2 * fy2**2 / 4 +  # noqa: E222
            ${Fl}[i1 - 1, j1 + 0] * fx3**2 * fy1 / 2 +  +  # noqa: E222
            ${Fl}[i1 + 0, j1 - 1] * fy3**2 * fx1 / 2 +  +  # noqa: E222
            ${Fl}[i1 - 1, j1 - 1] * fx3**2 * fy3**2 / 4 +  # noqa: E222
            ${Fl}[i1 - 1, j1 + 1] * fx3**2 * fy2**2 / 4 +  # noqa: E222
            ${Fl}[i1 + 1, j1 - 1] * fx2**2 * fy3**2 / 4
        )
        % endfor


cdef void move_particles(PlasmaSolverConfig config,
                         np.ndarray[plasma_particle.t] plasma_particles,
                         np.ndarray[plasma_particle.t] plasma_particles_cor,
                         double[:] Exs,  # same as plasma_particles
                         double[:] Eys,  # -||-
                         double[:] Ezs,  # -||-
                         double[:] Bxs,  # -||-
                         double[:] Bys,  # -||-
                         double[:] Bzs,  # -||-
                         double cor_weight=.5,
                         ):
    cdef long k
    cdef unsigned int itr
    cdef double Ex_k, Ey_k, Ez_k, Bx_k, By_k, Bz_k  # TODO: direct access
    cdef double px_0, py_0, pz_0, mod_p, vx_k, vy_k, vz_k, px_1, py_1, pz_1

    cdef double coeff, h3_1mvz_k
    cdef plasma_particle.t p

    # indexed for performance
    for k in prange(plasma_particles.shape[0],
                    nogil=True, num_threads=config.limit_threads):
        p = plasma_particles[k]

        Ex_k, Ey_k, Ez_k = Exs[k], Eys[k], Ezs[k]
        Bx_k, By_k, Bz_k = Bxs[k], Bys[k], Bzs[k]

        # Calculate the new momentum using iter_p iterations:
        px_0 = p.p[1]
        py_0 = p.p[2]
        pz_0 = p.p[0]
        for itr in range(config.iter_p):
            # vx_k -- at whole step
            # p.v[1] -- at half-step (before)
            # p.p[1] -- at half-step (after)
            # px_0 -- at half-step (before)
            # px_1 -- at half-step (after)
            vx_k = p.v[1] * (1 - cor_weight) + (
                plasma_particles_cor[k].v[1] * cor_weight  # same arrays
            )
            vy_k = p.v[2] * (1 - cor_weight) + (
                plasma_particles_cor[k].v[2] * cor_weight  # if not using
            )
            vz_k = p.v[0] * (1 - cor_weight) + (
                plasma_particles_cor[k].v[0] * cor_weight  # corrector
            )

            h3_1mvz_k = config.h3 / (1 - vz_k) * -p.q
            px_1 = px_0 - h3_1mvz_k * (Ex_k + vy_k * Bz_k - vz_k * By_k)
            py_1 = py_0 - h3_1mvz_k * (Ey_k - vx_k * Bz_k + vz_k * Bx_k)
            pz_1 = pz_0 - h3_1mvz_k * (Ez_k + vx_k * By_k - vy_k * Bx_k)
            mod_p = px_1**2 + py_1**2 + pz_1**2
            coeff = 1 / sqrt(p.m**2 + mod_p)
            vx_k = (p.v[1] + px_1 * coeff) / 2
            vy_k = (p.v[2] + py_1 * coeff) / 2
            vz_k = (p.v[0] + pz_1 * coeff) / 2

            # TODO: maybe we need to recalculate h3_1mvz_k?
            p.p[1] = px_0 - (Ex_k + vy_k * Bz_k - vz_k * By_k) * h3_1mvz_k
            p.p[2] = py_0 - (Ey_k - vx_k * Bz_k + vz_k * Bx_k) * h3_1mvz_k
            p.p[0] = pz_0 - (Ez_k + vx_k * By_k - vy_k * Bx_k) * h3_1mvz_k

            mod_p = p.p[1]**2 + p.p[2]**2 + p.p[0]**2
            coeff = 1 / sqrt(p.m**2 + mod_p)  # TODO: name
            p.v[1] = p.p[1] * coeff
            p.v[2] = p.p[2] * coeff
            p.v[0] = p.p[0] * coeff
            if (fabs(px_1 - p.p[1]) < config.eps and
                    fabs(py_1 - p.p[2]) < config.eps and
                    fabs(pz_1 - p.p[0]) < config.eps):
                break  # goto st10

        # Finally move the particle using new momentum
        p.x += config.h3 * p.v[1] / (1 - p.v[0])
        if p.x > config.x_max:
            p.x = 2 * config.x_max - p.x
            p.v[1] = -p.v[1]
        if p.x < -config.x_max:
            p.x = -2 * config.x_max - p.x
            p.v[1] = -p.v[1]

        p.y += config.h3 * p.v[2] / (1 - p.v[0])
        if p.y > config.x_max:
            p.y = 2 * config.x_max - p.y
            p.v[2] = -p.v[2]
        if p.y < -config.x_max:
            p.y = -2 * config.x_max - p.y
            p.v[2] = -p.v[2]

        # assert -config.x_max < p.x < config.x_max
        # assert -config.x_max < p.y < config.x_max

        plasma_particles[k] = p


cdef np.ndarray[double, ndim=2] pader_x(PlasmaSolverConfig config,
                                        double[:, :] v,  # n_dim, n_dim
                                        double[:, :, :] v_edges,  # 4, 2, n_dim
                                        ):
    cdef unsigned int i, j
    cdef np.ndarray[double, ndim=2] d = np.empty_like(v)

    for j in range(v.shape[0]):
        for i in range(1, v.shape[1] - 1):  # NOTE: 1
            d[i, j] = (v[i + 1, j] - v[i - 1, j]) / config.h / 2
        # 2 * (l2 - l1) / h
        d[0, j] = 2 * (v_edges[0, 1, j] - v_edges[0, 0, j]) / config.h
        # 2 * (r2 - r1) / h
        d[-1, j] = 2 * (v_edges[1, 1, j] - v_edges[1, 0, j]) / config.h

    return d


cdef np.ndarray[double, ndim=2] pader_y(PlasmaSolverConfig config,
                                        double[:, :] v,  # ndim, ndim
                                        double[:, :, :] v_edges,  # 4, 2, n_dim
                                        ):
    cdef unsigned int i, j
    cdef np.ndarray[double, ndim=2] d = np.empty_like(v)

    for i in range(v.shape[0]):
        for j in range(1, v.shape[1] - 1):  # NOTE: 1
            d[i, j] = (v[i, j + 1] - v[i, j - 1]) / config.h / 2
        # 2 * (b2 - b1) / h
        d[i, 0] = 2 * (v_edges[2, 1, i] - v_edges[2, 0, i]) / config.h
        # 2 * (u2 - u1) / h
        d[i, -1] = 2 * (v_edges[3, 1, i] - v_edges[3, 0, i]) / config.h

    return d


cdef class ProgonkaTmp:
    # two very popular temporary arrays
    cdef double[:] alf  # n_dim
    cdef double[:] bet  # n_dim + 1
    # temporary arrays for Posson_reduct_12 and reduction_Dirichlet1
    cdef double[:] RedFi  # n_dim
    cdef double[:] Svl  # n_dim
    cdef double[:] PrF  # n_dim
    cdef double[:] PrV  # n_dim
    cdef double[:] Psi  # n_dim

    def __init__(self, config):
        self.alf = np.zeros(config.n_dim)
        self.bet = np.zeros(config.n_dim + 1)
        self.RedFi = np.zeros(config.n_dim)
        self.Svl = np.zeros(config.n_dim)
        self.PrF = np.zeros(config.n_dim)
        self.PrV = np.zeros(config.n_dim)
        self.Psi = np.zeros(config.n_dim)


cdef void Progonka(double aa,
                   double[:] ff,  # n_dim
                   double[:] vv,  # n_dim
                   ProgonkaTmp tmp):
    cdef double qkapa, qmu1, qmu2
    cdef unsigned int i
    tmp.alf[0] = tmp.bet[0] = 0

    qkapa = 2 / aa
    qmu1 = ff[0] / aa
    qmu2 = ff[-1] / aa
    tmp.alf[1] = qkapa
    tmp.bet[1] = qmu1
    for i in range(1, ff.shape[0] - 2 + 1):
        tmp.alf[i + 1] = 1 / (aa - tmp.alf[i])
        tmp.bet[i + 1] = (ff[i] + tmp.bet[i]) * tmp.alf[i + 1]

    tmp.bet[-1] = (qmu2 + qkapa * tmp.bet[-2]) / (1 - qkapa * tmp.alf[-2])
    vv[-1] = tmp.bet[-1]
    for i in range(vv.shape[0] - 2, 0 - 1, -1):
        vv[i] = tmp.alf[i + 1] * vv[i + 1] + tmp.bet[i + 1]


cpdef void Progonka_Dirichlet(double aa,
                              double[:] ff,  # n_dim
                              double[:] vv,  # n_dim
                              ProgonkaTmp tmp):
    cdef unsigned int i
    tmp.alf[0] = tmp.alf[1] = tmp.bet[0] = tmp.bet[1] = 0

    assert(ff.shape[0] == tmp.alf.shape[0])
    assert(vv.shape[0] == tmp.alf.shape[0])

    for i in range(1, ff.shape[0] - 2 + 1):
        tmp.alf[i + 1] = 1 / (aa - tmp.alf[i])
        tmp.bet[i + 1] = (ff[i] + tmp.bet[i]) * tmp.alf[i + 1]
    vv[-1] = 0
    for i in range(vv.shape[0] - 2, 0 - 1, -1):
        vv[i] = tmp.alf[i + 1] * vv[i + 1] + tmp.bet[i + 1]


cpdef void Posson_reduct_12(PlasmaSolverConfig config,
                            double[:] r0,  # n_dim
                            double[:] r1,  # n_dim
                            double[:, :] Fi,  # n_dim, n_dim
                            double[:, :] P,  # n_dim, n_dim
                            ProgonkaTmp tmp):
    cdef double a, a_, alfa
    cdef unsigned int i, j, k
    cdef unsigned long nj1, ii, jk, l, nk
    cdef int m1p

    # Filling P
    for i in range(config.n_dim):
        P[0, i] = config.h**2 * (Fi[0, i] + 2 / config.h * r0[i])
        P[-1, i] = config.h**2 * (Fi[-1, i] + 2 / config.h * r1[i])

    for j in range(config.n_dim):
        P[j, 0] = 0
        P[j, -1] = 0

    for j in range(1, config.n_dim - 2 + 1):
        for i in range(1, config.n_dim - 1):
            P[i, j] = config.h**2 * Fi[i, j]

    # Direct way
    for k in range(1, config.npq - 1 + 1):  # NOTE: 1
        jk = <unsigned long> (2 ** (k - 1))
        nj1 = <unsigned long> (2 ** (config.npq - k) - 1)
        for ii in range(1, nj1 + 1):  # NOTE: 1
            j = <unsigned long> (ii * 2 * jk)
            for i in range(config.n_dim):
                tmp.RedFi[i] = P[i, j - jk] + P[i, j + jk]
                tmp.Svl[i] = 0

            for l in range(1, jk + 1):  # NOTE: 1
                a_ = 1 + (1 - cos((2 * l - 1) * pi / (2 * jk)))
                a = (2 * a_ + config.h**2)
                m1p = (-1) ** (l + 1)  # I hate that -1**2 == -1...
                alfa = sin((2 * l - 1) * pi / (2 * jk)) * m1p / jk
                for i in range(config.n_dim):
                    tmp.PrF[i] = alfa * tmp.RedFi[i]

                Progonka(a, tmp.PrF, tmp.PrV, tmp)
                for i in range(config.n_dim):
                    tmp.Svl[i] += tmp.PrV[i]

            for i in range(config.n_dim):
                P[i, j] = .5 * (P[i, j] + tmp.Svl[i])

    # Back way
    for k in range(config.npq, 0, -1):
        jk = <unsigned long> (2 ** (k - 1))
        nk = <unsigned long> (2 ** (config.npq - k))
        for ii in range(1, nk + 1):  # NOTE: 1
            j = (2 * ii - 1) * jk
            for i in range(config.n_dim):
                tmp.RedFi[i] = P[i, j - jk] + P[i, j + jk]
                tmp.Psi[i] = P[i, j]
                tmp.Svl[i] = 0

            for l in range(1, jk + 1):  # NOTE: 1
                a_ = 1 + (1 - cos((2 * l - 1) * pi / (2 * jk)))
                a = (2 * a_ + config.h**2)
                m1p = (-1) ** (l + 1)
                alfa = sin((2 * l - 1) * pi / (2 * jk)) * m1p / jk
                for i in range(config.n_dim):
                    tmp.PrF[i] = tmp.Psi[i] + alfa * tmp.RedFi[i]

                Progonka(a, tmp.PrF, tmp.PrV, tmp)
                for i in range(config.n_dim):
                    tmp.Svl[i] += tmp.PrV[i]

            for i in range(config.n_dim):
                P[i, j] = tmp.Svl[i]


cpdef void reduction_Dirichlet1(PlasmaSolverConfig config,
                                double[:, :] Fi,  # n_dim, n_dim
                                double[:, :] P,  # n_dim, n_dim
                                ProgonkaTmp tmp):
    cdef double a, alfa
    cdef unsigned int i, j, k
    cdef unsigned long nj1, ii, jk, nk, l
    cdef int m1p

    # Filling P
    for i in range(config.n_dim):
        P[0, i] = 0
        P[-1, i] = 0
        P[i, 0] = 0
        P[i, -1] = 0

    for j in range(1, config.n_dim - 1):
        for i in range(1, config.n_dim - 1):
            P[i, j] = config.h**2 * Fi[i, j]

    for k in range(1, config.npq):  # NOTE: 1
        jk = 2 ** (k - 1)
        nj1 = 2 ** (config.npq - k) - 1
        for ii in range(1, nj1 + 1):  # NOTE: 1
            j = ii * 2 * jk
            for i in range(1, tmp.RedFi.shape[0] - 1):  # NOTE: 1
                tmp.RedFi[i] = P[i, j - jk] + P[i, j + jk]
                tmp.Svl[i] = 0

            for l in range(1, jk + 1):  # NOTE: 1
                m1p = (-1) ** (l + 1)
                a = 2 * (1 + (1 - cos((2 * l - 1) * pi / (2 * jk))))
                alfa = sin((2 * l - 1) * pi / (2 * jk)) * m1p / jk
                for i in range(1, tmp.PrF.shape[0] - 1):  # NOTE: 1
                    tmp.PrF[i] = alfa * tmp.RedFi[i]

                Progonka_Dirichlet(a, tmp.PrF, tmp.PrV, tmp)
                for i in range(1, tmp.Svl.shape[0] - 1):  # NOTE: 1
                    tmp.Svl[i] += tmp.PrV[i]

            for i in range(1, P.shape[0] - 1):  # NOTE: 1
                P[i, j] = .5 * (P[i, j] + tmp.Svl[i])
# c  back way********************************************************
    for k in range(config.npq, 0, -1):
        jk = 2 ** (k - 1)
        nk = 2 ** (config.npq - k)
        for ii in range(1, nk + 1):  # NOTE: 1
            j = (2 * ii - 1) * jk
            for i in range(1, config.n_dim - 1):  # NOTE: 1
                tmp.RedFi[i] = P[i, j - jk] + P[i, j + jk]
                tmp.Psi[i] = P[i, j]
                tmp.Svl[i] = 0
            for l in range(1, jk + 1):  # NOTE: 1
                a = 2 * (1 + (1 - cos((2 * l - 1) * pi / (2 * jk))))
                m1p = (-1) ** (l + 1)
                alfa = sin((2 * l - 1) * pi / (2 * jk)) * m1p / jk
                for i in range(1, config.n_dim - 1):  # NOTE: 1
                    tmp.PrF[i] = tmp.Psi[i] + alfa * tmp.RedFi[i]
                Progonka_Dirichlet(a, tmp.PrF, tmp.PrV, tmp)
                for i in range(1, config.n_dim - 1):  # NOTE: 1
                    tmp.Svl[i] += tmp.PrV[i]
            for i in range(1, config.n_dim - 1):  # NOTE: 1
                P[i, j] = tmp.Svl[i]


cpdef Progonka_C(PlasmaSolverConfig config,
                 double[:, :] ff,  # n_dim, n_dim
                 double[:, :] vv,  # n_dim, n_dim
                 ProgonkaTmp tmp):
    cdef double aa, a_n_prka, b_0_prka
    cdef unsigned int i, j, M
    M = config.n_dim - 1
    a_n_prka = 2
    b_0_prka = 2
    aa = 4

    tmp.alf[1] = b_0_prka / aa
    for j in range(0, M + 1, 2):
        tmp.bet[1] = ff[0, j] / aa
        for i in range(1, M - 1 + 1):
            tmp.alf[i + 1] = 1 / (aa - tmp.alf[i])
            tmp.bet[i + 1] = (ff[i, j] + tmp.bet[i]) * tmp.alf[i + 1]

        tmp.bet[-1] = ((ff[M, j] + a_n_prka * tmp.bet[M]) /
                       (aa - a_n_prka * tmp.alf[M]))
        vv[M, j] = tmp.bet[-1]
        for i in range(M - 1, 0 - 1, -1):
            vv[i, j] = tmp.alf[i + 1] * vv[i + 1, j] + tmp.bet[i + 1]


cpdef Progonka_C_l_km1(PlasmaSolverConfig config,
                       double kk,
                       double[:, :] ff,  # n_dim, n_dim
                       double[:, :] vv,  # n_dim, n_dim
                       ProgonkaTmp tmp):
    # TODO: move this allocation to colder code
    cdef np.ndarray[double, ndim=2] p = np.zeros((config.n_dim, config.n_dim))
    cdef double aa, a_n_prka, b_0_prka
    cdef unsigned int i, j, M, l, i1, ii, nj1s, jks
    a_n_prka = 2
    b_0_prka = 2
    M = config.n_dim - 1
    nj1s = <unsigned int> (2 ** (config.npq - kk))
    jks = <unsigned int> (2 ** (kk - 1))

    # Filling p
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            p[i, j] = 0

    for l in range(1, jks + 1):  # NOTE: 1
        aa = 2 * (1 + (1 - cos((2 * l - 1) * pi / (2 * jks))))
        for i1 in range(1, M - 1 + 1):  # NOTE: 1
            p[i1, i1] = aa
            p[i1, i1 + 1] = -1
            p[i1, i1 - 1] = -1

        p[0, 0] = aa
        p[0, 1] = -1
        p[M, M - 1] = -1
        p[M, M] = aa

        tmp.alf[1] = b_0_prka / aa
        for ii in range(nj1s + 1):
            j = ii * 2 * jks
            tmp.bet[1] = ff[0, j] / aa

            for i in range(1, M - 1 + 1):  # NOTE: 1
                tmp.alf[i + 1] = 1 / (aa - tmp.alf[i])
                tmp.bet[i + 1] = (ff[i, j] + tmp.bet[i]) * tmp.alf[i + 1]

            tmp.bet[-1] = ((ff[M, j] + a_n_prka * tmp.bet[M]) /
                           (aa - a_n_prka * tmp.alf[M]))
            vv[M, j] = tmp.bet[-1]
            ff[M, j] = vv[M, j]
            for i in range(M - 1, 0 - 1, -1):
                vv[i, j] = tmp.alf[i + 1] * vv[i + 1, j] + tmp.bet[i + 1]
                ff[i, j] = vv[i, j]


cpdef Progonka_C_l_km1_0N(PlasmaSolverConfig config,
                          double kk,
                          double[:, :] ff,  # n_dim, n_dim
                          double[:, :] vv,  # n_dim, n_dim
                          ProgonkaTmp tmp):
    cdef double aa, a_n_prka, b_0_prka
    cdef unsigned int i, j, M, l, i1, ii, nj1s, jks

    M = config.n_dim - 1
    a_n_prka = 2
    b_0_prka = 2
    jks = <unsigned int> 2 ** (kk - 1)

    for l in range(1, jks + 1):  # NOTE: 1
        aa = 2 * (1 + (1 - cos((2 * l - 1) * pi / (2 * jks))))
        tmp.alf[1] = b_0_prka / aa
        for j in range(0, M + 1, M):  # TODO: make it Cython-optimizable
            tmp.bet[1] = ff[0, j] / aa
            for i in range(1, M - 1 + 1):  # NOTE: 1
                tmp.alf[i + 1] = 1 / (aa - tmp.alf[i])
                tmp.bet[i + 1] = (ff[i, j] + tmp.bet[i]) * tmp.alf[i + 1]

            tmp.bet[-1] = ((ff[M, j] + a_n_prka * tmp.bet[M]) /
                           (aa - a_n_prka * tmp.alf[M]))
            vv[M, j] = tmp.bet[-1]
            ff[M, j] = vv[M, j]
            for i in range(M - 1, 0 - 1, -1):
                vv[i, j] = tmp.alf[i + 1] * vv[i + 1, j] + tmp.bet[i + 1]
                ff[i, j] = vv[i, j]


cpdef Progonka_C_l_km1_0N_Y(PlasmaSolverConfig config,
                            double kk,
                            double p_0N,
                            double[:, :] ff,  # n_dim, n_dim
                            double[:, :] vv,  # n_dim, n_dim
                            ProgonkaTmp tmp):
    cdef double aa, a_n_prka, b_0_prka, nj1s
    cdef unsigned int i, j, M, l, i1, ii, jks, jks2

    a_n_prka = 2
    b_0_prka = 2
    M = config.n_dim - 1
    jks = <unsigned int> (2 ** (kk - 1))
    jks2 = 2 * jks
    for l in range(1, jks2 - 1 + 1):  # NOTE: 1
        aa = 2 * (1 + (1 - cos(pi * l / jks)))
        tmp.alf[1] = b_0_prka / aa
        j = 0
        tmp.bet[1] = ff[0, j] / aa
        for i in range(1, M - 1 + 1):  # NOTE: 1
            tmp.alf[i + 1] = 1 / (aa - tmp.alf[i])
            tmp.bet[i + 1] = (ff[i, j] + tmp.bet[i]) * tmp.alf[i + 1]

        tmp.bet[-1] = ((ff[M, j] + a_n_prka * tmp.bet[M]) /
                       (aa - a_n_prka * tmp.alf[M]))
        vv[M, j] = tmp.bet[-1]
        ff[M, j] = vv[M, j]
        for i in range(M - 1, 0 - 1, -1):
            vv[i, j] = tmp.alf[i + 1] * vv[i + 1, j] + tmp.bet[i + 1]
            ff[i, j] = vv[i, j]

    l = jks2
    aa = 2 * (1 + (1 - cos(pi * l / jks)))
    tmp.alf[1] = b_0_prka / aa
    j = 0
    tmp.bet[1] = ff[0, j] / aa
    for i in range(1, M - 1 + 1):  # NOTE: 1
        tmp.alf[i + 1] = 1 / (aa - tmp.alf[i])
        tmp.bet[i + 1] = (ff[i, j] + tmp.bet[i]) * tmp.alf[i + 1]

    tmp.bet[-1] = -p_0N
    vv[M, j] = tmp.bet[-1]
    ff[M, j] = vv[M, j]
    for i in range(M - 1, 0 - 1, -1):
        vv[i, j] = tmp.alf[i + 1] * vv[i + 1, j] + tmp.bet[i + 1]
        ff[i, j] = vv[i, j]


cpdef Progonka_C_Y(PlasmaSolverConfig config,
                   double[:, :] ff,  # n_dim, n_dim
                   double[:, :] vv,  # n_dim, n_dim
                   ProgonkaTmp tmp):
    cdef double aa, a_n_prka, b_0_prka
    cdef unsigned int i, j, M
    a_n_prka = 2
    b_0_prka = 2
    M = config.n_dim - 1
    aa = 4

    tmp.alf[1] = b_0_prka / aa
    for j in range(1, M - 1 + 1, 2):
        tmp.bet[1] = ff[0, j] / aa
        for i in range(1, M - 1 + 1):  # NOTE: 1
            tmp.alf[i + 1] = 1 / (aa - tmp.alf[i])
            tmp.bet[i + 1] = (ff[i, j] + tmp.bet[i]) * tmp.alf[i + 1]

        tmp.bet[-1] = ((ff[M, j] + a_n_prka * tmp.bet[M]) /
                       (aa - a_n_prka * tmp.alf[M]))
        vv[M, j] = tmp.bet[-1]
        for i in range(M - 1, 0 - 1, -1):
            vv[i, j] = tmp.alf[i + 1] * vv[i + 1, j] + tmp.bet[i + 1]


cpdef Progonka_C_l_km1_Y(PlasmaSolverConfig config,
                         unsigned int kk,
                         double[:, :] ff,  # n_dim, n_dim
                         double[:, :] vv,  # n_dim, n_dim
                         ProgonkaTmp tmp):
    cdef double aa, a_n_prka, b_0_prka
    cdef unsigned int i, j, M, l, ii, nks, jks
    a_n_prka = 2
    b_0_prka = 2

    M = config.n_dim - 1
    nks = <unsigned int> 2 ** (config.npq - kk + 1)
    jks = <unsigned int> 2 ** (kk - 1)

    for l in range(1, jks + 1):  # NOTE: 1
        aa = 2 * (1 + (1 - cos((2 * l - 1) * pi / (2 * jks))))
        tmp.alf[1] = b_0_prka / aa
        for ii in range(1, nks - 1 + 1, 2):
            j = ii * jks
            tmp.bet[1] = ff[0, j] / aa

            for i in range(1, M - 1 + 1):  # NOTE: 1
                tmp.alf[i + 1] = 1 / (aa - tmp.alf[i])
                tmp.bet[i + 1] = (ff[i, j] + tmp.bet[i]) * tmp.alf[i + 1]

            tmp.bet[-1] = ((ff[M, j] + a_n_prka * tmp.bet[M]) /
                           (aa - a_n_prka * tmp.alf[M]))
            vv[M, j] = tmp.bet[-1]
            ff[M, j] = vv[M, j]
            for i in range(M - 1, 0 - 1, -1):
                vv[i, j] = tmp.alf[i + 1] * vv[i + 1, j] + tmp.bet[i + 1]
                ff[i, j] = vv[i, j]


cpdef Neuman_red(PlasmaSolverConfig config,
                 double B_00,
                 double[:] r0,  # n_dim
                 double[:] r1,  # n_dim
                 double[:] rb,  # n_dim
                 double[:] ru,  # n_dim
                 double[:, :] q,  # n_dim, n_dim
                 double[:, :] YE,  # n_dim, n_dim
                 ProgonkaTmp tmp):
    # TODO: move this allocation to colder code
    cdef np.ndarray[double, ndim=2] p = np.zeros((config.n_dim, config.n_dim))
    cdef np.ndarray[double, ndim=2] v = np.zeros((config.n_dim, config.n_dim))
    cdef np.ndarray[double, ndim=2] v1 = np.zeros((config.n_dim, config.n_dim))
    cdef double p_M0, a, s_sol, x, y, s_numsol
    cdef unsigned int i, j, k, nj1, ii, jk, nk, M, N1
# c *************Direct way*************************************
    M = config.n_dim - 1
    for j in range(M + 1):
        for i in range(M + 1):
            YE[i, j] = 0
            p[i, j] = 0
            v[i, j] = 0
            v1[i, j] = 0

    for j in range(M + 1):
        q[0, j] = config.h**2 * (q[0, j] + 2 / config.h * r0[j])
        q[M, j] = config.h**2 * (q[M, j] + 2 / config.h * r1[j])

    for i in range(1, M):  # NOTE: 1
        q[i, 0] = config.h**2 * (q[i, 0] + 2 / config.h * rb[i])
        q[i, M] = config.h**2 * (q[i, M] + 2 / config.h * ru[i])

    q[0, 0] += config.h * 2 * rb[0]
    q[0, M] += config.h * 2 * ru[0]
    q[M, 0] += config.h * 2 * rb[M]
    q[M, M] += config.h * 2 * ru[M]

    for j in range(1, M):  # NOTE: 1
        for i in range(1, M):  # NOTE: 1
            q[i, j] = config.h**2 * q[i, j]

    # ******************************************
    k = 1
    jk = 1
    for j in range(0, M + 1, 2):
        for i in range(M + 1):
            v[i, j] = q[i, j]

    Progonka_C(config, v, v1, tmp)
    for j in range(2, M - 2 + 1, 2):
        for i in range(M + 1):
            p[i, j] = v1[i, j]
            q[i, j] = q[i, j - 1] + q[i, j + 1] + 2 * p[i, j]

    for i in range(M + 1):
        p[i, 0] = v1[i, 0]
        q[i, 0] = 2 * q[i, 1] + 2 * p[i, 0]
        p[i, M] = v1[i, M]
        q[i, M] = 2 * q[i, M - 1] + 2 * p[i, M]

    # *******************************************
    k = 2

    for k in range(2, config.npq - 1 + 1):  # NOTE: 1
        nj1 = <unsigned int> (2 ** (config.npq - k) - 1)
        jk = <unsigned int> (2 ** (k - 1))
        for ii in range(1, nj1 + 1):  # NOTE: 1
            j = ii * 2 * jk
            for i in range(M + 1):
                v[i, j] = p[i, j - jk] + p[i, j + jk] + q[i, j]

        for i in range(M + 1):
            v[i, 0] = 2 * p[i, jk] + q[i, 0]
            v[i, M] = 2 * p[i, M - jk] + q[i, M]

        Progonka_C_l_km1(config, k, v, v1, tmp)
        for i in range(M + 1):
            for ii in range(1, nj1 + 1):  # NOTE: 1
                j = ii * 2 * jk  # pow(2, k)
                p[i, j] += v1[i, j]
                q[i, j] = q[i, j - jk] + q[i, j + jk] + 2 * p[i, j]

            p[i, 0] += v1[i, 0]
            q[i, 0] = 2 * q[i, jk] + 2 * p[i, 0]
            p[i, M] += v1[i, M]
            q[i, M] = 2 * q[i, M - jk] + 2 * p[i, M]

    # ************************************
    k = config.npq

    jk = <unsigned int> (2 ** (config.npq - 1))
    for i in range(M + 1):
        v[i, 0] = 2 * p[i, jk] + q[i, 0]
        v[i, M] = 2 * p[i, M - jk] + q[i, M]

    Progonka_C_l_km1_0N(config, k, v, v1, tmp)
    for i in range(M + 1):
        p[i, 0] += v1[i, 0]
        q[i, 0] = 2 * q[i, jk] + 2 * p[i, 0]
        p[i, M] += v1[i, M]
        q[i, M] = 2 * q[i, M - jk] + 2 * p[i, M]

    # c  back way********* YN  Y0 ***************************
    p_M0 = p[M, 0]

    for i in range(M + 1):
        v[i, 0] = 2 * p[i, 0] + q[i, M]

    Progonka_C_l_km1_0N_Y(config, config.npq, p_M0, v, v1, tmp)
    for i in range(M + 1):
        YE[i, 0] = p[i, 0] + v1[i, 0]
        YE[i, M] = p[i, M] + v1[i, 0]

    for k in range(config.npq, 2 - 1, -1):
        jk = <unsigned int> (2 ** (k - 1))
        nk = <unsigned int> (2 ** (config.npq - k + 1))
        for ii in range(1, nk - 1 + 1, 2):  # NOTE: 1
            j = ii * jk
            for i in range(M + 1):
                v[i, j] = YE[i, j - jk] + YE[i, j + jk] + q[i, j]
        Progonka_C_l_km1_Y(config, k, v, v1, tmp)
        for ii in range(1, nk - 1 + 1, 2):  # NOTE: 1
            j = ii * jk
            for i in range(M + 1):
                YE[i, j] = p[i, j] + v1[i, j]
    k = 1
    for j in range(1, M - 1 + 1, 2):
        for i in range(M + 1):
            v[i, j] = YE[i, j - 1] + YE[i, j + 1] + q[i, j]
    Progonka_C_Y(config, v, v1, tmp)
    for j in range(1, M - 1 + 1, 2):
        for i in range(M + 1):
            YE[i, j] = v1[i, j]

    # Proverka*******************************************
    s_numsol = 0
    for i in range(M + 1):
        for j in range(M + 1):
            s_numsol += YE[i, j]

    s_numsol /= (M + 1) * (M + 1)
    # print("@@@@@@@@ s_numsol  %e" % s_numsol)
    for i in range(M + 1):
        for j in range(M + 1):
            YE[i, j] -= s_numsol + config.B_0 / (config.x_max * 2)**2


# Not returning void allows exception proparation

cpdef response(PlasmaSolverConfig config,
               unsigned long xi_i,
               np.ndarray[plasma_particle.t] in_plasma,
               np.ndarray[plasma_particle.t] in_plasma_cor,
               np.ndarray[double, ndim=2] beam_ro,  # n_dim, n_dim
               np.ndarray[RoJ_t, ndim=2] roj_pprv,  # n_dim, n_dim
               np.ndarray[RoJ_t, ndim=2] roj_prev,  # n_dim, n_dim
               np.ndarray[double, ndim=2] mut_Ex,  # n_dim
               np.ndarray[double, ndim=2] mut_Ey,  # n_dim
               np.ndarray[double, ndim=2] mut_Ez,  # n_dim
               np.ndarray[double, ndim=2] mut_Bx,  # n_dim
               np.ndarray[double, ndim=2] mut_By,  # n_dim
               np.ndarray[double, ndim=2] mut_Bz,  # n_dim
               np.ndarray[plasma_particle.t] out_plasma,
               np.ndarray[plasma_particle.t] out_plasma_cor,
               np.ndarray[RoJ_t, ndim=2] out_roj,  # n_dim, n_dim
               ):
    cdef:
        # Charge density and current on a grid
        np.ndarray[RoJ_t, ndim=2] roj

        # Same for corrector passes temporary storage
        np.ndarray[RoJ_t, ndim=2] roj_cor = (
            np.zeros((config.n_dim, config.n_dim), dtype=RoJ_dtype)
        )
        # Larger array for tail picking (symmetrical boundary condition)
        np.ndarray[RoJ_t, ndim=2] roj_cor_larger_tmp = (
            np.zeros((config.n_dim + 2, config.n_dim + 2), dtype=RoJ_dtype)
        )

        # 8 special arrays for outermost grid cells
        np.ndarray[RoJ_t, ndim=3] roj_edges = (
            np.zeros((4, 2, config.n_dim), dtype=RoJ_dtype)
        )
        # Indices meaning: (TODO: change to something saner)
        #     first index: 0 = l, 1 = r, 2 = b, 3 = u
        #     second index: 0 is left(bottom), 1 is right(top)
        #     4 = b1, 5 = b2, 6 = N, 7 = N1,
        # rl1[k] = roj_edges[0, 0, k].ro
        # cjyb2[k] == roj_edges[2, 1, k].jy

        # derivatives
        np.ndarray[double, ndim=2] djx_dx, djx_dy, djx_dxi
        np.ndarray[double, ndim=2] djy_dx, djy_dy, djy_dxi
        np.ndarray[double, ndim=2] djz_dx, djz_dy
        np.ndarray[double, ndim=2] dro_dx, dro_dy

        # Plasma particles
        np.ndarray[plasma_particle.t] plasma_particles

        # Same for corrector passes temporary storage
        np.ndarray[plasma_particle.t] plasma_particles_cor

        # fields
        np.ndarray[double, ndim=2] Ex, Ey, Ez, Bx, By, Bz
        np.ndarray[double, ndim=2] Ex_approx, Ey_approx, Bx_approx, By_approx

        # fields at particle positions
        np.ndarray[double] Exs = np.zeros(len(in_plasma))
        np.ndarray[double] Eys = np.zeros(len(in_plasma))
        np.ndarray[double] Ezs = np.zeros(len(in_plasma))
        np.ndarray[double] Bxs = np.zeros(len(in_plasma))
        np.ndarray[double] Bys = np.zeros(len(in_plasma))
        np.ndarray[double] Bzs = np.zeros(len(in_plasma))

    cdef unsigned int i, j, i_corr
    cdef double r

    # Pre-configure
    if config.interpolation_order == -1:
        interpolate_fields = interpolate_fields_fs
    elif config.interpolation_order == -2:
        interpolate_fields = interpolate_fields_fs9
    else:
        interpolate_fields = interpolate_fields_sl

    # if config.print_every and openmp.omp_get_max_threads() > 1:
    #     print('Using OpenMP')
    #     print('Up to', openmp.omp_get_max_threads(), 'threads available')
    #     if config.limit_threads:
    #         print('Manually limited to', config.limit_threads, 'threads')

    cdef ProgonkaTmp tmp = ProgonkaTmp(config)

    # Allocate memory if not using preallocated output arrays
    roj = out_roj if out_roj is not None else (
        np.zeros((config.n_dim, config.n_dim), dtype=RoJ_dtype)
    )

    Ex = mut_Ex if mut_Ex is not None else (
        np.zeros((config.n_dim, config.n_dim))
    )
    Ey = mut_Ey if mut_Ey is not None else np.zeros_like(Ex)
    Ez = mut_Ez if mut_Ez is not None else np.zeros_like(Ex)
    Bx = mut_Bx if mut_Bx is not None else np.zeros_like(Ex)
    By = mut_By if mut_By is not None else np.zeros_like(Ex)
    Bz = mut_Bz if mut_Bz is not None else np.zeros_like(Ex)
    Ex_approx = Ex.copy()
    Ey_approx = Ey.copy()
    Bx_approx = Bx.copy()
    By_approx = By.copy()

    if in_plasma_cor is None:
        in_plasma_cor = in_plasma.copy()
    if out_plasma is not None:
        plasma_particles = out_plasma
        plasma_particles[...] = in_plasma
    else:
        plasma_particles = in_plasma.copy()
    if out_plasma_cor is not None:
        plasma_particles_cor = out_plasma_cor
        plasma_particles_cor[...] = in_plasma_cor
    else:
        plasma_particles_cor = in_plasma_cor.copy()

    beam_ro_dx, beam_ro_dy = np.gradient(beam_ro, config.h, config.h,
                                         edge_order=2)

    # Predict phase
    interpolate_fields(config,
                       plasma_particles['x'], plasma_particles['y'],
                       Ex, Ey, Ez, Bx, By, Bz,
                       Exs, Eys, Ezs, Bxs, Bys, Bzs)

    move_particles(config, plasma_particles, plasma_particles_cor,
                   Exs, Eys, Ezs, Bxs, Bys, Bzs, .5)
    # plasma_particles_cor.v is an estimation @ m+1/2
    # plasma_particles_cor.x is an estimation @ m+1
    # plasma_particles.v was @ m-1/2, now is @ m+1/2

    plasma_virtualized = config.virtualize(plasma_particles)
    ro_and_j_ie_Vshivkov(config, plasma_virtualized, plasma_virtualized,
                         roj, roj_edges)
    if config.zero_edges:
        roj_edges[...] = 0
    # We only have v @ m+1/2 for now, using plasma_particles twice

    djx_dx = pader_x(config, roj['jx'], roj_edges['jx'])
    djx_dy = pader_y(config, roj['jx'], roj_edges['jx'])
    djy_dx = pader_x(config, roj['jy'], roj_edges['jy'])
    djy_dy = pader_y(config, roj['jy'], roj_edges['jy'])
    djz_dx = pader_x(config, roj['jz'], roj_edges['jz'])
    djz_dy = pader_y(config, roj['jz'], roj_edges['jz'])
    dro_dx = pader_x(config, roj['ro'], roj_edges['ro'])
    dro_dy = pader_y(config, roj['ro'], roj_edges['ro'])
    if xi_i == 0:
        djx_dxi = (roj_prev['jx'] - roj['jx']) / config.h3
        djy_dxi = (roj_prev['jy'] - roj['jy']) / config.h3
    else:
        djx_dxi = (-3 * roj['jx'] +
                   +4 * roj_prev['jx'] +
                   -1 * roj_pprv['jx']) / (config.h3 * 2)
        djy_dxi = (-3 * roj['jy'] +
                   +4 * roj_prev['jy'] +
                   -1 * roj_pprv['jy']) / (config.h3 * 2)

    # Field Ez, predict
    reduction_Dirichlet1(config, -(djx_dx + djy_dy), Ez, tmp)

    # Field Bz, predict
    Neuman_red(config, config.B_0,
               -roj[0, :]['jy'], roj[-1, :]['jy'],
               roj[:, 0]['jx'], -roj[:, -1]['jx'],
               -(djx_dy - djy_dx), Bz,
               tmp)

    # Field Ex, predict
    Posson_reduct_12(config,
                     -(beam_ro[0] +
                         roj['ro'][0] * config.boundary_suppression),
                     +(beam_ro[-1] +
                         roj['ro'][-1] * config.boundary_suppression),
                     -(beam_ro_dx + dro_dx - djx_dxi) + Ex, Ex_approx,
                     tmp)

    # Field Ey, predict
    # TODO: sign!
    Posson_reduct_12(config,
                     -(beam_ro[:, 0] +
                         roj['ro'][:, 0] * config.boundary_suppression),
                     +(beam_ro[:, -1] +
                         roj['ro'][:, -1] * config.boundary_suppression),
                     (-(beam_ro_dy + dro_dy - djy_dxi) + Ey).T, Ey_approx.T,
                     tmp)

    # Field By, predict
    # пр часть обчно, гр - наоборот  # TODO: translate
    # TODO: translate
    # c если поменять знак граничных условий (будет стандартно),
    # то By совпадет с Ex
    Posson_reduct_12(config,
                     beam_ro[0] + roj['jz'][0],
                     -(beam_ro[-1] + roj['jz'][-1]),
                     -(beam_ro_dx + djz_dx - djx_dxi) + By, By_approx,
                     tmp)

    # Field Bx, predict
    # TODO: signs!
    # TODO: translate
    # Fi_Bx  определили с точностью до -1##.
    # Поэтому Fi_Brout(j,i) = Fi_Bx[i,j]+Bx[i,j]
    # а в Posson_reduct_12, правая часть идет с +; граница -обычно
    # c если поменять знак правой части (минус перед скобкой),
    # то Bx совпадет с Ey
    Posson_reduct_12(config,
                     -(beam_ro[:, 0] + roj['jz'][:, 0]),
                     beam_ro[:, -1] + roj['jz'][:, -1],
                     (+(beam_ro_dy + djz_dy - djy_dxi) + Bx).T, Bx_approx.T,
                     tmp)

    # Correct phase

    # Ruins everything
    if config.reuse_EB:
        Ex[...] = Ex_approx
        Ey[...] = Ey_approx
        Bx[...] = Bx_approx
        By[...] = By_approx

    for i_corr in range(config.n_corr):
        plasma_particles_cor[...] = plasma_particles

        interpolate_fields(config,
                           plasma_particles_cor['x'],
                           plasma_particles_cor['y'],
                           Ex_approx, Ey_approx, Ez, Bx_approx, By_approx, Bz,
                           Exs, Eys, Ezs, Bxs, Bys, Bzs)
        move_particles(config, plasma_particles_cor, plasma_particles_cor,
                       Exs, Eys, Ezs, Bxs, Bys, Bzs)
        # plasma_particles_cor.v is @ m + 3/2
        # plasma_particles_cor.x is an estimation @ m+2

        if config.use_average_speed:
            plasma_virtualized = config.virtualize(plasma_particles)
            plasma_virtualized_cor = config.virtualize(plasma_particles_cor)
            ro_and_j_ie_Vshivkov(config, plasma_virtualized,
                                 plasma_virtualized_cor, roj, roj_edges)
            if config.zero_edges:
                roj_edges[...] = 0
            djx_dx = pader_x(config, roj['jx'], roj_edges['jx'])
            djx_dy = pader_y(config, roj['jx'], roj_edges['jx'])
            djy_dx = pader_x(config, roj['jy'], roj_edges['jy'])
            djy_dy = pader_y(config, roj['jy'], roj_edges['jy'])
            djz_dx = pader_x(config, roj['jz'], roj_edges['jz'])
            djz_dy = pader_y(config, roj['jz'], roj_edges['jz'])
            dro_dx = pader_x(config, roj['ro'], roj_edges['ro'])
            dro_dy = pader_y(config, roj['ro'], roj_edges['ro'])

        # We have v @ m+3/2 now, using both to estimate v @ m+1

        # Field Ez, correct (experimental)
        reduction_Dirichlet1(config, -(djx_dx + djy_dy), Ez, tmp)

        # Field Bz, correct (experimental)
        Neuman_red(config, config.B_0,
                   -roj[0, :]['jy'], roj[-1, :]['jy'],
                   roj[:, 0]['jx'], -roj[:, -1]['jx'],
                   -(djx_dy - djy_dx), Bz,
                   tmp)

        plasma_virtualized_cor = config.virtualize(plasma_particles_cor)
        ro_and_j_ie_cor_Vshivkov(config, plasma_virtualized_cor,
                                 roj_cor, roj_cor_larger_tmp)
        djx_dxi = (roj_prev['jx'] - roj_cor['jx']) / (config.h3 * 2)
        djy_dxi = (roj_prev['jy'] - roj_cor['jy']) / (config.h3 * 2)

        for i_corr_transverse in range(config.n_corr_transverse):
            # Field Bx, correct
            Posson_reduct_12(config,
                             -(beam_ro[:, 0] + roj['jz'][:, 0]),
                             beam_ro[:, -1] + roj['jz'][:, -1],
                             +((beam_ro_dy + djz_dy - djy_dxi) + Bx).T, Bx.T,
                             tmp)

            # Field Ex, correct
            Posson_reduct_12(config,
                             -(beam_ro[0] +
                                 roj['ro'][0] * config.boundary_suppression),
                             +(beam_ro[-1] +
                                 roj['ro'][-1] * config.boundary_suppression),
                             -(beam_ro_dx + dro_dx - djx_dxi) + Ex, Ex,
                             tmp)

            # Field Ey, correct
            Posson_reduct_12(config,
                             -(beam_ro[:, 0] + roj['ro'][:, 0] *
                                 config.boundary_suppression),
                             +(beam_ro[:, -1] + roj['ro'][:, -1] *
                                 config.boundary_suppression),
                             (-(beam_ro_dy + dro_dy - djy_dxi) + Ey).T, Ey.T,
                             tmp)

            # Field By, correct
            Posson_reduct_12(config,
                             beam_ro[0] + roj['jz'][0],
                             -(beam_ro[-1] + roj['jz'][-1]),
                             -(beam_ro_dx + djz_dx - djx_dxi) + By, By,  # !!!
                             tmp)

        # Now we have more precise fields, repeat outer corrector

    return plasma_particles, plasma_particles_cor, roj, Ex, Ey, Ez, Bx, By, Bz
