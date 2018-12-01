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
Secondary authors: A. P. Sosedkin <A.P.Sosedkin@inp.nsk.su>,
                   K. V. Lotov <K.V.Lotov@inp.nsk.su>
'''


import cython
cimport cython
from cython.parallel import prange, parallel
cimport openmp

from libc.math cimport sqrt, log2, sin, cos, pi, fabs, floor, atan2

import scipy.signal

cdef extern from "math.h":
    float floorf(float) nogil
    float sqrtf(float) nogil
    float cosf(float) nogil
    float sinf(float) nogil
    float fabsf(float) nogil
    float atan2f(float, float) nogil

import numpy as np
cimport numpy as np

from .. import plasma_particle

from . import gpu_functions


### Making compatible plasma

def make_plasma(window_width, steps, per_r_step=1):
    plasma_step = window_width / steps / per_r_step
    if per_r_step % 2:  # some on zero axes, none on cell corners
        right_half = np.arange(0, window_width / 2, plasma_step)
        left_half = -right_half[:0:-1]  # invert, reverse, drop zero
        plasma_grid = np.concatenate([left_half, right_half])
    else:  # none on zero axes, none on cell corners
        right_half = np.arange(plasma_step / 2, window_width / 2, plasma_step)
        left_half = -right_half[::-1]  # invert, reverse
        plasma_grid = np.concatenate([left_half, right_half])
    assert(np.array_equal(plasma_grid, -plasma_grid[::-1]))
    N = len(plasma_grid)
    plasma_grid_xs, plasma_grid_ys = plasma_grid[:, None], plasma_grid[None, :]

    # Electron-only plasma
    plasma = np.zeros(N**2, plasma_particle.dtype)
    plasma['N'] = np.arange(plasma.shape[0])
    electrons = plasma.reshape(N, N)
    electrons['x'] = plasma_grid_xs
    electrons['y'] = plasma_grid_ys
    electrons['m'] = plasma_particle.USUAL_ELECTRON_MASS / per_r_step**2
    electrons['q'] = plasma_particle.USUAL_ELECTRON_CHARGE / per_r_step**2
    # v, p == 0
    return plasma.ravel()


### Routines written by I. A. Shalimova, adapted by A. P. Sosedkin


from .. cimport plasma_particle
from .. import plasma_particle

from .field_solver cimport FieldSolver
from .field_solver import FieldSolver


# Config
cdef class PlasmaSolverConfig:
    """Wraps relevant config values in a classy C struct for faster access."""
    cdef public unsigned int npq, n_dim
    cdef public unsigned long Lq
    cdef public double x_max, h, h3, B_0, particle_boundary#, eps,
    cdef public object virtualize
    cdef public bint variant_A_predictor, variant_A_corrector
    cdef public bint noise_reductor_enable
    cdef public double noise_reductor_equalization
    cdef public double noise_reductor_friction
    cdef public double noise_reductor_friction_pz
    cdef public double noise_reductor_reach
    cdef public double noise_reductor_final_only
    cdef public unsigned int threads

    def __init__(self, global_config):
        self.npq, unwanted = divmod(log2(global_config.grid_steps - 1), 1)
        #if unwanted:
        #    raise RuntimeError('Grid step must be N**2 + 1')
        #self.n_dim = 2**self.npq + 1
        self.n_dim = global_config.grid_steps
        self.x_max = global_config.window_width / 2
        self.h = global_config.window_width / self.n_dim
        self.h3 = global_config.xi_step_size
        self.B_0 = global_config.plasma_solver_B_0
        self.particle_boundary = (
            self.x_max - global_config.plasma_padding * self.h
        )
        #self.eps = global_config.plasma_solver_eps
        self.Lq = global_config.xi_steps
        self.virtualize = global_config.virtualize
        self.variant_A_predictor = global_config.variant_A_predictor
        self.variant_A_corrector = global_config.variant_A_corrector

        self.noise_reductor_enable = global_config.noise_reductor_enable
        self.noise_reductor_equalization = global_config.noise_reductor_equalization
        self.noise_reductor_friction = global_config.noise_reductor_friction
        self.noise_reductor_friction_pz = global_config.noise_reductor_friction_pz
        self.noise_reductor_reach = global_config.noise_reductor_reach
        self.noise_reductor_final_only = global_config.noise_reductor_final_only

        self.threads = global_config.openmp_limit_threads


# RoJ for both scalar charge density ro and vector current j, TODO: get rid of
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



# see close_range.ipynb
cdef inline float error_by_dist(float[:] error_lut, float dist) nogil:
    cdef float p = dist * (16000 / 0.35)  # calculate an index in LUT
    cdef float sign = 1 if p > 0 else -1
    p = fabsf(p)
    cdef unsigned int i1 = <unsigned int> floorf(p)
    if i1 >= 16000 - 1:
        return 0
    cdef unsigned int i2 = i1 + 1
    cdef float w1 = i2 - p
    cdef float w2 = p - i1
    cdef float v1 = error_lut[i1]
    cdef float v2 = error_lut[i2]
    return sign * (v1 * w1 + v2 * w2)


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
                                  np.ndarray[double] Bzs,
                                  ):
    """
    Calculates fields at particle positions.
    TODO: describe
    """
    cdef long k
    cdef int i, j
    cdef int ii, jj
    cdef double x_loc, y_loc, x_h, y_h
    cdef double fx1, fy1, fx2, fy2, fx3, fy3

    #assert Ex.shape[0] == Ex.shape[1] == config.n_dim

    # indexed for performance
    for k in cython.parallel.prange(xs.shape[0],
                                    nogil=True, num_threads=config.threads):
        #assert -config.particle_boundary < xs[k] < config.particle_boundary
        #assert -config.particle_boundary < ys[k] < config.particle_boundary

        # i = <unsigned int> ((config.x_max + xs[k]) / config.h)
        # j = <unsigned int> ((config.x_max + ys[k]) / config.h)
        # x_loc = config.x_max + xs[k] - i * config.h - .5 * config.h
        # y_loc = config.x_max + ys[k] - j * config.h - .5 * config.h
        x_h = xs[k] / config.h + .5
        y_h = ys[k] / config.h + .5
        i = <int> floor(x_h) + config.n_dim // 2
        j = <int> floor(y_h) + config.n_dim // 2
        x_loc = x_h - floor(x_h) - .5  # centered to -.5 to 5, not 0 to 1 because
        y_loc = y_h - floor(y_h) - .5  # the latter formulas use offset from cell center

        fx1 = .75 - x_loc**2
        fy1 = .75 - y_loc**2
        fx2 = .5 + x_loc
        fy2 = .5 + y_loc
        fx3 = .5 - x_loc
        fy3 = .5 - y_loc

        % for Fl in 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz':
        ${Fl}s[k] = (  # noqa: E201
            ${Fl}[i + 0, j + 0] * (fx1 * fy1) +              # noqa: E222
            ${Fl}[i + 1, j + 0] * (fx2**2 * (fy1 / 2)) +     # noqa: E222
            ${Fl}[i + 0, j + 1] * (fy2**2 * (fx1 / 2)) +     # noqa: E222
            ${Fl}[i + 1, j + 1] * (fx2**2 * (fy2**2 / 4)) +  # noqa: E222
            ${Fl}[i - 1, j + 0] * (fx3**2 * (fy1 / 2)) +  +  # noqa: E222
            ${Fl}[i + 0, j - 1] * (fy3**2 * (fx1 / 2)) +  +  # noqa: E222
            ${Fl}[i - 1, j - 1] * (fx3**2 * (fy3**2 / 4)) +  # noqa: E222
            ${Fl}[i - 1, j + 1] * (fx3**2 * (fy2**2 / 4)) +  # noqa: E222
            ${Fl}[i + 1, j - 1] * (fx2**2 * (fy3**2 / 4))
        )
        % endfor

cpdef void interpolate_averaged_fields_fs9(PlasmaSolverConfig config,
                                           np.ndarray[double] xs,
                                           np.ndarray[double] ys,
                                           np.ndarray[double, ndim=2] Ex_1,
                                           np.ndarray[double, ndim=2] Ey_1,
                                           np.ndarray[double, ndim=2] Ez_1,
                                           np.ndarray[double, ndim=2] Bx_1,
                                           np.ndarray[double, ndim=2] By_1,
                                           np.ndarray[double, ndim=2] Bz_1,
                                           np.ndarray[double, ndim=2] Ex_2,
                                           np.ndarray[double, ndim=2] Ey_2,
                                           np.ndarray[double, ndim=2] Ez_2,
                                           np.ndarray[double, ndim=2] Bx_2,
                                           np.ndarray[double, ndim=2] By_2,
                                           np.ndarray[double, ndim=2] Bz_2,
                                           np.ndarray[double] Exs,
                                           np.ndarray[double] Eys,
                                           np.ndarray[double] Ezs,
                                           np.ndarray[double] Bxs,
                                           np.ndarray[double] Bys,
                                           np.ndarray[double] Bzs,
                                           ):
    """
    Calculates fields at particle positions.
    TODO: describe
    """
    cdef long k
    cdef int i, j
    cdef int ii, jj
    cdef double x_loc, y_loc, x_h, y_h
    cdef double fx1, fy1, fx2, fy2, fx3, fy3

    # indexed for performance
    for k in cython.parallel.prange(xs.shape[0],
                                    nogil=True, num_threads=config.threads):
        x_h = xs[k] / config.h + .5
        y_h = ys[k] / config.h + .5
        i = <int> floor(x_h) + config.n_dim // 2
        j = <int> floor(y_h) + config.n_dim // 2
        x_loc = x_h - floor(x_h) - .5  # centered to -.5 to 5, not 0 to 1 because
        y_loc = y_h - floor(y_h) - .5  # the latter formulas use offset from cell center

        fx1 = .75 - x_loc**2
        fy1 = .75 - y_loc**2
        fx2 = .5 + x_loc
        fy2 = .5 + y_loc
        fx3 = .5 - x_loc
        fy3 = .5 - y_loc

        % for Fl in 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz':
        ${Fl}s[k] = (  # noqa: E201
            (${Fl}_1[i + 0, j + 0] + ${Fl}_2[i + 0, j + 0]) * (fx1 * fy1) +              # noqa: E222
            (${Fl}_1[i + 1, j + 0] + ${Fl}_2[i + 1, j + 0]) * (fx2**2 * (fy1 / 2)) +     # noqa: E222
            (${Fl}_1[i + 0, j + 1] + ${Fl}_2[i + 0, j + 1]) * (fy2**2 * (fx1 / 2)) +     # noqa: E222
            (${Fl}_1[i + 1, j + 1] + ${Fl}_2[i + 1, j + 1]) * (fx2**2 * (fy2**2 / 4)) +  # noqa: E222
            (${Fl}_1[i - 1, j + 0] + ${Fl}_2[i - 1, j + 0]) * (fx3**2 * (fy1 / 2)) +  +  # noqa: E222
            (${Fl}_1[i + 0, j - 1] + ${Fl}_2[i + 0, j - 1]) * (fy3**2 * (fx1 / 2)) +  +  # noqa: E222
            (${Fl}_1[i - 1, j - 1] + ${Fl}_2[i - 1, j - 1]) * (fx3**2 * (fy3**2 / 4)) +  # noqa: E222
            (${Fl}_1[i - 1, j + 1] + ${Fl}_2[i - 1, j + 1]) * (fx3**2 * (fy2**2 / 4)) +  # noqa: E222
            (${Fl}_1[i + 1, j - 1] + ${Fl}_2[i + 1, j - 1]) * (fx2**2 * (fy3**2 / 4))
        ) / 2
        % endfor


cpdef void ro_and_j_ie_Vshivkov(PlasmaSolverConfig config,
                                plasma_particle.t[:] plasma_particles,
                                # n_dim, n_dim
                                np.ndarray[RoJ_t, ndim=2] roj,
                                ):
    cdef int i, j, z
    cdef int q
    cdef long k
    cdef double x_loc, y_loc, x_h, y_h
    cdef double fx1, fy1, fx2, fy2, fx3, fy3
    cdef double dro, djx, djy, djz
    cdef double gamma_m
    cdef plasma_particle.t p
    cdef unsigned int n_dim = roj.shape[0]
    assert roj.shape[0] == roj.shape[1]

    cdef np.ndarray[RoJ_t, ndim=3] roj_tmp = np.zeros(
        (config.threads, n_dim, n_dim), RoJ_dtype
    )
    cdef np.ndarray[RoJ_t, ndim=2] roj_thread

    roj[...] = 0

    # for p in plasma_particles: indexed for performance
    # NOTE: I'm kinda worried that it works even without thread-local arrays
    # NOTE: It doesn't
    cdef int tid
    with nogil, parallel(num_threads=config.threads):
        tid = cython.parallel.threadid()
        for k in cython.parallel.prange(plasma_particles.shape[0]):
            p = plasma_particles[k]
            gamma_m = sqrt(p.m**2 + p.p[0]**2 + p.p[1]**2 + p.p[2]**2)
            dro = p.q / (1 - p.p[0] / gamma_m)
            djz = p.p[0] * (dro / gamma_m)
            djx = p.p[1] * (dro / gamma_m)
            djy = p.p[2] * (dro / gamma_m)

            # particle indices in roj and adge arrays
            # i = <unsigned int> ((config.x_max + p.x) / config.h)
            # j = <unsigned int> ((config.x_max + p.y) / config.h)
            # x_loc = config.x_max + p.x - i * config.h - .5 * config.h
            # y_loc = config.x_max + p.y - j * config.h - .5 * config.h
            x_h = p.x / config.h + .5
            y_h = p.y / config.h + .5
            i = <int> floor(x_h) + config.n_dim // 2
            j = <int> floor(y_h) + config.n_dim // 2
            x_loc = x_h - floor(x_h) - 0.5  # centered to -.5 to 5, not 0 to 1 because
            y_loc = y_h - floor(y_h) - 0.5  # the latter formulas use offset from cell center

            fx1 = .75 - x_loc**2
            fy1 = .75 - y_loc**2
            fx2 = .5 + x_loc
            fy2 = .5 + y_loc
            fx3 = .5 - x_loc
            fy3 = .5 - y_loc

            % for comp in 'ro', 'jz', 'jx', 'jy':
            roj_tmp[tid, i + 0, j + 0].${comp} += d${comp} * fx1 * fy1
            roj_tmp[tid, i + 1, j + 0].${comp} += d${comp} * fx2**2 * fy1 / 2
            roj_tmp[tid, i + 0, j + 1].${comp} += d${comp} * fy2**2 * fx1 / 2
            roj_tmp[tid, i + 1, j + 1].${comp} += d${comp} * fx2**2 * fy2**2 / 4
            roj_tmp[tid, i - 1, j + 0].${comp} += d${comp} * fx3**2 * fy1 / 2
            roj_tmp[tid, i + 0, j - 1].${comp} += d${comp} * fy3**2 * fx1 / 2
            roj_tmp[tid, i - 1, j - 1].${comp} += d${comp} * fx3**2 * fy3**2 / 4
            roj_tmp[tid, i - 1, j + 1].${comp} += d${comp} * fx3**2 * fy2**2 / 4
            roj_tmp[tid, i + 1, j - 1].${comp} += d${comp} * fx2**2 * fy3**2 / 4
            % endfor

    for i in prange(n_dim, nogil=True, num_threads=config.threads):
        for j in range(n_dim):
            for z in range(config.threads):
                % for comp in 'ro', 'jz', 'jx', 'jy':
                roj[i, j].${comp} += roj_tmp[z, i, j].${comp}
                % endfor


### Convenience Python wrappers above them; TODO: get rid of

def interpolate_fields(config, xs, ys, Ex, Ey, Ez, Bx, By, Bz):
    Exs = np.empty_like(xs)
    Eys = np.empty_like(xs)
    Ezs = np.empty_like(xs)
    Bxs = np.empty_like(xs)
    Bys = np.empty_like(xs)
    Bzs = np.empty_like(xs)
    interpolate_fields_fs9(config, xs, ys,
                           Ex, Ey, Ez, Bx, By, Bz,
                           Exs, Eys, Ezs, Bxs, Bys, Bzs)
    # may assert that particles are contained in +- particle_boundary
    # and remove a lot of inner ifs
    return Exs, Eys, Ezs, Bxs, Bys, Bzs


def interpolate_averaged_fields(config, xs, ys,
                                Ex_1, Ey_1, Ez_1, Bx_1, By_1, Bz_1,
                                Ex_2, Ey_2, Ez_2, Bx_2, By_2, Bz_2):
    Exs = np.empty_like(xs)
    Eys = np.empty_like(xs)
    Ezs = np.empty_like(xs)
    Bxs = np.empty_like(xs)
    Bys = np.empty_like(xs)
    Bzs = np.empty_like(xs)
    interpolate_averaged_fields_fs9(config, xs, ys,
                                    Ex_1, Ey_1, Ez_1, Bx_1, By_1, Bz_1,
                                    Ex_2, Ey_2, Ez_2, Bx_2, By_2, Bz_2,
                                    Exs, Eys, Ezs, Bxs, Bys, Bzs)
    # may assert that particles are contained in +- particle_boundary
    # and remove a lot of inner ifs
    return Exs, Eys, Ezs, Bxs, Bys, Bzs



def deposit(config, plasma, ion_initial_ro):
    plasma_virtualized = config.virtualize(plasma)

    roj = np.zeros((config.n_dim, config.n_dim), dtype=RoJ_dtype)
    ro_and_j_ie_Vshivkov(config, plasma_virtualized, roj)
    # may assert that particles are contained in +- particle_boundary
    # and remove a lot of inner ifs
    roj['ro'] += ion_initial_ro  # background ions
    return roj

def calculate_fields(config, field_solver,
                     roj_cur, roj_prev,
                     Ex, Ey, Ez, Bx, By, Bz,
                     beam_ro, variant_A=False):
    out_Ex, out_Ey = np.empty_like(Ex), np.empty_like(Ey)
    out_Ez, out_Bz = np.empty_like(Ez), np.empty_like(Bz)
    out_Bx, out_By = np.empty_like(Bx), np.empty_like(By)
    field_solver.calculate_fields(
        roj_cur, roj_prev, Ex, Ey, Ez, Bx, By, Bz, beam_ro,
        config.h, config.npq, config.x_max, config.h3, config.B_0,
        out_Ex, out_Ey, out_Ez, out_Bx, out_By, out_Bz,
        variant_A
    )
    return out_Ex, out_Ey, out_Ez, out_Bx, out_By, out_Bz


### More convenience functions


def average_fields(Fl1, Fl2):
    return [(component1 + component2) / 2
            for component1, component2 in zip(Fl1, Fl2)]


def gamma_m(plasma):
    return np.sqrt(plasma['m']**2 + plasma['p'][:, 0]**2 + plasma['p'][:, 1]**2 + plasma['p'][:, 2]**2)


# Particle movement


def stick_particles(config, in_plasma):
    plasma = in_plasma.copy()
    out = (
        (plasma['x'] < -config.particle_boundary) |
        (plasma['x'] > +config.particle_boundary) |
        (plasma['y'] < -config.particle_boundary) |
        (plasma['y'] > +config.particle_boundary)
    )
    plasma['p'][out, :] = 0
    plasma['x'] = np.clip(plasma['x'], -config.particle_boundary, config.particle_boundary)
    plasma['y'] = np.clip(plasma['y'], -config.particle_boundary, config.particle_boundary)
    #assert np.all(plasma['x'] <= +config.particle_boundary)
    #assert np.all(plasma['x'] >= -config.particle_boundary)
    #assert np.all(plasma['y'] <= +config.particle_boundary)
    #assert np.all(plasma['y'] >= -config.particle_boundary)
    return plasma



def reflect_particles(config, in_plasma):
    plasma = in_plasma.copy()

    reflected_r = plasma['x'] > +config.particle_boundary
    plasma['x'][reflected_r] = +2 * config.particle_boundary - plasma['x'][reflected_r]
    plasma['p'][:, 1][reflected_r] *= -1

    reflected_l = plasma['x'] < -config.particle_boundary
    plasma['x'][reflected_l] = -2 * config.particle_boundary - plasma['x'][reflected_l]
    plasma['p'][:, 1][reflected_l] *= -1

    reflected_u = plasma['y'] > +config.particle_boundary
    plasma['y'][reflected_u] = +2 * config.particle_boundary - plasma['y'][reflected_u]
    plasma['p'][:, 2][reflected_u] *= -1

    reflected_d = plasma['y'] < -config.particle_boundary
    plasma['y'][reflected_d] = -2 * config.particle_boundary - plasma['y'][reflected_d]
    plasma['p'][:, 2][reflected_d] *= -1

    #assert np.all(plasma['x'] <= +config.particle_boundary)
    #assert np.all(plasma['x'] >= -config.particle_boundary)
    #assert np.all(plasma['y'] <= +config.particle_boundary)
    #assert np.all(plasma['y'] >= -config.particle_boundary)

    return plasma


def move_simple(config, in_plasma, dxiP, edge_mode=reflect_particles):
    #assert dxiP > 0
    plasma = in_plasma.copy()
    gamma_m_ = gamma_m(plasma)
    plasma['x'] += plasma['p'][:, 1] / (gamma_m_ - plasma['p'][:, 0]) * dxiP
    plasma['y'] += plasma['p'][:, 2] / (gamma_m_ - plasma['p'][:, 0]) * dxiP
    plasma = edge_mode(config, plasma)
    return plasma


cpdef void move_simple_fast_(PlasmaSolverConfig config,
                             np.ndarray[plasma_particle.t] plasma_particles,
                             double dxiP,
                             np.ndarray[plasma_particle.t] out_plasma,
                             ):
    cdef long k
    cdef double gamma_m
    cdef plasma_particle.t p

    # for p in plasma_particles: indexed for performance
    for k in cython.parallel.prange(plasma_particles.shape[0],
                                    nogil=True, num_threads=config.threads):
        p = plasma_particles[k]

        gamma_m = sqrt(p.m**2 + p.p[0]**2 + p.p[1]**2 + p.p[2]**2)
        p.x += p.p[1] / (gamma_m - p.p[0]) * dxiP
        p.y += p.p[2] / (gamma_m - p.p[0]) * dxiP

        if p.x > config.particle_boundary:
            p.x = +2 * config.particle_boundary - p.x
            p.p[1] *= -1
        if p.x < -config.particle_boundary:
            p.x = -2 * config.particle_boundary - p.x
            p.p[1] *= -1
        if p.y > config.particle_boundary:
            p.y = +2 * config.particle_boundary - p.y
            p.p[2] *= -1
        if p.y < -config.particle_boundary:
            p.y = -2 * config.particle_boundary - p.y
            p.p[2] *= -1

        out_plasma[k] = p


def move_simple_fast(config, in_plasma, dxiP):
    out_plasma = np.empty_like(in_plasma)
    move_simple_fast_(config, in_plasma, dxiP, out_plasma)
    return out_plasma


def calculate_dp(plasma, Exs, Eys, Ezs, Bxs, Bys, Bzs, dxiP):
    gamma_m_1 = gamma_m(plasma)
    #factor_1 = plasma['q'] * dxiP / (gamma_m_1 - plasma['p'][:, 0])
    factor_1 = plasma['q'] * dxiP / (1 - plasma['p'][:, 0] / gamma_m_1)
    vx = plasma['p'][:, 1] / gamma_m_1
    vy = plasma['p'][:, 2] / gamma_m_1
    vz = plasma['p'][:, 0] / gamma_m_1
    dpx = factor_1 * (Exs + vy * Bzs - vz * Bys)
    dpy = factor_1 * (Eys - vx * Bzs + vz * Bxs)
    dpz = factor_1 * (Ezs + vx * Bys - vy * Bxs)
    return dpx, dpy, dpz


def move_smart(config, plasma_, Exs, Eys, Ezs, Bxs, Bys, Bzs, edge_mode=reflect_particles):
    plasma = plasma_.copy()

    dpx_hs, dpy_hs, dpz_hs = calculate_dp(plasma, Exs, Eys, Ezs, Bxs, Bys, Bzs, config.h3 / 2)

    plasma_predicted_hs2 = plasma.copy()
    plasma_predicted_hs2['p'][:, 1] += dpx_hs
    plasma_predicted_hs2['p'][:, 2] += dpy_hs
    plasma_predicted_hs2['p'][:, 0] += dpz_hs

    dpx, dpy, dpz = calculate_dp(plasma_predicted_hs2, Exs, Eys, Ezs, Bxs, Bys, Bzs, config.h3)

    plasma_predicted_hs3 = plasma.copy()
    plasma_predicted_hs3['p'][:, 1] += dpx / 2
    plasma_predicted_hs3['p'][:, 2] += dpy / 2
    plasma_predicted_hs3['p'][:, 0] += dpz / 2

    gamma_m_hs3 = gamma_m(plasma_predicted_hs3)
    plasma['x'] += plasma_predicted_hs3['p'][:, 1] / (gamma_m_hs3 - plasma_predicted_hs3['p'][:, 0]) * config.h3
    plasma['y'] += plasma_predicted_hs3['p'][:, 2] / (gamma_m_hs3 - plasma_predicted_hs3['p'][:, 0]) * config.h3

    plasma['p'][:, 1] += dpx
    plasma['p'][:, 2] += dpy
    plasma['p'][:, 0] += dpz

    plasma = edge_mode(config, plasma)

    return plasma


cpdef void move_smart_fast_(PlasmaSolverConfig config,
                            np.ndarray[plasma_particle.t] plasma_particles,
                            double[:] Exs,  # same as plasma_particles
                            double[:] Eys,  # -||-
                            double[:] Ezs,  # -||-
                            double[:] Bxs,  # -||-
                            double[:] Bys,  # -||-
                            double[:] Bzs,  # -||-
                            np.ndarray[plasma_particle.t] out_plasma,
                            ):
    cdef long k
    cdef double gamma_m, dpx, dpy, dpz, px, py, pz, vx, vy, vz, factor_1
    cdef plasma_particle.t p

    # for p in plasma_particles: indexed for performance
    for k in cython.parallel.prange(plasma_particles.shape[0],
                                    nogil=True, num_threads=config.threads):
        p = plasma_particles[k]

        px = p.p[1]
        py = p.p[2]
        pz = p.p[0]
        gamma_m = sqrt(p.m**2 + pz**2 + px**2 + py**2)
        vx = px / gamma_m
        vy = py / gamma_m
        vz = pz / gamma_m
        factor_1 = p.q * config.h3 / (1 - pz / gamma_m)
        dpx = factor_1 * (Exs[k] + vy * Bzs[k] - vz * Bys[k])
        dpy = factor_1 * (Eys[k] - vx * Bzs[k] + vz * Bxs[k])
        dpz = factor_1 * (Ezs[k] + vx * Bys[k] - vy * Bxs[k])

        px = p.p[1] + dpx / 2
        py = p.p[2] + dpy / 2
        pz = p.p[0] + dpz / 2
        gamma_m = sqrt(p.m**2 + pz**2 + px**2 + py**2)
        vx = px / gamma_m
        vy = py / gamma_m
        vz = pz / gamma_m
        factor_1 = p.q * config.h3 / (1 - pz / gamma_m)
        dpx = factor_1 * (Exs[k] + vy * Bzs[k] - vz * Bys[k])
        dpy = factor_1 * (Eys[k] - vx * Bzs[k] + vz * Bxs[k])
        dpz = factor_1 * (Ezs[k] + vx * Bys[k] - vy * Bxs[k])

        px = p.p[1] + dpx / 2
        py = p.p[2] + dpy / 2
        pz = p.p[0] + dpz / 2
        gamma_m = sqrt(p.m**2 + pz**2 + px**2 + py**2)

        p.x += px / (gamma_m - pz) * config.h3
        p.y += py / (gamma_m - pz) * config.h3

        p.p[1] += dpx
        p.p[2] += dpy
        p.p[0] += dpz

        if p.x > config.particle_boundary:
            p.x = +2 * config.particle_boundary - p.x
            p.p[1] *= -1
        if p.x < -config.particle_boundary:
            p.x = -2 * config.particle_boundary - p.x
            p.p[1] *= -1
        if p.y > config.particle_boundary:
            p.y = +2 * config.particle_boundary - p.y
            p.p[2] *= -1
        if p.y < -config.particle_boundary:
            p.y = -2 * config.particle_boundary - p.y
            p.p[2] *= -1

        out_plasma[k] = p


def move_smart_fast(config, plasma, Exs, Eys, Ezs, Bxs, Bys, Bzs, initial_plasma, window, noise_reductor_enable=False):
    out_plasma = np.empty_like(plasma)
    #if noise_reductor_enable:
    #    plasma = noise_reductor(config, plasma)
    move_smart_fast_(config, plasma, Exs, Eys, Ezs, Bxs, Bys, Bzs, out_plasma)
    # TODO: call noisereductor only on final movement or on all movements?
    if noise_reductor_enable:
        out_plasma = noise_reductor(config, initial_plasma, window, out_plasma)
    return out_plasma


### Noise reductor draft

def blur(arr, window):
    #assert arr.shape[0] == arr.shape[1]
    #return np.abs(np.fft.ifft2(np.fft.fft2(arr) * window))
    return np.fft.ifft2(np.fft.fft2(arr) * window).real
    #import scipy.ndimage
    #blurred = scipy.ndimage.gaussian_filter(arr, sigma=1.5, mode='nearest')
    #return arr * (1 - mix) + blurred * mix

def noise_reductor(config, initial_plasma, window, in_plasma):
    #T = in_plasma.shape[0]
    #N = int(sqrt(T))
    #assert N**2 == T
    #plasma = in_plasma.copy().reshape(N, N)

    ##sigma = 0.25 * config.h * config.noise_reductor_reach
    #offt_x = plasma['x'] - initial_plasma['x']
    #offt_y = plasma['y'] - initial_plasma['y']
    #plasma['x'] = blur(offt_x, window) + initial_plasma['x']
    #plasma['y'] = blur(offt_y, window) + initial_plasma['y']

    #plasma['p'][:, :, 0] = blur(plasma['p'][:, :, 0], window)
    #plasma['p'][:, :, 1] = blur(plasma['p'][:, :, 1], window)
    #plasma['p'][:, :, 2] = blur(plasma['p'][:, :, 2], window)

    #return plasma.reshape(T)

    #return noise_reductor_(config, in_plasma)
    return noise_reductor_3x3(config, in_plasma)


#@cython.boundscheck(False)
@cython.cdivision(True)
cpdef np.ndarray[plasma_particle.t] noise_reductor_3x3(PlasmaSolverConfig config,
                                                    np.ndarray[plasma_particle.t] in_plasma,
                                                    # np.ndarray[double, ndim=2] ro
                                                    ):
    cdef Py_ssize_t T = in_plasma.shape[0]
    cdef Py_ssize_t N = <int> sqrt(T)
    assert N**2 == T
    if N % 3:
        print('N', N)
    assert N % 3 == 0
    cdef np.ndarray[plasma_particle.t, ndim=2] plasma1 = in_plasma.reshape(N, N)
    # TODO: skip copying most of the stuff?
    cdef np.ndarray[plasma_particle.t, ndim=2] plasma2 = plasma1.copy()
    cdef np.ndarray[double, ndim=2] pxs_avg = np.zeros((N, N))
    cdef np.ndarray[double, ndim=2] pys_avg = np.zeros((N, N))
    cdef np.ndarray[double, ndim=2] pzs_avg = np.zeros((N, N))
    cdef double dp_friction, dp_friction_pz
    # TODO: allow noise reductor parameters to be specified as 2d arrays!
    cdef double friction_c = config.noise_reductor_friction / config.h  # empiric for now
    cdef double friction_c_pz = config.noise_reductor_friction_pz / config.h  # empiric for now
    cdef Py_ssize_t i, j
    cdef int ii, jj

    for i in prange(1, N - 1, 3, nogil=True, num_threads=config.threads):
        for j in range(1, N - 1, 3):
            for ii in range(-1, 1 + 1):
                for jj in range(-1, 1 + 1):
                    pxs_avg[i, j] += plasma1[i + ii, j + jj].p[1] / 9
                    pys_avg[i, j] += plasma1[i + ii, j + jj].p[2] / 9
                    pzs_avg[i, j] += plasma1[i + ii, j + jj].p[0] / 9
    for i in prange(1, N - 1, 3, nogil=True, num_threads=config.threads):
        for j in range(1, N - 1, 3):
            for ii in range(-1, 1 + 1):
                for jj in range(-1, 1 + 1):
                    plasma2[i + ii, j + jj].p[1] = plasma1[i + ii, j + jj].p[1] * (1 - friction_c) + pxs_avg[i, j] * friction_c
                    plasma2[i + ii, j + jj].p[2] = plasma1[i + ii, j + jj].p[2] * (1 - friction_c) + pys_avg[i, j] * friction_c
                    plasma2[i + ii, j + jj].p[0] = plasma1[i + ii, j + jj].p[0] * (1 - friction_c_pz) + pzs_avg[i, j] * friction_c_pz

    return plasma2.reshape(T)

@cython.boundscheck(False)
@cython.cdivision(True)
cpdef np.ndarray[plasma_particle.t] noise_reductor_(PlasmaSolverConfig config,
                                                    np.ndarray[plasma_particle.t] in_plasma,
                                                    # np.ndarray[double, ndim=2] ro
                                                    ):
    cdef Py_ssize_t T = in_plasma.shape[0]
    cdef Py_ssize_t N = <int> sqrt(T)
    assert N**2 == T
    cdef np.ndarray[plasma_particle.t, ndim=2] plasma1 = in_plasma.reshape(N, N)
    # TODO: skip copying most of the stuff?
    cdef np.ndarray[plasma_particle.t, ndim=2] plasma2 = plasma1.copy()
    cdef np.ndarray[plasma_particle.t, ndim=2] plasma3
    cdef double coord_deviation, p_m_deviation
    cdef double dp_friction, dp_friction_pz, dp_equalization
    # TODO: allow noise reductor parameters to be specified as 2d arrays!
    cdef double friction_c = config.noise_reductor_friction / config.h  # empiric for now
    cdef double friction_c_pz = config.noise_reductor_friction_pz / config.h  # empiric for now
    cdef double equalization_c = config.noise_reductor_equalization * config.h  # empiric for now
    cdef double reach = config.noise_reductor_reach * config.h  # empiric for now
    cdef Py_ssize_t i, j

    # pass in x direction, equalization
    for i in prange(1, N - 1, nogil=True, num_threads=config.threads):
        for j in range(N):
            coord_deviation = plasma1[i, j].x - (plasma1[i - 1, j].x + plasma1[i + 1, j].x) / 2
            if fabs(coord_deviation) < reach:
                dp_equalization = equalization_c * sin(pi * coord_deviation / reach)
                plasma2[i, j].p[1] -= config.h3 * dp_equalization

    # pass in y direction, equalization
    for i in cython.parallel.prange(N, nogil=True, num_threads=config.threads):
        for j in range(1, N - 1):
            coord_deviation = plasma1[i, j].y - (plasma1[i, j - 1].y + plasma1[i, j + 1].y) / 2
            if fabs(coord_deviation) < reach:
                dp_equalization = equalization_c * sin(pi * coord_deviation / reach)
                plasma2[i, j].p[2] -= config.h3 * dp_equalization

    plasma3 = plasma2.copy()

    # pass in x direction, friction
    for i in prange(1, N - 1, nogil=True, num_threads=config.threads):
        for j in range(N):
            coord_deviation = plasma2[i, j].x - (plasma2[i - 1, j].x + plasma2[i + 1, j].x) / 2
            if fabs(coord_deviation) < reach:
                p_m_deviation = (plasma2[i, j].p[1] / plasma2[i, j].m -
                                 (plasma2[i - 1, j].p[1] / plasma2[i - 1, j].m +
                                  plasma2[i + 1, j].p[1] / plasma2[i + 1, j].m) / 2)
                dp_friction = friction_c * plasma2[i, j].m * p_m_deviation
                plasma3[i, j].p[1] -= config.h3 * dp_friction

    # pass in y direction, friction
    for i in cython.parallel.prange(N, nogil=True, num_threads=config.threads):
        for j in range(1, N - 1):
            coord_deviation = plasma2[i, j].y - (plasma2[i, j - 1].y + plasma2[i, j + 1].y) / 2
            if fabs(coord_deviation) < reach:
                p_m_deviation = (plasma2[i, j].p[2] / plasma2[i, j].m -
                                 (plasma2[i, j - 1].p[2] / plasma2[i, j - 1].m +
                                  plasma2[i, j + 1].p[2] / plasma2[i, j + 1].m) / 2)
                dp_friction = friction_c * plasma2[i, j].m * p_m_deviation
                plasma3[i, j].p[2] -= config.h3 * dp_friction

    # pass in x and y, friction in pz
    for i in prange(1, N - 1, nogil=True, num_threads=config.threads):
        for j in range(1, N - 1):
            p_m_deviation = (plasma2[i, j].p[0] / plasma2[i, j].m -
                             (plasma2[i - 1, j].p[0] / plasma2[i - 1, j].m +
                              plasma2[i + 1, j].p[0] / plasma2[i + 1, j].m +
                              plasma2[i, j - 1].p[0] / plasma2[i, j - 1].m +
                              plasma2[i, j + 1].p[0] / plasma2[i, j + 1].m) / 4)
            dp_friction_pz = friction_c_pz * plasma2[i, j].m * p_m_deviation
            plasma3[i, j].p[0] -= config.h3 * dp_friction_pz

    return plasma3.reshape(T)


### The main plot, written by K. V. Lotov


cdef class PlasmaSolver:
    cdef public FieldSolver field_solver
    cdef public object RoJ_dtype
    cdef public float[:] error_lut
    cdef public float[:] error_lut_diag
    cdef public object ion_initial_ro
    cdef public object initial_plasma
    cdef public object window
    # TODO: allocate everything else to make the solver allocation-free

    def __init__(self, config):
        # TODO: incapsulate PlasmaSolverConfig creation here?
        self.field_solver = FieldSolver(config.grid_steps,
                                        config.window_width / config.grid_steps,
                                        config.field_solver_subtraction_trick,
                                        config.openmp_limit_threads)
        self.RoJ_dtype = RoJ_dtype

    def PlasmaSolverConfig(self, config):
        return PlasmaSolverConfig(config)

    cpdef response(self,
                   config, xi_i, in_plasma, in_plasma_cor,
                   beam_ro, roj_pprv, roj_prev,
                   mut_Ex, mut_Ey, mut_Ez, mut_Bx, mut_By, mut_Bz,
                   out_plasma, out_plasma_cor, out_roj
                   ):
        plasma = in_plasma.copy()
        #config.noise_reductor_enable = (xi_i % 5 == 0)
        noise_reductor_predictions = config.noise_reductor_enable and not config.noise_reductor_final_only

        Fl = mut_Ex.copy(), mut_Ey.copy(), mut_Ez.copy(), mut_Bx.copy(), mut_By.copy(), mut_Bz.copy()

        if xi_i == 0:
            self.ion_initial_ro = -gpu_functions.deposit(config, plasma, 0)['ro']
            T = plasma.shape[0]
            N = int(sqrt(T))
            assert N**2 == T
            self.initial_plasma = plasma.copy().reshape(N, N)

            #self.window = scipy.signal.windows.tukey(N, .75)
            #PAD = int(N * .05) // 2
            PAD = int(N * .05) // 2
            #self.window = scipy.signal.windows.tukey(N - 2 * PAD, .75)
            self.window = scipy.signal.windows.tukey(N - 2 * PAD, .75)
            self.window = np.hstack([np.zeros(PAD), self.window, np.zeros(PAD)])
            self.window = np.fft.ifftshift(self.window[:, None] * self.window[None, :])


        # ===  1  ===
        plasma_predicted_half1 = move_simple_fast(config, plasma, config.h3 / 2)
        hs_xs, hs_ys = plasma_predicted_half1['x'], plasma_predicted_half1['y']
        #Exs, Eys, Ezs, Bxs, Bys, Bzs = interpolate_fields(config, hs_xs, hs_ys, *Fl)
        #plasma_1 = move_smart_fast(config, plasma, Exs, Eys, Ezs, Bxs, Bys, Bzs)
        Fls = interpolate_fields(config, hs_xs, hs_ys, *Fl)
        plasma_1 = move_smart_fast(config, plasma, *Fls, self.initial_plasma, self.window,
                                   noise_reductor_enable=noise_reductor_predictions)
        roj_1 = gpu_functions.deposit(config, plasma_1, self.ion_initial_ro)

        hs_xs = (plasma['x'] + plasma_1['x']) / 2
        hs_ys = (plasma['y'] + plasma_1['y']) / 2
        # ===  2  ===  + hs_xs, hs_ys, roj_1
        Fl_pred = calculate_fields(config, self.field_solver, roj_1, roj_prev,
                                   *Fl, beam_ro, config.variant_A_predictor)

        # ===  3  ===  + hs_xs, hs_ys, Fl_pred
        Fl_avg_1 = average_fields(Fl, Fl_pred)
        Fls_avg_1 = interpolate_fields(config, hs_xs, hs_ys, *Fl_avg_1)
        #Fls_avg_1 = interpolate_averaged_fields(config, hs_xs, hs_ys, *Fl, *Fl_pred)
        plasma_2 = move_smart_fast(config, plasma, *Fls_avg_1, self.initial_plasma, self.window,
                                   noise_reductor_enable=noise_reductor_predictions)
        roj_2 = gpu_functions.deposit(config, plasma_2, self.ion_initial_ro)

        hs_xs = (plasma['x'] + plasma_2['x']) / 2
        hs_ys = (plasma['y'] + plasma_2['y']) / 2
        # ===  4  ===  + hs_xs, hs_ys, roj_2, Fl_avg_1
        Fl_new = calculate_fields(config, self.field_solver, roj_2, roj_prev,
                                  *Fl_avg_1, beam_ro, config.variant_A_corrector)

        # ===  5  ===  + hs_xs, hs_ys, Fl_new
        Fls_avg_2 = interpolate_averaged_fields(config, hs_xs, hs_ys, *Fl, *Fl_new)
        plasma_new = move_smart_fast(config, plasma, *Fls_avg_2, self.initial_plasma, self.window,
                                     noise_reductor_enable=config.noise_reductor_enable)
        roj_new = gpu_functions.deposit(config, plasma_new, self.ion_initial_ro)

        out_plasma[...] = plasma_new
        out_plasma_cor[...] = plasma_new
        out_roj[...] = roj_new
        mut_Ex[...], mut_Ey[...], mut_Ez[...], mut_Bx[...], mut_By[...], mut_Bz[...] = Fl_new


# TODO: merge interpolate-move-deposit into a single routine?
# TODO: -||-, rewrite it in Cython and tune it up
