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


from libc.math cimport sqrt, log2, sin, pi

import numpy as np
cimport numpy as np


### Routines written by I. A. Shalimova, adapted by A. P. Sosedkin


from .. import plasma_particle
from .. cimport plasma_particle

from .field_solver import ThreadLocalStorage
from .field_solver import Neuman_red, reduction_Dirichlet1, Posson_reduct_12


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
    cdef public double noise_reductor_reach
    cdef public double noise_reductor_final_only

    def __init__(self, global_config):
        self.npq, unwanted = divmod(log2(global_config.grid_steps - 1), 1)
        if unwanted:
            raise RuntimeError('Grid step must be N**2 + 1')
        self.n_dim = 2**self.npq + 1
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
        self.noise_reductor_reach = global_config.noise_reductor_reach
        self.noise_reductor_final_only = global_config.noise_reductor_final_only

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
    cdef unsigned int i1, j1
    cdef double x_loc, y_loc
    cdef double fx1, fy1, fx2, fy2, fx3, fy3

    #assert Ex.shape[0] == Ex.shape[1] == config.n_dim

    # indexed for performance
    for k in range(xs.shape[0]):
        #assert -config.particle_boundary < xs[k] < config.particle_boundary
        #assert -config.particle_boundary < ys[k] < config.particle_boundary

        i1 = <unsigned int> ((config.x_max + xs[k]) / config.h)
        j1 = <unsigned int> ((config.x_max + ys[k]) / config.h)
        #assert 0 < i1 < Ex.shape[0] - 1
        #assert 0 < j1 < Ex.shape[0] - 1

        x_loc = config.x_max + xs[k] - i1 * config.h - .5 * config.h
        y_loc = config.x_max + ys[k] - j1 * config.h - .5 * config.h
        #assert 0 <= x_loc < 1
        #assert 0 <= x_loc < 1

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


cpdef void ro_and_j_ie_Vshivkov(PlasmaSolverConfig config,
                                plasma_particle.t[:] plasma_particles,
                                # n_dim, n_dim
                                np.ndarray[RoJ_t, ndim=2] roj,
                                ):
    cdef unsigned int i, j
    cdef int q
    cdef long k
    cdef double x_loc, y_loc
    cdef double fx1, fy1, fx2, fy2, fx3, fy3
    cdef double dro, djx, djy, djz
    cdef double gamma_m
    cdef plasma_particle.t p

    cdef unsigned int n_dim = roj.shape[0]
    #assert roj.shape[0] == roj.shape[1]

    roj[...] = 0

    # for p in plasma_particles: indexed for performance
    for k in range(plasma_particles.shape[0]):
        p = plasma_particles[k]
        gamma_m = sqrt(p.m**2 + p.p[0]**2 + p.p[1]**2 + p.p[2]**2)
        dro = p.q / (1 - p.p[0] / gamma_m)
        djz = p.p[0] * (dro / gamma_m)
        djx = p.p[1] * (dro / gamma_m)
        djy = p.p[2] * (dro / gamma_m)

        # particle indices in roj and adge arrays
        i = <unsigned int> ((config.x_max + p.x) / config.h)
        j = <unsigned int> ((config.x_max + p.y) / config.h)
        #assert 0 < i < n_dim - 1 and 0 < j < n_dim - 1

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


### Convenience Python wrappers above them; TODO: get rid of

def interpolate_fields(config, xs, ys, Ex, Ey, Ez, Bx, By, Bz):
    Exs = np.zeros_like(xs)
    Eys = np.zeros_like(xs)
    Ezs = np.zeros_like(xs)
    Bxs = np.zeros_like(xs)
    Bys = np.zeros_like(xs)
    Bzs = np.zeros_like(xs)
    interpolate_fields_fs9(config, xs, ys,
                           Ex, Ey, Ez, Bx, By, Bz,
                           Exs, Eys, Ezs, Bxs, Bys, Bzs)
    # may assert that particles are contained in +- particle_boundary
    # and remove a lot of inner ifs
    return Exs, Eys, Ezs, Bxs, Bys, Bzs


def deposit(config, plasma):
    plasma_virtualized = config.virtualize(plasma)

    roj = np.zeros((config.n_dim, config.n_dim), dtype=RoJ_dtype)
    ro_and_j_ie_Vshivkov(config, plasma_virtualized, roj)
    # may assert that particles are contained in +- particle_boundary
    # and remove a lot of inner ifs
    return roj


def calculate_fields(config, roj_cur, roj_prev, Ex_, Ey_, Ez_, Bx_, By_, Bz_, beam_ro, variant_A=False):
    tls = ThreadLocalStorage(config.n_dim)

    Ex, Ey, Ez = Ex_.copy(), Ey_.copy(), Ez_.copy()
    Bx, By, Bz = Bx_.copy(), By_.copy(), Bz_.copy()
    roj_edges = np.zeros((4, 2, config.n_dim), dtype=RoJ_dtype)

    if variant_A:
        roj = np.zeros_like(roj_cur)
        for comp in 'ro', 'jx', 'jy', 'jz':
            roj[comp] = (roj_cur[comp] + roj_prev[comp]) / 2
    else:
        roj = roj_cur.copy()

    djx_dx, djx_dy = np.gradient(roj['jx'], config.h, config.h, edge_order=2)
    djy_dx, djy_dy = np.gradient(roj['jy'], config.h, config.h, edge_order=2)
    djz_dx, djz_dy = np.gradient(roj['jz'], config.h, config.h, edge_order=2)
    dro_dx, dro_dy = np.gradient(roj['ro'], config.h, config.h, edge_order=2)
    beam_ro_dx, beam_ro_dy = np.gradient(beam_ro, config.h, config.h,
                                         edge_order=2)
    djx_dxi = (roj_prev['jx'] - roj_cur['jx']) / config.h3
    djy_dxi = (roj_prev['jy'] - roj_cur['jy']) / config.h3

    # Field Ez
    reduction_Dirichlet1(-(djx_dx + djy_dy), Ez,
                         tls, config.n_dim, config.h, config.npq)

    # Field Bz
    Neuman_red(config.B_0,
               -roj[0, :]['jy'], roj[-1, :]['jy'],
               roj[:, 0]['jx'], -roj[:, -1]['jx'],
               -(djx_dy - djy_dx), Bz,
               tls, config.n_dim, config.h, config.npq, config.x_max)

    # Field Ex
    Posson_reduct_12(-(beam_ro[0] + roj['ro'][0]),
                     +(beam_ro[-1] + roj['ro'][-1]),
                     -(beam_ro_dx + dro_dx - djx_dxi) + Ex, Ex,
                     tls, config.n_dim, config.h, config.npq)

    # Field Ey
    Posson_reduct_12(-(beam_ro[:, 0] + roj['ro'][:, 0]),
                     +(beam_ro[:, -1] + roj['ro'][:, -1]),
                     (-(beam_ro_dy + dro_dy - djy_dxi) + Ey).T, Ey.T,
                     tls, config.n_dim, config.h, config.npq)

    # Field Bx
    Posson_reduct_12(-(beam_ro[:, 0] + roj['jz'][:, 0]),
                     beam_ro[:, -1] + roj['jz'][:, -1],
                     +((beam_ro_dy + djz_dy - djy_dxi) + Bx).T, Bx.T,
                     tls, config.n_dim, config.h, config.npq)

    # Field By
    Posson_reduct_12(beam_ro[0] + roj['jz'][0],
                     -(beam_ro[-1] + roj['jz'][-1]),
                     -(beam_ro_dx + djz_dx - djx_dxi) + By, By,  # !!!
                     tls, config.n_dim, config.h, config.npq)

    if variant_A:
        # ??? VAR A/B? -- shove inside calculate_fields (E_new = 2 * E_result - E_input)
        Ex = 2 * Ex - Ex_
        Ey = 2 * Ey - Ey_
        Ez = 2 * Ez - Ez_
        Bx = 2 * Bx - Bx_
        By = 2 * By - By_
        Bz = 2 * Bz - Bz_

    return Ex, Ey, Ez, Bx, By, Bz


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
    for k in range(plasma_particles.shape[0]):
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
        if p.x < -config.particle_boundary:
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
    cdef double gamma_m, dpx, dpy, dpz, px, py, pz, vx, vy, vz, factor1
    cdef plasma_particle.t p

    # for p in plasma_particles: indexed for performance
    for k in range(plasma_particles.shape[0]):
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
        if p.x < -config.particle_boundary:
            p.y = -2 * config.particle_boundary - p.y
            p.p[2] *= -1

        out_plasma[k] = p


def move_smart_fast(config, plasma, Exs, Eys, Ezs, Bxs, Bys, Bzs, noise_reductor_enable=False):
    out_plasma = np.empty_like(plasma)
    move_smart_fast_(config, plasma, Exs, Eys, Ezs, Bxs, Bys, Bzs, out_plasma)
    # TODO: call noisereductor only on final movement or on all movements?
    if noise_reductor_enable:
        out_plasma = noise_reductor(config, out_plasma)
    return out_plasma


### Noise reductor draft

def noise_reductor(config, plasma):  #, ro
    plasma = plasma.copy()
    plasma[::2] = noise_reductor_(config, plasma[::2])    # ions
    plasma[1::2] = noise_reductor_(config, plasma[1::2])  # electrons
    return plasma


cpdef np.ndarray[plasma_particle.t] noise_reductor_(PlasmaSolverConfig config,
                                                   np.ndarray[plasma_particle.t] in_plasma,
                                                   # np.ndarray[double, ndim=2] ro
                                                   ):
    cdef long T = in_plasma.shape[0]
    cdef int N = <int> sqrt(T)
    cdef np.ndarray[plasma_particle.t, ndim=2] plasma = in_plasma.copy().reshape(N, N)
    cdef plasma_particle.t neighbor1, neighbor2
    cdef double coord_deviation, p_m_deviation
    cdef double dp_friction, dp_equalization
    # TODO: allow noise reductor parameters to be specified as 2d arrays!
    cdef double friction_c = config.noise_reductor_friction * config.h  # empiric for now
    cdef double equalization_c = config.noise_reductor_equalization / config.h  # empiric for now
    cdef double reach = config.noise_reductor_reach * config.h  # empiric for now
    cdef int i, j

    # pass in x direction
    for i in range(1, N - 1):
        for j in range(N):
            coord_deviation = plasma[i, j].x - (plasma[i - 1, j].x + plasma[i + 1, j].x) / 2
            if coord_deviation < config.noise_reductor_reach:
                p_m_deviation = (plasma[i, j].p[1] / plasma[i, j].m -
                                 (plasma[i - 1, j].p[1] / plasma[i - 1, j].m +
                                  plasma[i + 1, j].p[1] / plasma[i + 1, j].m) / 2)
                dp_equalization = equalization_c * sin(pi * reach * coord_deviation)
                dp_friction = friction_c * plasma[i, j].m * p_m_deviation
                plasma[i, j].p[1] -= config.h3 * (dp_friction + dp_equalization)

    # pass in y direction
    for i in range(N):
        for j in range(1, N - 1):
            coord_deviation = plasma[i, j].y - (plasma[i, j - 1].y + plasma[i, j + 1].y) / 2
            if coord_deviation < config.noise_reductor_reach:
                p_m_deviation = (plasma[i, j].p[2] / plasma[i, j].m -
                                 (plasma[i, j - 1].p[2] / plasma[i, j - 1].m +
                                  plasma[i, j + 1].p[2] / plasma[i, j + 1].m) / 2)
                dp_equalization = equalization_c * sin(pi * reach * coord_deviation)
                dp_friction = friction_c * plasma[i, j].m * p_m_deviation
                plasma[i, j].p[2] -= config.h3 * (dp_friction + dp_equalization)


    return plasma.reshape(T)


### The main plot, written by K. V. Lotov


cpdef response(config, xi_i, in_plasma, in_plasma_cor,
               beam_ro, roj_pprv, roj_prev,
               mut_Ex, mut_Ey, mut_Ez, mut_Bx, mut_By, mut_Bz,
               out_plasma, out_plasma_cor, out_roj
               ):
    plasma = in_plasma.copy()
    noise_reductor_predictions = config.noise_reductor_enable and not config.noise_reductor_final_only

    Fl = mut_Ex.copy(), mut_Ey.copy(), mut_Ez.copy(), mut_Bx.copy(), mut_By.copy(), mut_Bz.copy()

    # ===  1  ===
    plasma_predicted_half1 = move_simple_fast(config, plasma, config.h3 / 2)
    hs_xs, hs_ys = plasma_predicted_half1['x'], plasma_predicted_half1['y']
    #Exs, Eys, Ezs, Bxs, Bys, Bzs = interpolate_fields(config, hs_xs, hs_ys, *Fl)
    #plasma_1 = move_smart_fast(config, plasma, Exs, Eys, Ezs, Bxs, Bys, Bzs)
    Fls = interpolate_fields(config, hs_xs, hs_ys, *Fl)
    plasma_1 = move_smart_fast(config, plasma, *Fls,
                               noise_reductor_enable=noise_reductor_predictions)
    roj_1 = deposit(config, plasma_1)

    # ===  2  ===  + hs_xs, hs_ys, roj_1
    Fl_pred = calculate_fields(config, roj_1, roj_prev, *Fl, beam_ro, config.variant_A_predictor)

    # ===  3  ===  + hs_xs, hs_ys, Fl_pred
    Fl_avg_1 = average_fields(Fl, Fl_pred)
    Fls_avg_1 = interpolate_fields(config, hs_xs, hs_ys, *Fl_avg_1)
    plasma_2 = move_smart_fast(config, plasma, *Fls_avg_1,
                               noise_reductor_enable=noise_reductor_predictions)
    roj_2 = deposit(config, plasma_2)

    # ===  4  ===  + hs_xs, hs_ys, roj_2, Fl_avg_1
    Fl_new = calculate_fields(config, roj_2, roj_prev, *Fl_avg_1, beam_ro, config.variant_A_corrector)

    # ===  5  ===  + hs_xs, hs_ys, Fl_new
    Fl_avg_2 = average_fields(Fl, Fl_new)
    Fls_avg_2 = interpolate_fields(config, hs_xs, hs_ys, *Fl_avg_2)
    plasma_new = move_smart_fast(config, plasma, *Fls_avg_2,
                                 noise_reductor_enable=config.noise_reductor_enable)
    roj_new = deposit(config, plasma_new)

    #test_particle = plasma[plasma['q'] < 0]
    #test_particle = test_particle[np.abs(test_particle['x']) < 0.5]
    #test_particle = test_particle[np.abs(test_particle['y']) < 0.5]
    #test_particle = test_particle[len(test_particle) // 3]
    #print('ys', test_particle['p'])
    #print('xy', test_particle['x'], test_particle['y'])
    #print('m', test_particle['m'])
    #print('q', test_particle['q'])
    #print('gamma_m', gamma_m(np.array([test_particle])))

    out_plasma[...] = plasma_new
    out_plasma_cor[...] = plasma_new
    out_roj[...] = roj_new
    #print(beam_ro.max(), ['%+7e' % fl.ptp() for fl in Fl_new])
    mut_Ex[...], mut_Ey[...], mut_Ez[...], mut_Bx[...], mut_By[...], mut_Bz[...] = Fl_new


# TODO: merge interpolate-move-deposit into a single routine?
# TODO: -||-, rewrite it in Cython and tune it up
