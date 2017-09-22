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
Beam mover for LCODE.
Updates particle momenta and positions using field values.
Primary author: A. P. Sosedkin <A.P.Sosedkin@inp.nsk.su>
Secondary author: I. A. Shalimova <ias@osmf.sscc.ru>
'''

from libc.math cimport sqrt, ceil

import numpy as np
cimport numpy as np

from . import beam_particle
from . cimport beam_particle


cdef inline beam_particle.t move_particle(beam_particle.t p,
                                          double dt,
                                          np.ndarray[double, ndim=2] Ex,
                                          np.ndarray[double, ndim=2] Ey,
                                          np.ndarray[double, ndim=2] Ez,
                                          np.ndarray[double, ndim=2] Bx,
                                          np.ndarray[double, ndim=2] By,
                                          np.ndarray[double, ndim=2] Bz,
                                          double x_max,
                                          double h,
                                          int p_avg_iter=1,
                                          ):
    cdef double gamma_m, half_x, half_y  # half_xi
    cdef double Ex_p, Ey_p, Ez_p, Bx_p, By_p, Bz_p

    gamma_m = sqrt(p.p[0]**2 + p.p[1]**2 + p.p[2]**2 + p.m**2)

    # Calculate half-moved predicted position

    half_x = p.r[1] + dt / 2 * (p.p[1] / gamma_m)
    half_y = p.r[2] + dt / 2 * (p.p[2] / gamma_m)
    # half_xi = p.p[0] + dt / 2 * (p.p[0] / gamma_m - 1)
    half_x = min(max(half_x, -x_max), x_max)
    half_y = min(max(half_y, -x_max), x_max)

    # Interpolate fields at half-moved predicted position

    cdef unsigned int i, j
    cdef double dxqk, dyqk

    cdef double h2_reciprocal = 1 / h**2

    while True:
        i = <unsigned int> ((x_max + half_x + h * .5) / h)
        j = <unsigned int> ((x_max + half_y + h * .5) / h)

        dxqk = x_max + half_x - h * (i - .5)
        dyqk = x_max + half_y - h * (j - .5)

        if i == 0:
            if j == 0:
                % for Fl in 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz':
                ${Fl}_p = ${Fl}[0, 0]
                % endfor
                break

            if j == Ex.shape[1]:
                % for Fl in 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz':
                ${Fl}_p = ${Fl}[0, -1]
                % endfor
                break

            % for Fl in 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz':
            ${Fl}_p = (${Fl}[0, j - 1] +
                       (${Fl}[0, j] - ${Fl}[0, j - 1]) * dyqk / h)
            % endfor
            break

        if i == Ex.shape[0]:
            if j == 0:
                % for Fl in 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz':
                ${Fl}_p = ${Fl}[-1, 0]
                % endfor
                break

            if j == Ex.shape[1]:
                % for Fl in 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz':
                ${Fl}_p = ${Fl}[-1, -1]
                % endfor
                break

            % for Fl in 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz':
            ${Fl}_p = (${Fl}[-1, j - 1] +
                       (${Fl}[-1, j] - ${Fl}[-1, j - 1]) * dyqk / h)
            % endfor
            break

        if j == 0:
            % for Fl in 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz':
            ${Fl}_p = (${Fl}[i - 1, 0] +
                       (${Fl}[i, 0] - ${Fl}[i - 1, 0]) * dxqk / h)
            % endfor
            break

        if j == Ex.shape[1]:
            % for Fl in 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz':
            ${Fl}_p = (${Fl}[i - 1, -1] +
                       (${Fl}[i, -1] - ${Fl}[i - 1, -1]) * dxqk / h)
            % endfor
            break

        % for Fl in 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz':
        ${Fl}_p = h2_reciprocal * (  # noqa: E201
            ${Fl}[i + 0, j + 0] * dxqk * dyqk +  # noqa: E222
            ${Fl}[i - 1, j + 0] * (h - dxqk) * dyqk +  # noqa: E222
            ${Fl}[i + 0, j - 1] * dxqk * (h - dyqk) +  # noqa: E222
            ${Fl}[i - 1, j - 1] * (h - dxqk) * (h - dyqk)
        )
        % endfor
        break

    # Recalculate momenta

    cdef double p1_avg = p.p[1], p2_avg = p.p[2], p0_avg = p.p[0]
    cdef double new_p1, new_p2, new_p0
    cdef int approx_i

    for approx_i in range(p_avg_iter):
        # dpb / dt = q*E + q * [v x B]  (see 2D manual)
        # Ex + vy Bz - vz By
        gamma_m = sqrt(p0_avg**2 + p1_avg**2 + p2_avg**2 + p.m**2)
        new_p1 = p.p[1] + p.q * dt * (
            Ex_p + Bz_p * p2_avg / gamma_m - By_p * p0_avg / gamma_m
        )
        # Ey + vz Bx - vx Bz
        new_p2 = p.p[2] + p.q * dt * (
            Ey_p + Bx_p * p0_avg / gamma_m - Bz_p * p1_avg / gamma_m
        )
        # Ez + vx By - vy Bx
        new_p0 = p.p[0] + p.q * dt * (
            Ez_p + By_p * p1_avg / gamma_m - Bx_p * p2_avg / gamma_m
        )
        p1_avg = (p.p[1] + new_p1) / 2
        p2_avg = (p.p[2] + new_p2) / 2
        p0_avg = (p.p[0] + new_p0) / 2

    # Move particle

    gamma_m = sqrt(p0_avg**2 + p1_avg**2 + p2_avg**2 + p.m**2)

    p.r[0] += dt * (p0_avg / gamma_m - 1)
    p.r[1] += dt * (p1_avg / gamma_m)
    p.r[2] += dt * (p2_avg / gamma_m)
    # assert dt * (p0_avg / gamma_m - 1) <= 0
    # assert p.r[0] <= xi

    p.p[0] = new_p0
    p.p[1] = new_p1
    p.p[2] = new_p2

    p.t += dt

    return p


cpdef move(config,
           np.ndarray[beam_particle.t] particles,
           double t,
           double xi,
           np.ndarray[double, ndim=2] Ex,
           np.ndarray[double, ndim=2] Ey,
           np.ndarray[double, ndim=2] Ez,
           np.ndarray[double, ndim=2] Bx,
           np.ndarray[double, ndim=2] By,
           np.ndarray[double, ndim=2] Bz,
           ):
    cdef long k
    cdef beam_particle.t p, half_moved_p

    cdef double x_max = config.window_width / 2
    cdef double x_step_size = config.window_width / config.grid_steps
    cdef double base_time_step_size = config.time_step_size
    cdef double xi_step_size = config.xi_step_size

    cdef double approx_step_fraction
    cdef unsigned int substepping_depth, total_substeps = 0
    cdef double dt, target_dt, target_t = t + base_time_step_size
    cdef unsigned int sub_i
    cdef int p_iterations = config.beam_mover_p_iterations
    cdef unsigned int max_substepping = config.beam_mover_max_substepping
    cdef double substepping_trigger = config.beam_mover_substepping_trigger
    cdef int written

    cdef np.ndarray[beam_particle.t] moved = np.zeros_like(particles)
    cdef np.ndarray[beam_particle.t] fell = np.zeros_like(particles)
    cdef np.ndarray[beam_particle.t] lost = np.zeros_like(particles)

    cdef unsigned long moved_i = 0, fell_i = 0, lost_i = 0

    # indexed for performance
    for k in range(particles.shape[0]):
        p = particles[k]
        written = 0

        if p.t > t + base_time_step_size / 2:  # TODO: think it out more
            moved[moved_i] = p
            moved_i += 1
            continue  # a delayed particle, should not move for p.t

        target_dt = dt = target_t - p.t
        if target_dt == 0:
            moved[moved_i] = p
            moved_i += 1
            continue
        assert target_dt > 0

        half_moved_p = move_particle(p, dt / 2,
                                     Ex, Ey, Ez, Bx, By, Bz,
                                     x_max, x_step_size, p_avg_iter=1)

        approx_step_fraction = 2 * max(
            abs(half_moved_p.r[0] - p.r[0]) / xi_step_size,
            abs(half_moved_p.r[1] - p.r[1]) / x_step_size,
            abs(half_moved_p.r[2] - p.r[2]) / x_step_size
        )
        substepping_depth = <unsigned int> ceil(approx_step_fraction /
                                                substepping_trigger)
        substepping_depth = min(substepping_depth, max_substepping)
        dt = target_dt / substepping_depth

        for sub_i in range(substepping_depth):
            p = move_particle(p, dt,
                              Ex, Ey, Ez, Bx, By, Bz,
                              x_max, x_step_size,
                              p_avg_iter=p_iterations)

            total_substeps += 1

            if (abs(p.r[1]) > x_max or abs(p.r[2]) > x_max):  # escaped in xy
                lost[lost_i] = p
                lost_i += 1
                written = 1
                break
            elif p.r[0] < xi - xi_step_size:  # fell behind in xi to next level
                assert p.r[0] > xi - xi_step_size * 2  # but only one level!
                fell[fell_i] = p
                fell_i += 1
                written = 1
                break

        if not written:
            moved[moved_i] = p
            moved_i += 1

    # Return
    moved = moved[:moved_i].copy()
    moved = moved[moved['r'][:, 0].argsort()]
    fell = fell[:fell_i].copy()
    fell = fell[fell['r'][:, 0].argsort()]
    lost = lost[:lost_i].copy()
    lost = lost[lost['r'][:, 0].argsort()]
    return moved, fell, lost, total_substeps


# Plan: write "naive move", use with t/2, get half/moved particles
#       get fields at these positions, proceed to here
#       implement 'delaying' slow particles there
# Also: fields interpolation in xi

# Substepping criterion: do not cross more than 1 cell in each direction.
# It's required for correctness of these formulas.

# Move interpolation invocation here?
# Both in xy and in xi... that's gotta be a lot of code

# Group arrays into E and B?.. or not (interpolation purposes)?

# Move macro-degree out of charge?..
