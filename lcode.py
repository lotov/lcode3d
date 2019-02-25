#!/usr/bin/env python3

# Copyright (c) 2016-2019 LCODE team <team@lcode.info>.

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


from math import sqrt, floor

import sys
import os

import matplotlib.pyplot as plt

import numpy as np

import numba
import numba.cuda

import cupy

import scipy.ndimage
import scipy.signal


os.environ['OMP_NUM_THREADS'] = '1'


USUAL_ELECTRON_CHARGE = -1
USUAL_ELECTRON_MASS = 1
USUAL_ION_CHARGE = 1
USUAL_ION_MASS = 1836.152674 * 85.4678


# TODO: macrosity
plasma_particle_dtype = np.dtype([
    ('v', np.double, (3,)),
    ('p', np.double, (3,)),  # TODO: internal to move_particles, do not store
    ('N', np.long),
    ('x_init', np.double),
    ('y_init', np.double),
    ('x_offt', np.double),
    ('y_offt', np.double),
    ('q', np.double),
    ('m', np.double),
], align=False)


@numba.cuda.jit
def zerofill_kernel(arr1d):
    index = numba.cuda.grid(1)
    stride = numba.cuda.blockDim.x * numba.cuda.gridDim.x
    for k in range(index, arr1d.size, stride):
        arr1d[k] = 0


@numba.cuda.jit
def move_predict_halfstep_kernel(xi_step_size, reflect_boundary, ms,
                                 x_init, y_init, old_x_offt, old_y_offt,
                                 pxs, pys, pzs,
                                 halfstep_x_offt, halfstep_y_offt):
    index = numba.cuda.grid(1)
    stride = numba.cuda.blockDim.x * numba.cuda.gridDim.x
    for k in range(index, ms.size, stride):
        m = ms[k]
        x, y = x_init[k] + old_x_offt[k], y_init[k] + old_y_offt[k]
        px, py, pz = pxs[k], pys[k], pzs[k]

        gamma_m = sqrt(m**2 + pz**2 + px**2 + py**2)

        x += px / (gamma_m - pz) * (xi_step_size / 2)
        y += py / (gamma_m - pz) * (xi_step_size / 2)

        # TODO: avoid branching?
        x = x if x <= +reflect_boundary else +2 * reflect_boundary - x
        x = x if x >= -reflect_boundary else -2 * reflect_boundary - x
        y = y if y <= +reflect_boundary else +2 * reflect_boundary - y
        y = y if y >= -reflect_boundary else -2 * reflect_boundary - y

        halfstep_x_offt[k], halfstep_y_offt[k] = x - x_init[k], y - y_init[k]


# TODO: write a version fused with averaging
# TODO: fuse with moving?
@numba.cuda.jit
def interpolate_kernel(x_init, y_init, x_offt, y_offt, Ex, Ey, Ez, Bx, By, Bz,
                       grid_step_size, grid_steps,
                       Exs, Eys, Ezs, Bxs, Bys, Bzs):
    index = numba.cuda.grid(1)
    stride = numba.cuda.blockDim.x * numba.cuda.gridDim.x
    for k in range(index, x_init.size, stride):
        x_h = (x_init[k] + x_offt[k]) / grid_step_size + .5
        y_h = (y_init[k] + y_offt[k]) / grid_step_size + .5
        i = int(floor(x_h) + grid_steps // 2)
        j = int(floor(y_h) + grid_steps // 2)
        x_loc = x_h - floor(x_h) - .5  # centered to -.5 to 5, not 0 to 1 because
        y_loc = y_h - floor(y_h) - .5  # the latter formulas use offset from cell center

        fx1 = .75 - x_loc**2
        fy1 = .75 - y_loc**2
        fx2 = .5 + x_loc
        fy2 = .5 + y_loc
        fx3 = .5 - x_loc
        fy3 = .5 - y_loc

        # TODO: use the same names in deposition?
        w00 = fx1 * fy1
        wP0 = fx2**2 * (fy1 / 2)
        w0P = fy2**2 * (fx1 / 2)
        wPP = fx2**2 * (fy2**2 / 4)
        wM0 = fx3**2 * (fy1 / 2)
        w0M = fy3**2 * (fx1 / 2)
        wMM = fx3**2 * (fy3**2 / 4)
        wMP = fx3**2 * (fy2**2 / 4)
        wPM = fx2**2 * (fy3**2 / 4)

        # wx0, wy0 = .75 - x_loc**2, .75 - y_loc**2  # fx1, fy1
        # wxP, wyP = (.5 + x_loc)**2 / 2, (.5 + y_loc)**2 / 2 # fx2**2 / 2, fy2**2 / 2
        # wxM, wyM = (.5 - x_loc)**2 / 2, (.5 - y_loc)**2 / 2 # fx3**2 / 2, fy3**2 / 2

        Exs[k] = (
            Ex[i + 0, j + 0] * w00 +
            Ex[i + 1, j + 0] * wP0 +
            Ex[i + 0, j + 1] * w0P +
            Ex[i + 1, j + 1] * wPP +
            Ex[i - 1, j + 0] * wM0 +
            Ex[i + 0, j - 1] * w0M +
            Ex[i - 1, j - 1] * wMM +
            Ex[i - 1, j + 1] * wMP +
            Ex[i + 1, j - 1] * wPM
        )

        Eys[k] = (
            Ey[i + 0, j + 0] * w00 +
            Ey[i + 1, j + 0] * wP0 +
            Ey[i + 0, j + 1] * w0P +
            Ey[i + 1, j + 1] * wPP +
            Ey[i - 1, j + 0] * wM0 +
            Ey[i + 0, j - 1] * w0M +
            Ey[i - 1, j - 1] * wMM +
            Ey[i - 1, j + 1] * wMP +
            Ey[i + 1, j - 1] * wPM
        )

        Ezs[k] = (
            Ez[i + 0, j + 0] * w00 +
            Ez[i + 1, j + 0] * wP0 +
            Ez[i + 0, j + 1] * w0P +
            Ez[i + 1, j + 1] * wPP +
            Ez[i - 1, j + 0] * wM0 +
            Ez[i + 0, j - 1] * w0M +
            Ez[i - 1, j - 1] * wMM +
            Ez[i - 1, j + 1] * wMP +
            Ez[i + 1, j - 1] * wPM
        )

        Bxs[k] = (
            Bx[i + 0, j + 0] * w00 +
            Bx[i + 1, j + 0] * wP0 +
            Bx[i + 0, j + 1] * w0P +
            Bx[i + 1, j + 1] * wPP +
            Bx[i - 1, j + 0] * wM0 +
            Bx[i + 0, j - 1] * w0M +
            Bx[i - 1, j - 1] * wMM +
            Bx[i - 1, j + 1] * wMP +
            Bx[i + 1, j - 1] * wPM
        )

        Bys[k] = (
            By[i + 0, j + 0] * w00 +
            By[i + 1, j + 0] * wP0 +
            By[i + 0, j + 1] * w0P +
            By[i + 1, j + 1] * wPP +
            By[i - 1, j + 0] * wM0 +
            By[i + 0, j - 1] * w0M +
            By[i - 1, j - 1] * wMM +
            By[i - 1, j + 1] * wMP +
            By[i + 1, j - 1] * wPM
        )

        Bzs[k] = 0  # Bz = 0 for now


# TODO: add ro_initial the last, as it is comparatively large (float tricks)?
@numba.cuda.jit
def roj_init_kernel(ro, jx, jy, jz, ro_initial):
    index = numba.cuda.grid(1)
    stride = numba.cuda.blockDim.x * numba.cuda.gridDim.x
    for k in range(index, ro.size, stride):
        ro[k] = ro_initial[k]
        jx[k] = jy[k] = jz[k] = 0


@numba.cuda.jit
def deposit_kernel(n_dim, h,
                   fine_grid,
                   c_x_init, c_y_init, c_x_offt, c_y_offt,
                   c_m, c_q, c_p_x, c_p_y, c_p_z,  # coarse
                   A_weights, B_weights, C_weights, D_weights,
                   indices_prev, indices_next, smallness_factor,
                   out_ro, out_jx, out_jy, out_jz):
    index = numba.cuda.grid(1)
    stride = numba.cuda.blockDim.x * numba.cuda.gridDim.x
    for pk in range(index, A_weights.size, stride):
        pi, pj = pk // A_weights.shape[0], pk % A_weights.shape[0]

        px, nx = indices_prev[pi], indices_next[pi]
        py, ny = indices_prev[pj], indices_next[pj]

        A = A_weights[pi, pj]
        B = B_weights[pi, pj]
        C = C_weights[pi, pj]
        D = D_weights[pi, pj]

        x_offt = A * c_x_offt[px, py] + B * c_x_offt[nx, py] + C * c_x_offt[px, ny] + D * c_x_offt[nx, ny]
        y_offt = A * c_y_offt[px, py] + B * c_y_offt[nx, py] + C * c_y_offt[px, ny] + D * c_y_offt[nx, ny]
        x = fine_grid[pi] + x_offt  # x_fine_init
        y = fine_grid[pj] + y_offt  # y_fine_init
        m = A * c_m[px, py] + B * c_m[nx, py] + C * c_m[px, ny] + D * c_m[nx, ny]
        q = A * c_q[px, py] + B * c_q[nx, py] + C * c_q[px, ny] + D * c_q[nx, ny]
        p_x = A * c_p_x[px, py] + B * c_p_x[nx, py] + C * c_p_x[px, ny] + D * c_p_x[nx, ny]
        p_y = A * c_p_y[px, py] + B * c_p_y[nx, py] + C * c_p_y[px, ny] + D * c_p_y[nx, ny]
        p_z = A * c_p_z[px, py] + B * c_p_z[nx, py] + C * c_p_z[px, ny] + D * c_p_z[nx, ny]
        m *= smallness_factor
        q *= smallness_factor
        p_x *= smallness_factor
        p_y *= smallness_factor
        p_z *= smallness_factor

        gamma_m = sqrt(m**2 + p_x**2 + p_y**2 + p_z**2)
        dro = q / (1 - p_z / gamma_m)
        djx = p_x * (dro / gamma_m)
        djy = p_y * (dro / gamma_m)
        djz = p_z * (dro / gamma_m)

        x_h = x / h + .5
        y_h = y / h + .5
        i = int(floor(x_h)) + n_dim // 2
        j = int(floor(y_h)) + n_dim // 2
        x_loc = x_h - floor(x_h) - 0.5
        y_loc = y_h - floor(y_h) - 0.5

        fx1 = .75 - x_loc**2
        fy1 = .75 - y_loc**2
        fx2 = .5  + x_loc
        fy2 = .5  + y_loc
        fx3 = .5  - x_loc
        fy3 = .5  - y_loc

        fx2_sq = fx2**2
        fy2_sq = fy2**2
        fx3_sq = fx3**2
        fy3_sq = fy3**2

        # atomic +=, thread-safe
        numba.cuda.atomic.add(out_ro, (i + 0, j + 0), dro * (fx1 * fy1))
        numba.cuda.atomic.add(out_ro, (i + 1, j + 0), dro * (fx2_sq * (fy1 / 2)))
        numba.cuda.atomic.add(out_ro, (i + 0, j + 1), dro * (fy2_sq * (fx1 / 2)))
        numba.cuda.atomic.add(out_ro, (i + 1, j + 1), dro * (fx2_sq * (fy2_sq / 4)))
        numba.cuda.atomic.add(out_ro, (i - 1, j + 0), dro * (fx3_sq * (fy1 / 2)))
        numba.cuda.atomic.add(out_ro, (i + 0, j - 1), dro * (fy3_sq * (fx1 / 2)))
        numba.cuda.atomic.add(out_ro, (i - 1, j - 1), dro * (fx3_sq * (fy3_sq / 4)))
        numba.cuda.atomic.add(out_ro, (i - 1, j + 1), dro * (fx3_sq * (fy2_sq / 4)))
        numba.cuda.atomic.add(out_ro, (i + 1, j - 1), dro * (fx2_sq * (fy3_sq / 4)))

        numba.cuda.atomic.add(out_jx, (i + 0, j + 0), djx * (fx1 * fy1))
        numba.cuda.atomic.add(out_jx, (i + 1, j + 0), djx * (fx2_sq * (fy1 / 2)))
        numba.cuda.atomic.add(out_jx, (i + 0, j + 1), djx * (fy2_sq * (fx1 / 2)))
        numba.cuda.atomic.add(out_jx, (i + 1, j + 1), djx * (fx2_sq * (fy2_sq / 4)))
        numba.cuda.atomic.add(out_jx, (i - 1, j + 0), djx * (fx3_sq * (fy1 / 2)))
        numba.cuda.atomic.add(out_jx, (i + 0, j - 1), djx * (fy3_sq * (fx1 / 2)))
        numba.cuda.atomic.add(out_jx, (i - 1, j - 1), djx * (fx3_sq * (fy3_sq / 4)))
        numba.cuda.atomic.add(out_jx, (i - 1, j + 1), djx * (fx3_sq * (fy2_sq / 4)))
        numba.cuda.atomic.add(out_jx, (i + 1, j - 1), djx * (fx2_sq * (fy3_sq / 4)))

        numba.cuda.atomic.add(out_jy, (i + 0, j + 0), djy * (fx1 * fy1))
        numba.cuda.atomic.add(out_jy, (i + 1, j + 0), djy * (fx2_sq * (fy1 / 2)))
        numba.cuda.atomic.add(out_jy, (i + 0, j + 1), djy * (fy2_sq * (fx1 / 2)))
        numba.cuda.atomic.add(out_jy, (i + 1, j + 1), djy * (fx2_sq * (fy2_sq / 4)))
        numba.cuda.atomic.add(out_jy, (i - 1, j + 0), djy * (fx3_sq * (fy1 / 2)))
        numba.cuda.atomic.add(out_jy, (i + 0, j - 1), djy * (fy3_sq * (fx1 / 2)))
        numba.cuda.atomic.add(out_jy, (i - 1, j - 1), djy * (fx3_sq * (fy3_sq / 4)))
        numba.cuda.atomic.add(out_jy, (i - 1, j + 1), djy * (fx3_sq * (fy2_sq / 4)))
        numba.cuda.atomic.add(out_jy, (i + 1, j - 1), djy * (fx2_sq * (fy3_sq / 4)))

        numba.cuda.atomic.add(out_jz, (i + 0, j + 0), djz * (fx1 * fy1))
        numba.cuda.atomic.add(out_jz, (i + 1, j + 0), djz * (fx2_sq * (fy1 / 2)))
        numba.cuda.atomic.add(out_jz, (i + 0, j + 1), djz * (fy2_sq * (fx1 / 2)))
        numba.cuda.atomic.add(out_jz, (i + 1, j + 1), djz * (fx2_sq * (fy2_sq / 4)))
        numba.cuda.atomic.add(out_jz, (i - 1, j + 0), djz * (fx3_sq * (fy1 / 2)))
        numba.cuda.atomic.add(out_jz, (i + 0, j - 1), djz * (fy3_sq * (fx1 / 2)))
        numba.cuda.atomic.add(out_jz, (i - 1, j - 1), djz * (fx3_sq * (fy3_sq / 4)))
        numba.cuda.atomic.add(out_jz, (i - 1, j + 1), djz * (fx3_sq * (fy2_sq / 4)))
        numba.cuda.atomic.add(out_jz, (i + 1, j - 1), djz * (fx2_sq * (fy3_sq / 4)))
    #numba.cuda.syncthreads()


@numba.cuda.jit
def calculate_RHS_Ex_Ey_Bx_By_kernel(Ex_sub, Ey_sub, Bx_sub, By_sub,
                                     beam_ro, ro, jx, jx_prev, jy, jy_prev, jz,
                                     grid_step_size, xi_step_size,
                                     subtraction_trick,
                                     Ex_dct1_in, Ey_dct1_in,
                                     Bx_dct1_in, By_dct1_in):
    N = ro.shape[0]
    index = numba.cuda.grid(1)
    stride = numba.cuda.blockDim.x * numba.cuda.gridDim.x
    for k in range(index, ro.size, stride):
        i, j = k // N, k % N

        dro_dx = (((+ro[i + 1, j] + beam_ro[i + 1, j]
                    -ro[i - 1, j] - beam_ro[i - 1, j])
                  ) / (2 * grid_step_size)  # - ?
                  if 0 < i < N - 1 else 0)
        dro_dy = (((+ro[i, j + 1] + beam_ro[i, j + 1]
                    -ro[i, j - 1] - beam_ro[i, j - 1])
                  ) / (2 * grid_step_size)  # - ?
                  if 0 < j < N - 1 else 0)
        djz_dx = (((+jz[i + 1, j] + beam_ro[i + 1, j]
                    -jz[i - 1, j] - beam_ro[i - 1, j])
                  ) / (2 * grid_step_size)  # - ?
                  if 0 < i < N - 1 else 0)
        djz_dy = (((+jz[i, j + 1] + beam_ro[i, j + 1]
                    -jz[i, j - 1] - beam_ro[i, j - 1])
                  ) / (2 * grid_step_size)  # - ?
                  if 0 < j < N - 1 else 0)
        djx_dxi = (jx_prev[i, j] - jx[i, j]) / xi_step_size               # - ?
        djy_dxi = (jy_prev[i, j] - jy[i, j]) / xi_step_size               # - ?

        Ex_rhs = -((dro_dx - djx_dxi) - Ex_sub[i, j] * subtraction_trick)
        Ey_rhs = -((dro_dy - djy_dxi) - Ey_sub[i, j] * subtraction_trick)
        Bx_rhs = +((djz_dy - djy_dxi) + Bx_sub[i, j] * subtraction_trick)
        By_rhs = -((djz_dx - djx_dxi) - By_sub[i, j] * subtraction_trick)
        Ex_dct1_in[j, i] = Ex_rhs
        Ey_dct1_in[i, j] = Ey_rhs
        Bx_dct1_in[i, j] = Bx_rhs
        By_dct1_in[j, i] = By_rhs
        # symmetrically pad dct1_in to apply DCT-via-FFT later
        ii = max(i, 1)  # avoid writing to dct_in[:, 2 * N - 2], w/o branching
        jj = max(j, 1)
        Ex_dct1_in[j, 2 * N - 2 - ii] = Ex_rhs
        Ey_dct1_in[i, 2 * N - 2 - jj] = Ey_rhs
        Bx_dct1_in[i, 2 * N - 2 - jj] = Bx_rhs
        By_dct1_in[j, 2 * N - 2 - ii] = By_rhs

        # applying non-zero boundary conditions to the RHS would be:
        # for i in range(self.N):
            # rhs_fixed[i, 0] += top[i] * (2 / self.grid_step_size)
            # rhs_fixed[i, self.N - 1] += bot[i] * (2 / self.grid_step_size)
            ## rhs_fixed[0, i] = rhs_fixed[self.N - 1, i] = 0
            ### changes nothing, as there's a particle-free padding zone?


@numba.cuda.jit
def mid_dct_transform(Ex_dct1_out, Ex_dct2_in,
                      Ey_dct1_out, Ey_dct2_in,
                      Bx_dct1_out, Bx_dct2_in,
                      By_dct1_out, By_dct2_in,
                      Ex_bet, Ey_bet, Bx_bet, By_bet,
                      alf, mul):
    N = Ex_dct1_out.shape[0]
    index = numba.cuda.grid(1)
    stride = numba.cuda.blockDim.x * numba.cuda.gridDim.x

    # Solve tridiagonal matrix equation for each spectral column with Thomas method:
    # A @ tmp_2[k, :] = tmp_1[k, :]
    # A has -1 on superdiagonal, -1 on subdiagonal and aa[k] at the main diagonal
    # The edge elements of each column are forced to 0!
    for i in range(index, N, stride):
        Ex_bet[i, 0] = Ey_bet[i, 0] = Bx_bet[i, 0] = By_bet[i, 0] = 0
        for j in range(1, N - 1):
            # Note the transposition for dct1_out!
            Ex_bet[i, j + 1] = (mul * Ex_dct1_out[j, i].real + Ex_bet[i, j]) * alf[i, j + 1]
            Ey_bet[i, j + 1] = (mul * Ey_dct1_out[j, i].real + Ey_bet[i, j]) * alf[i, j + 1]
            Bx_bet[i, j + 1] = (mul * Bx_dct1_out[j, i].real + Bx_bet[i, j]) * alf[i, j + 1]
            By_bet[i, j + 1] = (mul * By_dct1_out[j, i].real + By_bet[i, j]) * alf[i, j + 1]
        # Note the transposition for dct2_in!
        # TODO: it can be set once only? Maybe we can comment that out then?
        Ex_dct2_in[N - 1, i] = Ey_dct2_in[N - 1, i] = 0  # Note the forced zero
        Bx_dct2_in[N - 1, i] = By_dct2_in[N - 1, i] = 0
        for j in range(N - 2, 0 - 1, -1):
            Ex_dct2_in[j, i] = alf[i, j + 1] * Ex_dct2_in[j + 1, i] + Ex_bet[i, j + 1]
            Ey_dct2_in[j, i] = alf[i, j + 1] * Ey_dct2_in[j + 1, i] + Ey_bet[i, j + 1]
            Bx_dct2_in[j, i] = alf[i, j + 1] * Bx_dct2_in[j + 1, i] + Bx_bet[i, j + 1]
            By_dct2_in[j, i] = alf[i, j + 1] * By_dct2_in[j + 1, i] + By_bet[i, j + 1]
            # also symmetrical-fill the array in preparation for a second DCT
            ii = max(i, 1)  # avoid writing to dct_in[:, 2 * N - 2], w/o branching
            Ex_dct2_in[j, 2 * N - 2 - ii] = Ex_dct2_in[j, ii]
            Ey_dct2_in[j, 2 * N - 2 - ii] = Ey_dct2_in[j, ii]
            Bx_dct2_in[j, 2 * N - 2 - ii] = Bx_dct2_in[j, ii]
            By_dct2_in[j, 2 * N - 2 - ii] = By_dct2_in[j, ii]
        # dct2_in[:, 0] == 0  # happens by itself


@numba.cuda.jit
def unpack_Ex_Ey_Bx_By_fields_kernel(Ex_dct2_out, Ey_dct2_out,
                                     Bx_dct2_out, By_dct2_out,
                                     Ex, Ey, Bx, By):
    N = Ex.shape[0]
    index = numba.cuda.grid(1)
    stride = numba.cuda.blockDim.x * numba.cuda.gridDim.x
    for k in range(index, Ex.size, stride):
        i, j = k // N, k % N
        Ex[i, j] = Ex_dct2_out[j, i].real
        Ey[i, j] = Ey_dct2_out[i, j].real
        Bx[i, j] = Bx_dct2_out[i, j].real
        By[i, j] = By_dct2_out[j, i].real


@numba.cuda.jit
def calculate_RHS_Ez_kernel(jx, jy, grid_step_size, Ez_dst1_in):
    N = jx.shape[0]
    Ns = N - 2
    index = numba.cuda.grid(1)
    stride = numba.cuda.blockDim.x * numba.cuda.gridDim.x
    for k in range(index, Ns**2, stride):
        i0, j0 = k // Ns, k % Ns
        i, j = i0 + 1, j0 + 1

        djx_dx = (jx[i + 1, j] - jx[i - 1, j]) / (2 * grid_step_size)  # - ?
        djy_dy = (jy[i, j + 1] - jy[i, j - 1]) / (2 * grid_step_size)  # - ?

        Ez_rhs = -(djx_dx + djy_dy)
        Ez_dst1_in[i0, j0 + 1] = Ez_rhs
        # anti-symmetrically pad dct1_in to apply DCT-via-FFT later
        Ez_dst1_in[i0, 2 * Ns + 1 - j0] = -Ez_rhs

@numba.cuda.jit
def mid_dst_transform(Ez_dst1_out, Ez_dst2_in,
                      Ez_bet, Ez_alf, mul):
    Ns = Ez_dst1_out.shape[0]  # == N - 2
    index = numba.cuda.grid(1)
    stride = numba.cuda.blockDim.x * numba.cuda.gridDim.x

    # Solve tridiagonal matrix equation for each spectral column with Thomas method:
    # A @ tmp_2[k, :] = tmp_1[k, :]
    # A has -1 on superdiagonal, -1 on subdiagonal and aa[i] at the main diagonal
    for i in range(index, Ns, stride):
        Ez_bet[i, 0] = 0
        for j in range(Ns):
            # Note the transposition for dst1_out!
            Ez_bet[i, j + 1] = (mul * -Ez_dst1_out[j, i + 1].imag + Ez_bet[i, j]) * Ez_alf[i, j + 1]
        # Note the transposition for dct2_in!
        Ez_dst2_in[Ns - 1, i + 1] = 0 + Ez_bet[i, Ns]  # 0 = Ez_dst2_in[i, Ns] (fake)
        Ez_dst2_in[Ns - 1, 2 * Ns + 1 - i] = -Ez_dst2_in[Ns - 1, i + 1]
        for j in range(Ns - 2, 0 - 1, -1):
            Ez_dst2_in[j, i + 1] = Ez_alf[i, j + 1] * Ez_dst2_in[j + 1, i + 1] + Ez_bet[i, j + 1]
            # anti-symmetrically pad dct1_in to apply DCT-via-FFT later
            Ez_dst2_in[j, 2 * Ns + 1 - i] = -Ez_dst2_in[j, i + 1]


@numba.cuda.jit
def unpack_Ez_kernel(Ez_dst2_out, Ez,
                     Ez_dst1_in, Ez_dst1_out, Ez_dst2_in):
    N = Ez.shape[0]
    Ns = N - 2
    index = numba.cuda.grid(1)
    stride = numba.cuda.blockDim.x * numba.cuda.gridDim.x
    for k in range(index, Ns**2, stride):
        i0, j0 = k // Ns, k % Ns
        i, j = i0 + 1, j0 + 1
        Ez[i, j] = -Ez_dst2_out[i0, j0 + 1].imag


# TODO: try averaging many arrays at once, * .5,
#       maybe even combining field arrays into one
@numba.cuda.jit
def average_arrays_kernel(arr1, arr2, out):
    index = numba.cuda.grid(1)
    stride = numba.cuda.blockDim.x * numba.cuda.gridDim.x
    for k in range(index, out.size, stride):
        out[k] = (arr1[k] + arr2[k]) / 2


@numba.cuda.jit
def move_smart_kernel(xi_step_size, reflect_boundary,
                      ms, qs,
                      x_init, y_init, old_x_offt, old_y_offt,
                      old_px, old_py, old_pz,
                      Exs, Eys, Ezs, Bxs, Bys, Bzs,
                      new_x_offt, new_y_offt, new_px, new_py, new_pz):
    index = numba.cuda.grid(1)
    stride = numba.cuda.blockDim.x * numba.cuda.gridDim.x
    for k in range(index, ms.size, stride):
        m, q = ms[k], qs[k]
        opx, opy, opz = old_px[k], old_py[k], old_pz[k]
        px, py, pz = opx, opy, opz
        x_offt, y_offt = old_x_offt[k], old_y_offt[k]
        Ex, Ey, Ez, Bx, By, Bz = Exs[k], Eys[k], Ezs[k], Bxs[k], Bys[k], Bzs[k]

        gamma_m = sqrt(m**2 + pz**2 + px**2 + py**2)
        vx, vy, vz = px / gamma_m, py / gamma_m, pz / gamma_m
        factor_1 = q * xi_step_size / (1 - pz / gamma_m)
        dpx = factor_1 * (Ex + vy * Bz - vz * By)
        dpy = factor_1 * (Ey - vx * Bz + vz * Bx)
        dpz = factor_1 * (Ez + vx * By - vy * Bx)
        px, py, pz = opx + dpx / 2, opy + dpy / 2, opz + dpz / 2

        gamma_m = sqrt(m**2 + pz**2 + px**2 + py**2)
        vx, vy, vz = px / gamma_m, py / gamma_m, pz / gamma_m
        factor_1 = q * xi_step_size / (1 - pz / gamma_m)
        dpx = factor_1 * (Ex + vy * Bz - vz * By)
        dpy = factor_1 * (Ey - vx * Bz + vz * Bx)
        dpz = factor_1 * (Ez + vx * By - vy * Bx)
        px, py, pz = opx + dpx / 2, opy + dpy / 2, opz + dpz / 2

        gamma_m = sqrt(m**2 + pz**2 + px**2 + py**2)

        x_offt += px / (gamma_m - pz) * xi_step_size
        y_offt += py / (gamma_m - pz) * xi_step_size

        px, py, pz = opx + dpx, opy + dpy, opz + dpz

        # TODO: avoid branching?
        x = x_init[k] + x_offt
        y = y_init[k] + y_offt
        if x > +reflect_boundary:
            x = +2 * reflect_boundary - x
            x_offt = x - x_init[k]
            px = -px
        if x < -reflect_boundary:
            x = -2 * reflect_boundary - x
            x_offt = x - x_init[k]
            px = -px
        if y > +reflect_boundary:
            y = +2 * reflect_boundary - y
            y_offt = y - y_init[k]
            py = -py
        if y < -reflect_boundary:
            y = -2 * reflect_boundary - y
            y_offt = y - y_init[k]
            py = -py

        new_x_offt[k], new_y_offt[k] = x_offt, y_offt  # TODO: get rid of vars
        new_px[k], new_py[k], new_pz[k] = px, py, pz


class GPUMonolith:
    cfg = (19, 384)  # empirical guess for a GTX 1070 Ti

    def __init__(self, config, plasma,
                 A_weights, B_weights, C_weights, D_weights,
                 fine_grid_init,
                 indices_prev, indices_next):
        self.Nc = Nc = int(sqrt(plasma.size))
        assert Nc**2 == plasma.size

        # virtual particles should not reach the window pre-boundary cells
        assert config.reflect_padding_steps > config.plasma_coarseness + 1
        # the alternative is to reflect after plasma virtualization

        self.grid_steps = N = config.grid_steps
        self.xi_step_size = config.xi_step_size
        self.grid_step_size = config.window_width / config.grid_steps
        self.subtraction_trick = config.field_solver_subtraction_trick
        self.reflect_boundary = (
            +config.window_width / 2
            -config.reflect_padding_steps * self.grid_step_size
        )

        self.virtplasma_smallness_factor = 1 / (config.plasma_coarseness *
                                                config.plasma_fineness)**2

        self.x_init = plasma['x_init'].copy()
        self.y_init = plasma['y_init'].copy()
        self._x_init = numba.cuda.to_device(self.x_init.reshape((Nc, Nc)))
        self._y_init = numba.cuda.to_device(self.y_init.reshape((Nc, Nc)))
        self._fine_grid = numba.cuda.to_device(fine_grid_init)

        self._m = numba.cuda.device_array((Nc, Nc))
        self._q = numba.cuda.device_array((Nc, Nc))
        self._x_prev_offt = numba.cuda.device_array((Nc, Nc))
        self._y_prev_offt = numba.cuda.device_array((Nc, Nc))
        self._px_prev = numba.cuda.device_array((Nc, Nc))
        self._py_prev = numba.cuda.device_array((Nc, Nc))
        self._pz_prev = numba.cuda.device_array((Nc, Nc))

        self._x_new_offt = numba.cuda.device_array((Nc, Nc))
        self._y_new_offt = numba.cuda.device_array((Nc, Nc))
        self._px_new = numba.cuda.device_array((Nc, Nc))
        self._py_new = numba.cuda.device_array((Nc, Nc))
        self._pz_new = numba.cuda.device_array((Nc, Nc))

        self._halfstep_x_offt = numba.cuda.device_array((Nc, Nc))
        self._halfstep_y_offt = numba.cuda.device_array((Nc, Nc))

        self._Exs = numba.cuda.device_array((Nc, Nc))
        self._Eys = numba.cuda.device_array((Nc, Nc))
        self._Ezs = numba.cuda.device_array((Nc, Nc))
        self._Bxs = numba.cuda.device_array((Nc, Nc))
        self._Bys = numba.cuda.device_array((Nc, Nc))
        self._Bzs = numba.cuda.device_array((Nc, Nc))

        self._A_weights = numba.cuda.to_device(A_weights)
        self._B_weights = numba.cuda.to_device(B_weights)
        self._C_weights = numba.cuda.to_device(C_weights)
        self._D_weights = numba.cuda.to_device(D_weights)
        self._indices_prev = numba.cuda.to_device(indices_prev)
        self._indices_next = numba.cuda.to_device(indices_next)

        # Arrays for mixed boundary conditions solver
        # * diagonal matrix elements (used in the next one)
        aa = 2 + 4 * np.sin(np.arange(0, N) * np.pi / (2 * (N - 1)))**2
        if self.subtraction_trick:
            aa += self.grid_step_size**2 * self.subtraction_trick
        alf = np.zeros((N, N + 1))
        # * precalculated internal coefficients for tridiagonal solving
        for i in range(1, N):
            alf[:, i + 1] = 1 / (aa - alf[:, i])
        self._mix_alf = numba.cuda.to_device(alf)
        # * scratchpad arrays for mixed boundary conditions solver
        self._Ex_bet = numba.cuda.device_array((N, N))
        self._Ey_bet = numba.cuda.device_array((N, N))
        self._Bx_bet = numba.cuda.device_array((N, N))
        self._By_bet = numba.cuda.device_array((N, N))

        # Arrays for Dirichlet boundary conditions solver
        # * diagonal matrix elements (used in the next one)
        Ez_a = 2 + 4 * np.sin(np.arange(1, N - 1) * np.pi / (2 * (N - 1)))**2
        #  +  h**2  # only used with xi derivatives
        # * precalculated internal coefficients for tridiagonal solving
        Ez_alf = np.zeros((N - 2, N - 1))
        Ez_alf[:, 0] = 0
        for k in range(N - 2):
            for i in range(N - 2):
                Ez_alf[k, i + 1] = 1 / (Ez_a[k] - Ez_alf[k, i])
        self._Ez_alf = numba.cuda.to_device(Ez_alf)
        # * scratchpad arrays for Dirichlet boundary conditions solver
        self._Ez_bet = numba.cuda.device_array((N, N))

        #self.dct_plan = pyculib.fft.FFTPlan(shape=(2 * N - 2,),
        #                                    itype=np.float64,
        #                                    otype=np.complex128,
        #                                    batch=(4 * N))
        # (2 * N - 2) // 2 + 1 == (N - 1) + 1 == N
        self._combined_dct1_in = numba.cuda.device_array((4 * N, 2 * N - 2))
        self._combined_dct1_out = numba.cuda.device_array((4 * N, N), dtype=np.complex128)
        self._combined_dct2_in = numba.cuda.device_array((4 * N, 2 * N - 2))
        self._combined_dct2_out = numba.cuda.device_array((4 * N, N), dtype=np.complex128)
        self._Ex_dct1_in = self._combined_dct1_in[:N, :]
        self._Ex_dct1_out = self._combined_dct1_out[:N, :]
        self._Ex_dct2_in = self._combined_dct2_in[:N, :]
        self._Ex_dct2_out = self._combined_dct2_out[:N, :]
        self._Ey_dct1_in = self._combined_dct1_in[N:2*N, :]
        self._Ey_dct1_out = self._combined_dct1_out[N:2*N, :]
        self._Ey_dct2_in = self._combined_dct2_in[N:2*N, :]
        self._Ey_dct2_out = self._combined_dct2_out[N:2*N, :]
        self._Bx_dct1_in = self._combined_dct1_in[2*N:3*N, :]
        self._Bx_dct1_out = self._combined_dct1_out[2*N:3*N, :]
        self._Bx_dct2_in = self._combined_dct2_in[2*N:3*N, :]
        self._Bx_dct2_out = self._combined_dct2_out[2*N:3*N, :]
        self._By_dct1_in = self._combined_dct1_in[3*N:, :]
        self._By_dct1_out = self._combined_dct1_out[3*N:, :]
        self._By_dct2_in = self._combined_dct2_in[3*N:, :]
        self._By_dct2_out = self._combined_dct2_out[3*N:, :]
        self._Ex = numba.cuda.device_array((N, N))
        self._Ey = numba.cuda.device_array((N, N))
        self._Bx = numba.cuda.device_array((N, N))
        self._By = numba.cuda.device_array((N, N))
        self._Ex_sub = numba.cuda.device_array((N, N))
        self._Ey_sub = numba.cuda.device_array((N, N))
        self._Bx_sub = numba.cuda.device_array((N, N))
        self._By_sub = numba.cuda.device_array((N, N))

        #self.dst_plan = pyculib.fft.FFTPlan(shape=(2 * N - 2,),
        #                                    itype=np.float64,
        #                                    otype=np.complex128,
        #                                    batch=(N - 2))
        self._Ez_dst1_in = numba.cuda.device_array((N - 2, 2 * N - 2))
        self._Ez_dst1_out = numba.cuda.device_array((N - 2, N), dtype=np.complex128)
        self._Ez_dst2_in = numba.cuda.device_array((N - 2, 2 * N - 2))
        self._Ez_dst2_out = numba.cuda.device_array((N - 2, N), dtype=np.complex128)
        self._Ez = numba.cuda.device_array((N, N))

        self._Ez_dst1_in[:, :] = 0
        self._Ez_dst2_in[:, :] = 0
        self._Ez[:, :] = 0

        # total multiplier to compensate for the iDST+DST transforms
        self.Ez_mul = self.grid_step_size**2
        self.Ez_mul /= 2 * N - 2  # don't ask

        # total multiplier to compensate for the iDCT+DCT transforms
        self.mix_mul = self.grid_step_size**2
        self.mix_mul /= 2 * N - 2  # don't ask

        self._Bz = numba.cuda.device_array((N, N))
        self._Bz[:, :] = 0  # Bz = 0 for now

        self._ro_initial = numba.cuda.device_array((N, N))
        self._ro = numba.cuda.device_array((N, N))
        self._jx = numba.cuda.device_array((N, N))
        self._jy = numba.cuda.device_array((N, N))
        self._jz = numba.cuda.device_array((N, N))

        self._beam_ro = numba.cuda.device_array((N, N))

        self._jx_prev = numba.cuda.device_array((N, N))
        self._jy_prev = numba.cuda.device_array((N, N))

        self._Ex_prev = numba.cuda.device_array((N, N))
        self._Ey_prev = numba.cuda.device_array((N, N))
        self._Ez_prev = numba.cuda.device_array((N, N))
        self._Bx_prev = numba.cuda.device_array((N, N))
        self._By_prev = numba.cuda.device_array((N, N))
        self._Bz_prev = numba.cuda.device_array((N, N))

        self._Ex_avg = numba.cuda.device_array((N, N))
        self._Ey_avg = numba.cuda.device_array((N, N))
        self._Ez_avg = numba.cuda.device_array((N, N))
        self._Bx_avg = numba.cuda.device_array((N, N))
        self._By_avg = numba.cuda.device_array((N, N))
        self._Bz_avg = numba.cuda.device_array((N, N))


    def load(self, beam_ro, plasma_prev,
             Ex_prev, Ey_prev, Ez_prev, Bx_prev, By_prev, Bz_prev,
             jx_prev, jy_prev):
        self._beam_ro[:, :] = np.ascontiguousarray(beam_ro)

        self._Ex_sub[:, :] = np.ascontiguousarray(Ex_prev)
        self._Ey_sub[:, :] = np.ascontiguousarray(Ey_prev)
        self._Bx_sub[:, :] = np.ascontiguousarray(Bx_prev)
        self._By_sub[:, :] = np.ascontiguousarray(By_prev)

        Nc = self.Nc

        self._m[:, :] = np.ascontiguousarray(plasma_prev['m'].reshape(Nc, Nc))
        self._q[:, :] = np.ascontiguousarray(plasma_prev['q'].reshape(Nc, Nc))
        self._x_prev_offt[:, :] = np.ascontiguousarray(plasma_prev['x_offt'].reshape(Nc, Nc))
        self._y_prev_offt[:, :] = np.ascontiguousarray(plasma_prev['y_offt'].reshape(Nc, Nc))
        self._px_prev[:, :] = np.ascontiguousarray(plasma_prev['p'][:, 1].reshape(Nc, Nc))
        self._py_prev[:, :] = np.ascontiguousarray(plasma_prev['p'][:, 2].reshape(Nc, Nc))
        self._pz_prev[:, :] = np.ascontiguousarray(plasma_prev['p'][:, 0].reshape(Nc, Nc))

        self._Ex_prev[:, :] = np.ascontiguousarray(Ex_prev)
        self._Ey_prev[:, :] = np.ascontiguousarray(Ey_prev)
        self._Ez_prev[:, :] = np.ascontiguousarray(Ez_prev)
        self._Bx_prev[:, :] = np.ascontiguousarray(Bx_prev)
        self._By_prev[:, :] = np.ascontiguousarray(By_prev)
        self._Bz_prev[:, :] = np.ascontiguousarray(Bz_prev)
        self._jx_prev[:, :] = np.ascontiguousarray(jx_prev)
        self._jy_prev[:, :] = np.ascontiguousarray(jy_prev)

        #self._Ex[:, :] = self._Ex_prev
        #self._Ey[:, :] = self._Ey_prev
        #self._Ez[:, :] = self._Ez_prev
        #self._Bx[:, :] = self._Bx_prev
        #self._By[:, :] = self._By_prev
        #self._Bz[:, :] = self._Bz_prev

        self._Ex_avg[:, :] = self._Ex
        self._Ey_avg[:, :] = self._Ey
        self._Ez_avg[:, :] = self._Ez
        self._Bx_avg[:, :] = self._Bx
        self._By_avg[:, :] = self._By
        self._Bz_avg[:, :] = self._Bz

        self.___plasma = plasma_prev.copy()

        numba.cuda.synchronize()


    def move_predict_halfstep(self):
        move_predict_halfstep_kernel[self.cfg](self.xi_step_size,
                                               self.reflect_boundary,
                                               self._m.ravel(),
                                               self._x_init.ravel(),
                                               self._y_init.ravel(),
                                               self._x_prev_offt.ravel(),
                                               self._y_prev_offt.ravel(),
                                               self._px_prev.ravel(),
                                               self._py_prev.ravel(),
                                               self._pz_prev.ravel(),
                                               self._halfstep_x_offt.ravel(),
                                               self._halfstep_y_offt.ravel())
        numba.cuda.synchronize()


    def interpolate(self):
        interpolate_kernel[self.cfg](self._x_init.ravel(),
                                     self._y_init.ravel(),
                                     self._halfstep_x_offt.ravel(),
                                     self._halfstep_y_offt.ravel(),
                                     self._Ex_avg, self._Ey_avg, self._Ez_avg,
                                     self._Bx_avg, self._By_avg, self._Bz_avg,
                                     self.grid_step_size, self.grid_steps,
                                     self._Exs.ravel(), self._Eys.ravel(),
                                     self._Ezs.ravel(), self._Bxs.ravel(),
                                     self._Bys.ravel(), self._Bzs.ravel())
        numba.cuda.synchronize()


    def move_smart(self):
        move_smart_kernel[self.cfg](self.xi_step_size,
                                    self.reflect_boundary,
                                    self._m.ravel(), self._q.ravel(),
                                    self._x_init.ravel(), self._y_init.ravel(),
                                    self._x_prev_offt.ravel(), self._y_prev_offt.ravel(),
                                    self._px_prev.ravel(),
                                    self._py_prev.ravel(),
                                    self._pz_prev.ravel(),
                                    self._Exs.ravel(), self._Eys.ravel(),
                                    self._Ezs.ravel(), self._Bxs.ravel(),
                                    self._Bys.ravel(), self._Bzs.ravel(),
                                    self._x_new_offt.ravel(), self._y_new_offt.ravel(),
                                    self._px_new.ravel(), self._py_new.ravel(),
                                    self._pz_new.ravel())
        numba.cuda.synchronize()


    def deposit(self):
        roj_init_kernel[self.cfg](self._ro.ravel(), self._jx.ravel(),
                                  self._jy.ravel(), self._jz.ravel(),
                                  self._ro_initial.ravel())
        numba.cuda.synchronize()

        deposit_kernel[self.cfg](self.grid_steps, self.grid_step_size,
                                 self._fine_grid,
                                 self._x_init, self._y_init,
                                 self._x_new_offt, self._y_new_offt,
                                 self._m, self._q,
                                 self._px_new, self._py_new, self._pz_new,
                                 self._A_weights, self._B_weights,
                                 self._C_weights, self._D_weights,
                                 self._indices_prev, self._indices_next,
                                 self.virtplasma_smallness_factor,
                                 self._ro, self._jx, self._jy, self._jz)
        numba.cuda.synchronize()

    def initial_deposition(self, plasma_prev):
        self._ro_initial[:, :] = 0
        self._ro[:, :] = 0
        self._jx[:, :] = 0
        self._jy[:, :] = 0
        self._jz[:, :] = 0

        Nc = self.Nc
        self._m[:, :] = np.ascontiguousarray(plasma_prev['m'].reshape(Nc, Nc))
        self._q[:, :] = np.ascontiguousarray(plasma_prev['q'].reshape(Nc, Nc))
        self._x_new_offt[:, :] = np.ascontiguousarray(plasma_prev['x_offt'].reshape(Nc, Nc))
        self._y_new_offt[:, :] = np.ascontiguousarray(plasma_prev['y_offt'].reshape(Nc, Nc))
        self._px_new[:, :] = np.ascontiguousarray(plasma_prev['p'][:, 1].reshape(Nc, Nc))
        self._py_new[:, :] = np.ascontiguousarray(plasma_prev['p'][:, 2].reshape(Nc, Nc))
        self._pz_new[:, :] = np.ascontiguousarray(plasma_prev['p'][:, 0].reshape(Nc, Nc))

        self.deposit()

        self._ro_initial[:, :] = -np.array(self._ro.copy_to_host())
        numba.cuda.synchronize()


    def calculate_Ex_Ey_Bx_By(self):
        # The grand plan: mul * iDCT(SPECTRAL_MAGIC(DCT(in.T).T)).T).T for Ex/By
        # and mul * iDCT(SPECTRAL_MAGIC(DCT(in).T)).T) for Ey/Bx
        # where iDCT is DCT;
        # and DCT is jury-rigged from symmetrically-padded DFT
        self.calculate_RHS_Ex_Ey_Bx_By()
        self.calculate_Ex_Ey_Bx_By_1()
        self.calculate_Ex_Ey_Bx_By_2()
        self.calculate_Ex_Ey_Bx_By_3()
        self.calculate_Ex_Ey_Bx_By_4()

    def calculate_RHS_Ex_Ey_Bx_By(self):
        calculate_RHS_Ex_Ey_Bx_By_kernel[self.cfg](self._Ex_sub,
                                                   self._Ey_sub,
                                                   self._Bx_sub,
                                                   self._By_sub,
                                                   self._beam_ro,
                                                   self._ro,
                                                   self._jx,
                                                   self._jx_prev,
                                                   self._jy,
                                                   self._jy_prev,
                                                   self._jz,
                                                   self.grid_step_size, self.xi_step_size,
                                                   self.subtraction_trick,
                                                   self._Ex_dct1_in,
                                                   self._Ey_dct1_in,
                                                   self._Bx_dct1_in,
                                                   self._By_dct1_in)
        numba.cuda.synchronize()

    def calculate_Ex_Ey_Bx_By_1(self):
        # 1. Apply iDCT-1 (Discrete Cosine Transform Type 1) to the RHS
        # iDCT-1 is just DCT-1 in cuFFT
        #self.dct_plan.forward(self._combined_dct1_in.ravel(),
        #                      self._combined_dct1_out.ravel())
        self._combined_dct1_out[:, :] = cupy.fft.rfft(cupy.asarray(self._combined_dct1_in))
        numba.cuda.synchronize()
        # This implementation of DCT is real-to-complex, so scrapping the i, j
        # element of the transposed answer would be dct1_out[j, i].real

    def calculate_Ex_Ey_Bx_By_2(self):
        # 2. Solve tridiagonal matrix equation for each spectral column with Thomas method:
        mid_dct_transform[self.cfg](self._Ex_dct1_out, self._Ex_dct2_in,
                                    self._Ey_dct1_out, self._Ey_dct2_in,
                                    self._Bx_dct1_out, self._Bx_dct2_in,
                                    self._By_dct1_out, self._By_dct2_in,
                                    self._Ex_bet, self._Ey_bet,
                                    self._Bx_bet, self._By_bet,
                                    self._mix_alf, self.mix_mul)
        numba.cuda.synchronize()

    def calculate_Ex_Ey_Bx_By_3(self):
        # 3. Apply DCT-1 (Discrete Cosine Transform Type 1) to the transformed spectra
        #self.dct_plan.forward(self._combined_dct2_in.ravel(),
        #                      self._combined_dct2_out.ravel())
        self._combined_dct2_out[:, :] = cupy.fft.rfft(cupy.asarray(self._combined_dct2_in))
        numba.cuda.synchronize()

    def calculate_Ex_Ey_Bx_By_4(self):
        # 4. Transpose the resulting Ex (TODO: fuse this step into later steps?)
        unpack_Ex_Ey_Bx_By_fields_kernel[self.cfg](self._Ex_dct2_out,
                                                   self._Ey_dct2_out,
                                                   self._Bx_dct2_out,
                                                   self._By_dct2_out,
                                                   self._Ex, self._Ey,
                                                   self._Bx, self._By)
        numba.cuda.synchronize()


    def calculate_Ez(self):
        # The grand plan: mul * iDST(SPECTRAL_MAGIC(DST(in).T)).T)
        # where iDST is DST;
        # and DST is jury-rigged from symmetrically-padded DFT
        self.calculate_RHS_Ez()
        self.calculate_Ez_1()
        self.calculate_Ez_2()
        self.calculate_Ez_3()
        self.calculate_Ez_4()

    def calculate_RHS_Ez(self):
        calculate_RHS_Ez_kernel[self.cfg](self._jx, self._jy,
                                          self.grid_step_size,
                                          self._Ez_dst1_in)
        numba.cuda.synchronize()

    def calculate_Ez_1(self):
        # 1. Apply iDST-1 (Discrete Sine Transform Type 1) to the RHS
        # iDST-1 is just DST-1 in cuFFT
        #self.dst_plan.forward(self._Ez_dst1_in.ravel(),
        #                      self._Ez_dst1_out.ravel())
        self._Ez_dst1_out[:, :] = cupy.fft.rfft(cupy.asarray(self._Ez_dst1_in))
        numba.cuda.synchronize()
        # This implementation of DST is real-to-complex, so scrapping the i, j
        # element of the transposed answer would be -dst1_out[j, i + 1].imag

    def calculate_Ez_2(self):
        # 2. Solve tridiagonal matrix equation for each spectral column with Thomas method:
        mid_dst_transform[self.cfg](self._Ez_dst1_out, self._Ez_dst2_in,
                                    self._Ez_bet, self._Ez_alf, self.Ez_mul)
        numba.cuda.synchronize()

    def calculate_Ez_3(self):
        # 3. Apply DST-1 (Discrete Sine Transform Type 1) to the transformed spectra
        #self.dst_plan.forward(self._Ez_dst2_in.ravel(),
        #                      self._Ez_dst2_out.ravel())
        self._Ez_dst2_out[:, :] = cupy.fft.rfft(cupy.asarray(self._Ez_dst2_in))
        numba.cuda.synchronize()

    def calculate_Ez_4(self):
        # 4. Transpose the resulting Ex (TODO: fuse this step into later steps?)
        unpack_Ez_kernel[self.cfg](self._Ez_dst2_out, self._Ez,
                                   self._Ez_dst1_in, self._Ez_dst1_out, self._Ez_dst2_in)
        numba.cuda.synchronize()


    def average_fields(self):
        average_arrays_kernel[self.cfg](self._Ex_prev.ravel(), self._Ex.ravel(), self._Ex_avg.ravel())
        average_arrays_kernel[self.cfg](self._Ey_prev.ravel(), self._Ey.ravel(), self._Ey_avg.ravel())
        average_arrays_kernel[self.cfg](self._Ez_prev.ravel(), self._Ez.ravel(), self._Ez_avg.ravel())
        average_arrays_kernel[self.cfg](self._Bx_prev.ravel(), self._Bx.ravel(), self._Bx_avg.ravel())
        average_arrays_kernel[self.cfg](self._By_prev.ravel(), self._By.ravel(), self._By_avg.ravel())
        self._Ex_sub[:, :] = self._Ex_avg
        self._Ey_sub[:, :] = self._Ey_avg
        self._Bx_sub[:, :] = self._Bx_avg
        self._By_sub[:, :] = self._By_avg
        # average_arrays_kernel[self.cfg](self._Bz_prev.ravel(), self._Bz.ravel(), self._Bz_avg.ravel())  # 0 for now
        numba.cuda.synchronize()


    def average_halfstep(self):
        average_arrays_kernel[self.cfg](self._x_prev_offt.ravel(), self._x_new_offt.ravel(),
                                        self._halfstep_x_offt.ravel())
        average_arrays_kernel[self.cfg](self._y_prev_offt.ravel(), self._y_new_offt.ravel(),
                                        self._halfstep_y_offt.ravel())
        numba.cuda.synchronize()


    def step(self):
        self.move_predict_halfstep()  # ... -> v1 [xy]_halfstep
        self.interpolate()            # ... -> v1 [EB][xyz]s  # fake-avg
        self.move_smart()             # ... -> v1 [xy]_new, p[xyz]_new
        self.deposit()                # ... -> v1 ro, j[xyz
        self.calculate_Ex_Ey_Bx_By()  # ... -> v2 [EB][xy]
        self.calculate_Ez()           # ... -> v2 Ez
        # Bz = 0 for now
        self.average_fields()         # ... -> v2 [EB][xyz]_avg -> [EB][xy]_sub  # TODO: are avg/sub both needed?

        self.average_halfstep()       # ... -> v2 [xy]_halfstep
        self.interpolate()            # ... -> v2 [EB][xyz]s
        self.move_smart()             # ... -> v2 [xy]_new, p[xyz]_new
        self.deposit()                # ... -> v2 ro, j[xyz]
        self.calculate_Ex_Ey_Bx_By()  # ... -> v3 [EB][xy]
        self.calculate_Ez()           # ... -> v3 Ez
        # Bz = 0 for now
        self.average_fields()         # ... -> v3 [EB][xyz]_avg -> [EB][xy]_sub  # TODO: are avg/sub both needed?

        self.average_halfstep()       # ... -> v3 [xy]_halfstep
        self.interpolate()            # ... -> v3 [EB][xyz]s
        self.move_smart()             # ... -> v3 [xy]_new, p[xyz]_new
        self.deposit()                # ... -> v3 ro, j[xyz]

        # TODO: what do we need that roj_new for, jx_prev/jy_prev only?


    def __getattr__(self, array_name):
        '''
        Access GPU arrays of GPUMonolith conveniently, copying them to host.
        Example: `gpu_monolith.ro` becomes `gpu_monolith._ro.copy_to_host()`.
        '''
        return getattr(self, '_' + array_name).copy_to_host()


    def reload(self, beam_ro):
        self._beam_ro[:, :] = np.ascontiguousarray(beam_ro)

        # TODO: array relabeling instead of copying?..

        # Intact: self._m, self._q
        self._x_prev_offt[:, :] = self._x_new_offt
        self._y_prev_offt[:, :] = self._y_new_offt
        self._px_prev[:, :] = self._px_new
        self._py_prev[:, :] = self._py_new
        self._pz_prev[:, :] = self._pz_new

        self._Ex_prev[:, :] = self._Ex
        self._Ey_prev[:, :] = self._Ey
        self._Ez_prev[:, :] = self._Ez
        self._Bx_prev[:, :] = self._Bx
        self._By_prev[:, :] = self._By
        self._Bz_prev[:, :] = self._Bz
        self._jx_prev[:, :] = self._jx
        self._jy_prev[:, :] = self._jy

        self._Ex_avg[:, :] = self._Ex
        self._Ey_avg[:, :] = self._Ey
        self._Ez_avg[:, :] = self._Ez
        self._Bx_avg[:, :] = self._Bx
        self._By_avg[:, :] = self._By
        self._Bz_avg[:, :] = self._Bz

        self._Ex_sub[:, :] = self._Ex
        self._Ey_sub[:, :] = self._Ey
        self._Bx_sub[:, :] = self._Bx
        self._By_sub[:, :] = self._By

        numba.cuda.synchronize()


# TODO: try local arrays for bet (on larger grid sizes)?
# TODO: specialize for specific grid sizes?
# TODO: try going syncless


def make_coarse_plasma_grid(window_width, steps, coarseness):
    plasma_step = window_width * coarseness / steps
    right_half = np.arange(0, window_width / 2, plasma_step)
    left_half = -right_half[:0:-1]  # invert, reverse, drop zero
    plasma_grid = np.concatenate([left_half, right_half])
    assert(np.array_equal(plasma_grid, -plasma_grid[::-1]))
    return plasma_grid


def make_fine_plasma_grid(window_width, steps, fineness):
    plasma_step = window_width / steps / fineness
    assert(fineness == int(fineness))
    if fineness % 2:  # some on zero axes, none on cell corners
        right_half = np.arange(0, window_width / 2, plasma_step)
        left_half = -right_half[:0:-1]  # invert, reverse, drop zero
        plasma_grid = np.concatenate([left_half, right_half])
    else:  # none on zero axes, none on cell corners
        right_half = np.arange(plasma_step / 2, window_width / 2, plasma_step)
        left_half = -right_half[::-1]  # invert, reverse
        plasma_grid = np.concatenate([left_half, right_half])
    assert(np.array_equal(plasma_grid, -plasma_grid[::-1]))
    return plasma_grid


def plasma_make(window_width, steps, coarseness=2, fineness=2):
    cell_size = window_width / steps
    half_width = window_width / 2
    coarse_step, fine_step = cell_size * coarseness, cell_size / fineness

    # Make two initial grids of plasma particles, coarse and fine.
    # Coarse is the one that will evolve and fine is the one to be bilinearly
    # interpolated from the coarse one based on the initial positions.

    coarse_grid = make_coarse_plasma_grid(window_width, steps, coarseness)
    coarse_grid_xs, coarse_grid_ys = coarse_grid[:, None], coarse_grid[None, :]

    fine_grid = make_fine_plasma_grid(window_width, steps, fineness)
    fine_grid_xs, fine_grid_ys = fine_grid[:, None], fine_grid[None, :]

    Nc, Nf = len(coarse_grid), len(fine_grid)

    # Create plasma particles on that grids

    coarse_plasma = np.zeros(Nc**2, plasma_particle_dtype)
    coarse_plasma['N'] = np.arange(coarse_plasma.size)
    coarse_electrons = coarse_plasma.reshape(Nc, Nc)
    coarse_electrons['x_init'] = coarse_grid_xs
    coarse_electrons['y_init'] = coarse_grid_ys
    coarse_electrons['m'] = USUAL_ELECTRON_MASS * coarseness**2
    coarse_electrons['q'] = USUAL_ELECTRON_CHARGE * coarseness**2
    # v, p, x_offt, y_offt == 0

    # TODO: remove
    #fine_plasma = np.zeros(Nf**2, plasma_particle_dtype)
    #fine_plasma['N'] = np.arange(fine_plasma.size)       # not really needed
    #fine_electrons = fine_plasma.reshape(Nf, Nf)
    #fine_electrons['x_init'] = fine_grid_xs
    #fine_electrons['y_init'] = fine_grid_ys
    #fine_electrons['m'] = USUAL_ELECTRON_MASS / fineness**2
    #fine_electrons['q'] = USUAL_ELECTRON_CHARGE / fineness**2

    # Calculate indices for coarse -> fine bilinear interpolation

    # 1D, same in both x and y direction
    # Example: [0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 5 5 5 5 6 6 6]
    indices = np.searchsorted(coarse_grid, fine_grid)
    indices_next = np.clip(indices, 0, Nc - 1)  # [0 0 0 0 0 0 0 0 1 1 ...]
    indices_prev = np.clip(indices - 1, 0, Nc - 1)  # [... 4 5 5 5 5 5 5 5]

    # 2D
    i_prev_in_x, i_next_in_x = indices_prev[:, None], indices_next[:, None]
    i_prev_in_y, i_next_in_y = indices_prev[None, :], indices_next[None, :]

    # Calculate weights for coarse -> fine interpolation from initial positions

    # 1D linear interpolation coefficients in 2 directions
    # The further the fine particle is from closest right coarse particles,
    # the more influence the left ones have.
    influence_prev_x = (coarse_grid[i_next_in_x] - fine_grid_xs) / coarse_step
    influence_next_x = (fine_grid_xs - coarse_grid[i_prev_in_x]) / coarse_step
    influence_prev_y = (coarse_grid[i_next_in_y] - fine_grid_ys) / coarse_step
    influence_next_y = (fine_grid_ys - coarse_grid[i_prev_in_y]) / coarse_step

    # Fix for boundary cases of missing cornering particles
    influence_prev_x[fine_grid_xs <= coarse_grid[0]] = 0   # nothing on left?
    influence_next_x[fine_grid_xs <= coarse_grid[0]] = 1   # use right
    influence_next_x[fine_grid_xs >= coarse_grid[-1]] = 0  # nothing on right?
    influence_prev_x[fine_grid_xs >= coarse_grid[-1]] = 1  # use left
    influence_prev_y[fine_grid_ys <= coarse_grid[0]] = 0   # nothing on bottom?
    influence_next_y[fine_grid_ys <= coarse_grid[0]] = 1   # use top
    influence_next_y[fine_grid_ys >= coarse_grid[-1]] = 0  # nothing on top?
    influence_prev_y[fine_grid_ys >= coarse_grid[-1]] = 1  # use bottom

    # Calculate 2D bilinear interpolation coefficients for four initially
    # cornering coarse plasma particles.

    #  C    D  #  y ^
    #     .    #    |
    #          #    +---->
    #  A    B  #         x

    # A is coarse_plasma[i_prev_in_x, i_prev_in_y], the closest coarse particle
    # in bottom-left quadrant (for each fine particle)
    A_weights = influence_prev_x * influence_prev_y
    # B is coarse_plasma[i_next_in_x, i_prev_in_y], same for lower right
    B_weights = influence_next_x * influence_prev_y
    # C is coarse_plasma[i_prev_in_x, i_next_in_y], same for upper left
    C_weights = influence_prev_x * influence_next_y
    # D is coarse_plasma[i_next_in_x, i_next_in_y], same for upper right
    D_weights = influence_next_x * influence_next_y

    ratio = coarseness ** 2 * fineness ** 2

    # TODO: decide on a flat-or-square plasma
    return (coarse_plasma.ravel(), A_weights, B_weights, C_weights, D_weights,
            fine_grid, indices_prev, indices_next)


max_zn = 0
def diags_ro_zn(config, ro):
    global max_zn

    sigma = 0.25 * config.grid_steps / config.window_width
    blurred = scipy.ndimage.gaussian_filter(ro, sigma=sigma)
    hf = ro - blurred
    zn = np.abs(hf).mean() / 4.23045376e-04
    max_zn = max(max_zn, zn)
    return zn, max_zn


Ez_00_history = []
def diags_peak_msg_just_store(Ez_00):
    global Ez_00_history
    Ez_00_history.append(Ez_00)

def diags_peak_msg(config, Ez_00):
    global Ez_00_history
    Ez_00_array = np.array(Ez_00_history)
    peak_indices = scipy.signal.argrelmax(Ez_00_array)[0]

    if peak_indices.size:
        peak_values = Ez_00_array[peak_indices]
        rel_deviations_perc = 100 * (peak_values / peak_values[0] - 1)
        return (f'{peak_values[-1]:0.4e} '
                f'{rel_deviations_perc[-1]:+0.2f}%')
                #f' ±{rel_deviations_perc.ptp() / 2:0.2f}%')
    else:
        return '...'


def diags_ro_slice(config, xi_i, xi, ro):
    if xi_i % int(1 / config.xi_step_size):
        return
    if not os.path.isdir('transverse'):
        os.mkdir('transverse')

    fname = f'ro_{xi:+09.2f}.png' if xi else 'ro_-00000.00.png'
    plt.imsave(os.path.join('transverse', fname), ro.T,
               origin='lower', vmin=-0.1, vmax=0.1, cmap='bwr')


def diagnostics(gpu, config, xi_i):
    xi = -xi_i * config.xi_step_size

    Ez_00 = gpu._Ez[config.grid_steps // 2, config.grid_steps // 2]
    peak_report = diags_peak_msg(config, Ez_00)

    ro = gpu.ro
    zn, max_zn = diags_ro_zn(config, ro)
    diags_ro_slice(config, xi_i, xi, ro)

    print(f'xi={xi:+.4f} {Ez_00:+.4e}|{peak_report}|zn={max_zn:.3f}')
    sys.stdout.flush()


def init(config):
    grid = ((np.arange(config.grid_steps) + .5) *
            config.window_width / config.grid_steps -
            config.window_width / 2)
    xs, ys = grid[:, None], grid[None, :]

    grid_step_size = config.window_width / config.grid_steps  # TODO: -1 or not?
    plasma, *virt_params = plasma_make(
        config.window_width - config.plasma_padding_steps * 2 * grid_step_size,
        config.grid_steps - config.plasma_padding_steps * 2,
        coarseness=config.plasma_coarseness, fineness=config.plasma_fineness
    )

    gpu = GPUMonolith(config, plasma, *virt_params)
    gpu.load(0, plasma, 0, 0, 0, 0, 0, 0, 0, 0)
    gpu.initial_deposition(plasma)

    return gpu, xs, ys, plasma


def main():
    import config
    gpu, xs, ys, plasma = init(config)

    for xi_i in range(config.xi_steps):
        beam_ro = config.beam(xi_i, xs, ys)

        gpu.reload(beam_ro)
        gpu.step()

        Ez_00 = gpu._Ez[config.grid_steps // 2, config.grid_steps // 2]
        diags_peak_msg_just_store(Ez_00)

        time_for_diags = xi_i % config.diagnostics_each_N_steps == 0
        last_step = xi_i == config.xi_steps - 1
        if time_for_diags or last_step:
            diagnostics(gpu, config, xi_i)


if __name__ == '__main__':
    main()

