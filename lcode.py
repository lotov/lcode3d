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

import os
import sys

import matplotlib.pyplot as plt

import numpy as np

import numba
import numba.cuda

import cupy as cp

import scipy.ndimage
import scipy.signal


# Prevent all CPU cores waiting for the GPU at 100% utilization (under conda).
# os.environ['OMP_NUM_THREADS'] = '1'


ELECTRON_CHARGE = -1
ELECTRON_MASS = 1


# TODO: macrosity


### Solving Laplace equation with Dirichlet boundary conditions (Ez)


def calculate_RHS_Ez(grid_step_size, jx, jy):
    h2 = grid_step_size * 2

    # NOTE: use gradient instead if available (cupy doesn't have gradient)
    djx_dx, _ = dx_dy(jx, h2)
    _, djy_dy = dx_dy(jy, h2)

    return -(djx_dx + djy_dy)


def dst2d(a):
    # DST-Type1-2D, jury-rigged from symmetrically-padded rFFT
    assert a.shape[0] == a.shape[1]
    N = a.shape[0]
    #                                 / 0  0  0  0  0  0 \
    #                                |  0  1  2  0 -2 -1  |
    #  / 1  2 \  anti-symmetrically  |  0  3  4  0 -4 -3  |
    #  \ 3  4 /      padded to       |  0  0  0  0  0  0  |
    #                                |  0 -3 -4  0 +4 +3  |
    #                                 \ 0 -1 -2  0 +2 +1 /
    p = cp.zeros((2 * N + 2, 2 * N + 2))
    p[1:N+1, 1:N+1], p[1:N+1, N+2:] = a,             -cp.fliplr(a)
    p[N+2:,  1:N+1], p[N+2:,  N+2:] = -cp.flipud(a), +cp.fliplr(cp.flipud(a))

    # rFFT-2D, cut out the top-left corner, take -Re
    return -cp.fft.rfft2(p)[1:N+1, 1:N+1].real


class DirichletSolver:
    def __init__(self, N, h):
        self.N, self.h = N, h

        # Samarskiy-Nikolaev, p. 187
        k = np.arange(1, N)
        # 4 / h**2 * sin(k * pi * h / (2 * L2))**2, where L2 = h * (N - 1)
        lamb = 4 / self.h**2 * np.sin(k * np.pi / (2 * (N - 1)))**2
        mul = np.zeros((N - 2, N - 2))
        for i in range(N - 2):
            for j in range(N - 2):
                # 1 / (2 * (N - 1))**2 makes up for DST+iDST scaling
                # 1 / (lamb[i] + lamb[j] is part of the method
                mul[i, j] = 1 / (2 * (N - 1))**2 / (lamb[i] + lamb[j])
        self._mul = cp.array(mul)


    def solve(self, rhs):
        # TODO: Try to optimize pad-dst-mul-unpad-pad-dst-unpad-pad
        #       down to pad-dst-mul-dst-unpad, but carefully.
        #       Or maybe not.

        # Solve Laplace x = -RHS for x with Dirichlet boundary conditions.
        # The perimeter of rhs and out is assumed to be zero.
        N = self.N
        assert rhs.shape[0] == rhs.shape[1] == N
        out = cp.zeros((N, N))

        # 1. Apply DST-Type1-2D (Discrete Sine Transform Type 1 2D) to the RHS
        #f = scipy.fftpack.dstn(rhs[1:-1, 1:-1].get(), type=1)
        f = dst2d(rhs[1:-1, 1:-1])

        # 2. Multiply f by mul
        #f *= self._mul.get()
        f *= self._mul

        # 3. Apply iDST-Type1-2D (Inverse Discrete Sine Transform Type 1 2D),
        #    which matches DST-Type1-2D to the multiplier.
        #out[1:-1, 1:-1] = cp.asarray(scipy.fftpack.idstn(f, type=1))
        out[1:-1, 1:-1] = dst2d(f)

        numba.cuda.synchronize()
        return out


def calculate_Ez(dirichlet_solver, grid_step_size, jx, jy):
    Ez_rhs = calculate_RHS_Ez(grid_step_size, jx, jy)
    Ez = dirichlet_solver.solve(Ez_rhs)
    numba.cuda.synchronize()
    return Ez


### Solving Laplace or Helmholtz equation with mixed boundary conditions
# TODO: do not prettify too much, replace with DCT-2D instead


def dx_dy(arr, h2):
    # NOTE: use gradient instead if available (cupy doesn't have gradient)
    dx, dy = cp.zeros_like(arr), cp.zeros_like(arr)
    dx[1:-1, 1:-1] = arr[2:, 1:-1] - arr[:-2, 1:-1]  # we have 0s
    dy[1:-1, 1:-1] = arr[1:-1, 2:] - arr[1:-1, :-2]  # on the perimeter
    return dx / h2, dy / h2


def calculate_RHS_Ex_Ey_Bx_By(grid_step_size, xi_step_size,
                              subtraction_trick,
                              Ex_avg, Ey_avg, Bx_avg, By_avg,
                              beam_ro, ro, jx, jy, jz, jx_prev, jy_prev):
    h2 = grid_step_size * 2

    # NOTE: use gradient instead if available (cupy doesn't have gradient)
    dro_dx, dro_dy = dx_dy(ro + beam_ro, h2)
    djz_dx, djz_dy = dx_dy(jz + beam_ro, h2)
    djx_dxi = (jx_prev - jx) / xi_step_size               # - ?
    djy_dxi = (jy_prev - jy) / xi_step_size               # - ?

    Ex_rhs = -((dro_dx - djx_dxi) - Ex_avg * subtraction_trick)
    Ey_rhs = -((dro_dy - djy_dxi) - Ey_avg * subtraction_trick)
    Bx_rhs = +((djz_dy - djy_dxi) + Bx_avg * subtraction_trick)
    By_rhs = -((djz_dx - djx_dxi) - By_avg * subtraction_trick)

    return Ex_rhs, Ey_rhs, Bx_rhs, By_rhs


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


class MixedSolver:
    def __init__(self, N, h, subtraction_trick, cfg):
        # Arrays for mixed boundary conditions solver
        # * diagonal matrix elements (used in the next one)
        aa = 2 + 4 * np.sin(np.arange(0, N) * np.pi / (2 * (N - 1)))**2
        if subtraction_trick:
            aa += h**2 * subtraction_trick
        alf = np.zeros((N, N + 1))
        # * precalculated internal coefficients for tridiagonal solving
        for i in range(1, N):
            alf[:, i + 1] = 1 / (aa - alf[:, i])
        self._mix_alf = cp.array(alf)

        # * scratchpad arrays for mixed boundary conditions solver
        #self.dct_plan = pyculib.fft.FFTPlan(shape=(2 * N - 2,),
        #                                    itype=np.float64,
        #                                    otype=np.complex128,
        #                                    batch=(4 * N))
        # (2 * N - 2) // 2 + 1 == (N - 1) + 1 == N
        self._Ex = cp.zeros((N, N))
        self._Ey = cp.zeros((N, N))
        self._Bx = cp.zeros((N, N))
        self._By = cp.zeros((N, N))
        self._Ex_dct1_in = cp.zeros((N, 2 * N - 2))
        self._Ex_dct1_out = cp.zeros((N, N), dtype=cp.complex128)
        self._Ex_dct2_in = cp.zeros((N, 2 * N - 2))
        self._Ex_dct2_out = cp.zeros((N, N), dtype=cp.complex128)
        self._Ey_dct1_in = cp.zeros((N, 2 * N - 2))
        self._Ey_dct1_out = cp.zeros((N, N), dtype=cp.complex128)
        self._Ey_dct2_in = cp.zeros((N, 2 * N - 2))
        self._Ey_dct2_out = cp.zeros((N, N), dtype=cp.complex128)
        self._Bx_dct1_in = cp.zeros((N, 2 * N - 2))
        self._Bx_dct1_out = cp.zeros((N, N), dtype=cp.complex128)
        self._Bx_dct2_in = cp.zeros((N, 2 * N - 2))
        self._Bx_dct2_out = cp.zeros((N, N), dtype=cp.complex128)
        self._By_dct1_in = cp.zeros((N, 2 * N - 2))
        self._By_dct1_out = cp.zeros((N, N), dtype=cp.complex128)
        self._By_dct2_in = cp.zeros((N, 2 * N - 2))
        self._By_dct2_out = cp.zeros((N, N), dtype=cp.complex128)
        self._Ex_bet = cp.zeros((N, N))
        self._Ey_bet = cp.zeros((N, N))
        self._Bx_bet = cp.zeros((N, N))
        self._By_bet = cp.zeros((N, N))


        # total multiplier to compensate for the iDCT+DCT transforms
        self.mix_mul = h**2
        self.mix_mul /= 2 * N - 2  # don't ask

        self.cfg = cfg

    def solve(self, Ex_rhs, Ey_rhs, Bx_rhs, By_rhs):
        # 0. Symmetrically pad dct1_in to apply DCT-via-FFT later
        N = Ex_rhs.shape[0]
        self._Ex_dct1_in[:, :N] = Ex_rhs.T
        self._Ey_dct1_in[:, :N] = Ey_rhs
        self._Bx_dct1_in[:, :N] = Bx_rhs
        self._By_dct1_in[:, :N] = By_rhs.T

        self._Ex_dct1_in[:, N:] = Ex_rhs.T[:, -2:0:-1]  # [1:-1][:, ::-1]
        self._Ey_dct1_in[:, N:] = Ey_rhs[:, -2:0:-1]
        self._Bx_dct1_in[:, N:] = Bx_rhs[:, -2:0:-1]
        self._By_dct1_in[:, N:] = By_rhs.T[:, -2:0:-1]

        # 1. Apply iDCT-1 (Discrete Cosine Transform Type 1) to the RHS
        # iDCT-1 is just DCT-1 in cuFFT
        self._Ex_dct1_out[...] = cp.fft.rfft(cp.asarray(self._Ex_dct1_in))
        self._Ey_dct1_out[...] = cp.fft.rfft(cp.asarray(self._Ey_dct1_in))
        self._Bx_dct1_out[...] = cp.fft.rfft(cp.asarray(self._Bx_dct1_in))
        self._By_dct1_out[...] = cp.fft.rfft(cp.asarray(self._By_dct1_in))
        # This implementation of DCT is real-to-complex, so scrapping the i, j
        # element of the transposed answer would be dct1_out[j, i].real

        # 2. Solve tridiagonal matrix equation for each spectral column with Thomas method:
        mid_dct_transform[self.cfg](self._Ex_dct1_out, self._Ex_dct2_in,
                                    self._Ey_dct1_out, self._Ey_dct2_in,
                                    self._Bx_dct1_out, self._Bx_dct2_in,
                                    self._By_dct1_out, self._By_dct2_in,
                                    self._Ex_bet, self._Ey_bet,
                                    self._Bx_bet, self._By_bet,
                                    self._mix_alf, self.mix_mul)

        # 3. Apply DCT-1 (Discrete Cosine Transform Type 1) to the transformed spectra
        self._Ex_dct2_out[...] = cp.fft.rfft(cp.asarray(self._Ex_dct2_in))
        self._Ey_dct2_out[...] = cp.fft.rfft(cp.asarray(self._Ey_dct2_in))
        self._Bx_dct2_out[...] = cp.fft.rfft(cp.asarray(self._Bx_dct2_in))
        self._By_dct2_out[...] = cp.fft.rfft(cp.asarray(self._By_dct2_in))

        # 4. Transpose the resulting Ex (TODO: fuse this step into later steps?)
        unpack_Ex_Ey_Bx_By_fields_kernel[self.cfg](self._Ex_dct2_out,
                                                   self._Ey_dct2_out,
                                                   self._Bx_dct2_out,
                                                   self._By_dct2_out,
                                                   self._Ex, self._Ey,
                                                   self._Bx, self._By)

        numba.cuda.synchronize()

        return self._Ex, self._Ey, self._Bx, self._By


def calculate_Ex_Ey_Bx_By(grid_step_size, xi_step_size, subtraction_trick,
                          mixed_solver, Ex_avg, Ey_avg, Bx_avg, By_avg,
                          beam_ro, ro, jx, jy, jz, jx_prev, jy_prev):
    Ex_rhs, Ey_rhs, Bx_rhs, By_rhs = \
        calculate_RHS_Ex_Ey_Bx_By(grid_step_size, xi_step_size,
                                  subtraction_trick,
                                  Ex_avg, Ey_avg, Bx_avg, By_avg,
                                  beam_ro, ro, jx, jy, jz, jx_prev, jy_prev)
    return mixed_solver.solve(Ex_rhs, Ey_rhs, Bx_rhs, By_rhs)


### Unsorted


@numba.cuda.jit
def move_estimate_wo_fields_kernel(xi_step_size, reflect_boundary, ms,
                                   x_init, y_init, old_x_offt, old_y_offt,
                                   pxs, pys, pzs,
                                   estimated_x_offt, estimated_y_offt):
    index = numba.cuda.grid(1)
    stride = numba.cuda.blockDim.x * numba.cuda.gridDim.x
    for k in range(index, ms.size, stride):
        m = ms[k]
        x, y = x_init[k] + old_x_offt[k], y_init[k] + old_y_offt[k]
        px, py, pz = pxs[k], pys[k], pzs[k]

        gamma_m = sqrt(m**2 + pz**2 + px**2 + py**2)

        x += px / (gamma_m - pz) * xi_step_size
        y += py / (gamma_m - pz) * xi_step_size

        # TODO: avoid branching?
        x = x if x <= +reflect_boundary else +2 * reflect_boundary - x
        x = x if x >= -reflect_boundary else -2 * reflect_boundary - x
        y = y if y <= +reflect_boundary else +2 * reflect_boundary - y
        y = y if y >= -reflect_boundary else -2 * reflect_boundary - y

        estimated_x_offt[k], estimated_y_offt[k] = x - x_init[k], y - y_init[k]


@numba.jit(inline=True)
def weights(x, y, grid_steps, grid_step_size):
    x_h, y_h = x / grid_step_size + .5, y / grid_step_size + .5
    i, j = int(floor(x_h) + grid_steps // 2), int(floor(y_h) + grid_steps // 2)
    x_loc, y_loc = x_h - floor(x_h) - .5, y_h - floor(y_h) - .5
    # centered to -.5 to 5, not 0 to 1, as formulas use offset from cell center
    # TODO: get rid of this deoffsetting/reoffsetting festival

    wx0, wy0 = .75 - x_loc**2, .75 - y_loc**2  # fx1, fy1
    wxP, wyP = (.5 + x_loc)**2 / 2, (.5 + y_loc)**2 / 2  # fx2**2/2, fy2**2/2
    wxM, wyM = (.5 - x_loc)**2 / 2, (.5 - y_loc)**2 / 2  # fx3**2/2, fy3**2/2

    wMP, w0P, wPP = wxM * wyP, wx0 * wyP, wxP * wyP
    wM0, w00, wP0 = wxM * wy0, wx0 * wy0, wxP * wy0
    wMM, w0M, wPM = wxM * wyM, wx0 * wyM, wxP * wyM

    return i, j, wMP, w0P, wPP, wM0, w00, wP0, wMM, w0M, wPM


@numba.jit(inline=True)
def interp9(a, i, j, wMP, w0P, wPP, wM0, w00, wP0, wMM, w0M, wPM):
    return (
        a[i - 1, j + 1] * wMP + a[i + 0, j + 1] * w0P + a[i + 1, j + 1] * wPP +
        a[i - 1, j + 0] * wM0 + a[i + 0, j + 0] * w00 + a[i + 1, j + 0] * wP0 +
        a[i - 1, j - 1] * wMM + a[i + 0, j - 1] * w0M + a[i + 1, j - 1] * wPM
    )


@numba.jit(inline=True)
def deposit9(a, i, j, val, wMP, w0P, wPP, wM0, w00, wP0, wMM, w0M, wPM):
    # atomic +=, thread-safe
    numba.cuda.atomic.add(a, (i - 1, j + 1), val * wMP)
    numba.cuda.atomic.add(a, (i + 0, j + 1), val * w0P)
    numba.cuda.atomic.add(a, (i + 1, j + 1), val * wPP)
    numba.cuda.atomic.add(a, (i - 1, j + 0), val * wM0)
    numba.cuda.atomic.add(a, (i + 0, j + 0), val * w00)
    numba.cuda.atomic.add(a, (i + 1, j + 0), val * wP0)
    numba.cuda.atomic.add(a, (i - 1, j - 1), val * wMM)
    numba.cuda.atomic.add(a, (i + 0, j - 1), val * w0M)
    numba.cuda.atomic.add(a, (i + 1, j - 1), val * wPM)


@numba.cuda.jit
def deposit_kernel(grid_steps, grid_step_size,
                   fine_grid, c_x_offt, c_y_offt,
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

        i, j, wMP, w0P, wPP, wM0, w00, wP0, wMM, w0M, wPM = weights(
            x, y, grid_steps, grid_step_size
        )
        deposit9(out_ro, i, j, dro, wMP, w0P, wPP, wM0, w00, wP0, wMM, w0M, wPM)
        deposit9(out_jx, i, j, djx, wMP, w0P, wPP, wM0, w00, wP0, wMM, w0M, wPM)
        deposit9(out_jy, i, j, djy, wMP, w0P, wPP, wM0, w00, wP0, wMM, w0M, wPM)
        deposit9(out_jz, i, j, djz, wMP, w0P, wPP, wM0, w00, wP0, wMM, w0M, wPM)


def deposit(cfg, ro_initial,
            grid_steps, grid_step_size, fine_grid, x_offt_new, y_offt_new,
            m, q, px_new, py_new, pz_new,
            A_weights, B_weights, C_weights, D_weights,
            indices_prev, indices_next, virtplasma_smallness_factor):
    ro = cp.zeros((grid_steps, grid_steps))
    jx = cp.zeros((grid_steps, grid_steps))
    jy = cp.zeros((grid_steps, grid_steps))
    jz = cp.zeros((grid_steps, grid_steps))
    deposit_kernel[cfg](grid_steps, grid_step_size, fine_grid,
                        x_offt_new, y_offt_new, m, q, px_new, py_new, pz_new,
                        A_weights, B_weights, C_weights, D_weights,
                        indices_prev, indices_next,
                        virtplasma_smallness_factor,
                        ro, jx, jy, jz)
    ro += ro_initial  # Do it last to preserve more float precision
    numba.cuda.synchronize()
    return ro, jx, jy, jz


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
def move_smart_kernel(xi_step_size, reflect_boundary,
                      grid_step_size, grid_steps,
                      ms, qs,
                      x_init, y_init,
                      old_x_offt, old_y_offt,
                      estimated_x_offt, estimated_y_offt,
                      old_px, old_py, old_pz,
                      Ex_avg, Ey_avg, Ez_avg, Bx_avg, By_avg, Bz_avg,
                      new_x_offt, new_y_offt, new_px, new_py, new_pz):
    index = numba.cuda.grid(1)
    stride = numba.cuda.blockDim.x * numba.cuda.gridDim.x
    for k in range(index, ms.size, stride):
        m, q = ms[k], qs[k]

        opx, opy, opz = old_px[k], old_py[k], old_pz[k]
        px, py, pz = opx, opy, opz
        x_offt, y_offt = old_x_offt[k], old_y_offt[k]

        x_halfstep = x_init[k] + (old_x_offt[k] + estimated_x_offt[k]) / 2
        y_halfstep = y_init[k] + (old_y_offt[k] + estimated_y_offt[k]) / 2
        i, j, wMP, w0P, wPP, wM0, w00, wP0, wMM, w0M, wPM = weights(
            x_halfstep, y_halfstep, grid_steps, grid_step_size
        )
        Ex = interp9(Ex_avg, i, j, wMP, w0P, wPP, wM0, w00, wP0, wMM, w0M, wPM)
        Ey = interp9(Ey_avg, i, j, wMP, w0P, wPP, wM0, w00, wP0, wMM, w0M, wPM)
        Ez = interp9(Ez_avg, i, j, wMP, w0P, wPP, wM0, w00, wP0, wMM, w0M, wPM)
        Bx = interp9(Bx_avg, i, j, wMP, w0P, wPP, wM0, w00, wP0, wMM, w0M, wPM)
        By = interp9(By_avg, i, j, wMP, w0P, wPP, wM0, w00, wP0, wMM, w0M, wPM)
        Bz = 0  # Bz = 0 for now

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

        x_offt += px / (gamma_m - pz) * xi_step_size  # no mixing with x_init
        y_offt += py / (gamma_m - pz) * xi_step_size  # no mixing with y_init

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


def move_smart(cfg, xi_step_size, reflect_boundary, grid_step_size, grid_steps,
               m, q, x_init, y_init, x_prev_offt, y_prev_offt,
               estimated_x_offt, estimated_y_offt, px_prev, py_prev, pz_prev,
               Ex_avg, Ey_avg, Ez_avg, Bx_avg, By_avg, Bz_avg):
    x_offt_new = cp.zeros_like(x_prev_offt)
    y_offt_new = cp.zeros_like(y_prev_offt)
    px_new = cp.zeros_like(px_prev)
    py_new = cp.zeros_like(py_prev)
    pz_new = cp.zeros_like(pz_prev)
    move_smart_kernel[cfg](xi_step_size, reflect_boundary,
                           grid_step_size, grid_steps,
                           m.ravel(), q.ravel(),
                           x_init.ravel(), y_init.ravel(),
                           x_prev_offt.ravel(), y_prev_offt.ravel(),
                           estimated_x_offt.ravel(), estimated_y_offt.ravel(),
                           px_prev.ravel(), py_prev.ravel(), pz_prev.ravel(),
                           Ex_avg, Ey_avg, Ez_avg, Bx_avg, By_avg, Bz_avg,
                           x_offt_new.ravel(), y_offt_new.ravel(),
                           px_new.ravel(), py_new.ravel(), pz_new.ravel())
    numba.cuda.synchronize()
    return x_offt_new, y_offt_new, px_new, py_new, pz_new


class GPUMonolith:
    cfg = (19, 384)  # empirical guess for a GTX 1070 Ti

    def __init__(self, config,
                 pl_x_init, pl_y_init, pl_x_offt, pl_y_offt,
                 pl_px, pl_py, pl_pz, pl_m, pl_q,
                 A_weights, B_weights, C_weights, D_weights,
                 fine_grid_init,
                 indices_prev, indices_next):
        # TODO: compare shapes, not sizes
        self.Nc = Nc = int(sqrt(pl_q.size))
        assert Nc**2 == pl_x_init.size == pl_y_init.size
        assert Nc**2 == pl_x_offt.size == pl_y_offt.size
        assert Nc**2 == pl_px.size == pl_py.size == pl_pz.size
        assert Nc**2 == pl_m.size == pl_q.size

        # virtual particles should not reach the window pre-boundary cells
        assert config.reflect_padding_steps > config.plasma_coarseness + 1
        # the alternative is to reflect after plasma virtualization

        self.grid_steps = N = config.grid_steps
        assert self.grid_steps % 2 == 1
        self.xi_step_size = config.xi_step_size
        #self.grid_step_size = config.window_width / (config.grid_steps - 1)
        self.grid_step_size = config.grid_step_size
        self.subtraction_trick = config.field_solver_subtraction_trick
        self.reflect_boundary = self.grid_step_size * (
            config.grid_steps / 2 - config.reflect_padding_steps
        )

        self.virtplasma_smallness_factor = 1 / (config.plasma_coarseness *
                                                config.plasma_fineness)**2

        self._x_init = cp.array(pl_x_init)
        self._y_init = cp.array(pl_y_init)
        self._fine_grid = cp.array(fine_grid_init)

        self._m = cp.zeros((Nc, Nc))
        self._q = cp.zeros((Nc, Nc))
        self._x_prev_offt = cp.zeros((Nc, Nc))
        self._y_prev_offt = cp.zeros((Nc, Nc))
        self._px_prev = cp.zeros((Nc, Nc))
        self._py_prev = cp.zeros((Nc, Nc))
        self._pz_prev = cp.zeros((Nc, Nc))

        self._x_new_offt = cp.zeros((Nc, Nc))
        self._y_new_offt = cp.zeros((Nc, Nc))
        self._px_new = cp.zeros((Nc, Nc))
        self._py_new = cp.zeros((Nc, Nc))
        self._pz_new = cp.zeros((Nc, Nc))

        self._A_weights = cp.array(A_weights)
        self._B_weights = cp.array(B_weights)
        self._C_weights = cp.array(C_weights)
        self._D_weights = cp.array(D_weights)
        self._indices_prev = cp.array(indices_prev)
        self._indices_next = cp.array(indices_next)

        self.mixed_solver = MixedSolver(N, self.grid_step_size, self.subtraction_trick, self.cfg)
        self._Ex = cp.zeros((N, N))
        self._Ey = cp.zeros((N, N))
        self._Bx = cp.zeros((N, N))
        self._By = cp.zeros((N, N))

        self.dirichlet_solver = DirichletSolver(N, self.grid_step_size)
        self._Ez = cp.zeros((N, N))

        self._Bz = cp.zeros((N, N))

        self._ro_initial = cp.zeros((N, N))
        self._ro = cp.zeros((N, N))
        self._jx = cp.zeros((N, N))
        self._jy = cp.zeros((N, N))
        self._jz = cp.zeros((N, N))

        self._beam_ro = cp.zeros((N, N))

        self._jx_prev = cp.zeros((N, N))
        self._jy_prev = cp.zeros((N, N))

        self._Ex_prev = cp.zeros((N, N))
        self._Ey_prev = cp.zeros((N, N))
        self._Ez_prev = cp.zeros((N, N))
        self._Bx_prev = cp.zeros((N, N))
        self._By_prev = cp.zeros((N, N))
        self._Bz_prev = cp.zeros((N, N))

        # Allow accessing `gpu_monolith.ro`
        # without typing the whole `gpu_monolith._ro.copy_to_host()`.
        # and setting its value with `gpu_monolith.ro = ...`
        gpu_array_type = type(self._m)
        for attrname in dir(self):
            if attrname.startswith('_'):
                attrname_unpref = attrname[1:]
                attr = getattr(self, attrname)
                if isinstance(attr, gpu_array_type):
                    # a separate func for copying attrname into another closure
                    def hook_property(cls, attrname):
                        def getter(self):
                            return getattr(self, attrname).copy_to_host()
                        def setter(self, val):
                            getattr(self, attrname)[...] = val
                        setattr(cls, attrname_unpref, property(getter, setter))
                    hook_property(type(self), attrname)
        # Allow accessing `gpu_monolith.ro`
        # without typing the whole `gpu_monolith._ro.get()`.
        # and setting its value with `gpu_monolith.ro = ...`
                if isinstance(attr, cp.ndarray):
                    # a separate func for copying attrname into another closure
                    def hook_property(cls, attrname):
                        def getter(self):
                            return getattr(self, attrname).get()
                        def setter(self, val):
                            getattr(self, attrname)[...] = val
                        setattr(cls, attrname_unpref, property(getter, setter))
                    hook_property(type(self), attrname)


    def load(self, beam_ro,
             pl_x_offt, pl_y_offt, pl_px, pl_py, pl_pz, pl_m, pl_q,
             Ex, Ey, Ez, Bx, By, Bz, jx, jy):
        self._beam_ro[...] = cp.array(beam_ro)

        self._m[...] = cp.array(pl_m)
        self._q[...] = cp.array(pl_q)
        self._x_prev_offt[...] = cp.array(pl_x_offt)
        self._y_prev_offt[...] = cp.array(pl_y_offt)
        self._px_prev[...] = cp.array(pl_px)
        self._py_prev[...] = cp.array(pl_py)
        self._pz_prev[...] = cp.array(pl_pz)

        self._Ex[...] = cp.array(Ex)
        self._Ey[...] = cp.array(Ey)
        self._Ez[...] = cp.array(Ez)
        self._Bx[...] = cp.array(Bx)
        self._By[...] = cp.array(By)
        self._Bz[...] = cp.array(Bz)
        self._jx[...] = cp.array(jx)
        self._jy[...] = cp.array(jy)

        numba.cuda.synchronize()


    def move_estimate_wo_fields(self,
                                m, x_init, y_init, x_prev_offt, y_prev_offt,
                                px_prev, py_prev, pz_prev):
        estimated_x_offt = cp.zeros_like(x_init)
        estimated_y_offt = cp.zeros_like(y_init)
        move_estimate_wo_fields_kernel[self.cfg](self.xi_step_size,
                                                 self.reflect_boundary,
                                                 m.ravel(),
                                                 x_init.ravel(),
                                                 y_init.ravel(),
                                                 x_prev_offt.ravel(),
                                                 y_prev_offt.ravel(),
                                                 px_prev.ravel(),
                                                 py_prev.ravel(),
                                                 pz_prev.ravel(),
                                                 estimated_x_offt.ravel(),
                                                 estimated_y_offt.ravel())
        numba.cuda.synchronize()
        return estimated_x_offt, estimated_y_offt


    def initial_deposition(self, pl_x_offt, pl_y_offt,
                           pl_px, pl_py, pl_pz, pl_m, pl_q):
        # Don't allow initial speeds for calculations with background ions
        assert np.array_equiv(pl_px, 0)
        assert np.array_equiv(pl_py, 0)
        assert np.array_equiv(pl_pz, 0)

        ro_initial = cp.zeros((self.grid_steps, self.grid_steps))
        ro_electrons_initial, _, _, _ = deposit(
            self.cfg, ro_initial, self.grid_steps, self.grid_step_size,
            self._fine_grid,
            pl_x_offt, pl_y_offt, pl_m, pl_q, pl_px, pl_py, pl_pz,
            self._A_weights, self._B_weights, self._C_weights, self._D_weights,
            self._indices_prev, self._indices_next,
            self.virtplasma_smallness_factor)

        self._ro_initial = -ro_electrons_initial  # Right on the GPU, huh
        numba.cuda.synchronize()


    def step(self, beam_ro):
        self.reload(beam_ro)

        m = self._m
        x_init, y_init = self._x_init, self._y_init
        x_prev_offt, y_prev_offt = self.x_prev_offt, self.y_prev_offt
        px_prev, py_prev, pz_prev = self._px_prev, self.py_prev, self.pz_prev
        Ex_avg = self._Ex.copy()

        # TODO: use regular pusher?
        estimated_x_offt, estimated_y_offt = self.move_estimate_wo_fields(
            m, x_init, y_init, x_prev_offt, y_prev_offt,
            px_prev, py_prev, pz_prev
        )

        x_offt_new, y_offt_new, px_new, py_new, pz_new = move_smart(
            self.cfg, self.xi_step_size, self.reflect_boundary,
            self.grid_step_size, self.grid_steps,
            self._m, self._q, self._x_init, self._y_init,
            self._x_prev_offt, self._y_prev_offt,
            estimated_x_offt, estimated_y_offt,
            self._px_prev, self._py_prev, self._pz_prev,
            # no halfstep-averaged fields yet
            self._Ex_prev, self._Ey_prev, self._Ez_prev,
            self._Bx_prev, self._By_prev, Bz_avg=0,
        )
        self._ro[...], self._jx[...], self._jy[...], self._jz[...] = deposit(
            self.cfg, self._ro_initial, self.grid_steps, self.grid_step_size,
            self._fine_grid, x_offt_new, y_offt_new, self._m, self._q,
            px_new, py_new, pz_new,
            self._A_weights, self._B_weights, self._C_weights, self._D_weights,
            self._indices_prev, self._indices_next,
            self.virtplasma_smallness_factor)

        self._Ex[...], self._Ey[...], self._Bx[...], self._By[...] = \
            calculate_Ex_Ey_Bx_By(self.grid_step_size, self.xi_step_size,
                                  self.subtraction_trick, self.mixed_solver,
                                  # no halfstep-averaged fields yet
                                  self._Ex_prev, self._Ey_prev,
                                  self._Bx_prev, self._By_prev,
                                  self._beam_ro, self._ro,
                                  self._jx, self._jy, self._jz,
                                  self._jx_prev, self._jy_prev)
        self._Ez[...] = calculate_Ez(self.dirichlet_solver,
                                     self.grid_step_size, self._jx, self._jy)
        # Bz = 0 for now
        Ex_avg = (self._Ex + self._Ex_prev) / 2
        Ey_avg = (self._Ey + self._Ey_prev) / 2
        Ez_avg = (self._Ez + self._Ez_prev) / 2
        Bx_avg = (self._Bx + self._Bx_prev) / 2
        By_avg = (self._By + self._By_prev) / 2

        x_offt_new, y_offt_new, px_new, py_new, pz_new = move_smart(
            self.cfg, self.xi_step_size, self.reflect_boundary,
            self.grid_step_size, self.grid_steps,
            self._m, self._q, self._x_init, self._y_init,
            self._x_prev_offt, self._y_prev_offt,
            x_offt_new, y_offt_new,
            self._px_prev, self._py_prev, self._pz_prev,
            Ex_avg, Ey_avg, Ez_avg, Bx_avg, By_avg, Bz_avg=0
        )
        self._ro[...], self._jx[...], self._jy[...], self._jz[...] = deposit(
            self.cfg, self._ro_initial, self.grid_steps, self.grid_step_size,
            self._fine_grid, x_offt_new, y_offt_new, self._m, self._q,
            px_new, py_new, pz_new,
            self._A_weights, self._B_weights, self._C_weights, self._D_weights,
            self._indices_prev, self._indices_next,
            self.virtplasma_smallness_factor)
        self._Ex[...], self._Ey[...], self._Bx[...], self._By[...] = \
            calculate_Ex_Ey_Bx_By(self.grid_step_size, self.xi_step_size,
                                  self.subtraction_trick, self.mixed_solver,
                                  Ex_avg, Ey_avg, Bx_avg, By_avg,
                                  self._beam_ro, self._ro,
                                  self._jx, self._jy, self._jz,
                                  self._jx_prev, self._jy_prev)
        self._Ez[...] = calculate_Ez(self.dirichlet_solver,
                                     self.grid_step_size, self._jx, self._jy)
        # Bz = 0 for now
        Ex_avg = (self._Ex + self._Ex_prev) / 2
        Ey_avg = (self._Ey + self._Ey_prev) / 2
        Ez_avg = (self._Ez + self._Ez_prev) / 2
        Bx_avg = (self._Bx + self._Bx_prev) / 2
        By_avg = (self._By + self._By_prev) / 2

        x_offt_new, y_offt_new, px_new, py_new, pz_new = move_smart(
            self.cfg, self.xi_step_size, self.reflect_boundary,
            self.grid_step_size, self.grid_steps,
            self._m, self._q, self._x_init, self._y_init,
            self._x_prev_offt, self._y_prev_offt,
            x_offt_new, y_offt_new,
            self._px_prev, self._py_prev, self._pz_prev,
            Ex_avg, Ey_avg, Ez_avg, Bx_avg, By_avg, Bz_avg=0
        )
        self._ro[...], self._jx[...], self._jy[...], self._jz[...] = deposit(
            self.cfg, self._ro_initial, self.grid_steps, self.grid_step_size,
            self._fine_grid, x_offt_new, y_offt_new, self._m, self._q,
            px_new, py_new, pz_new,
            self._A_weights, self._B_weights, self._C_weights, self._D_weights,
            self._indices_prev, self._indices_next,
            self.virtplasma_smallness_factor)

        # TODO: what do we need that roj_new for, jx_prev/jy_prev only?

        self._x_new_offt[...] = x_offt_new
        self._y_new_offt[...] = y_offt_new
        self._px_new[...] = px_new
        self._py_new[...] = py_new
        self._pz_new[...] = pz_new

    def reload(self, beam_ro):
        self._beam_ro[...] = cp.array(beam_ro)

        # TODO: array relabeling instead of copying?..

        # Intact: self._m, self._q
        self._x_prev_offt[...] = cp.array(self._x_new_offt)
        self._y_prev_offt[...] = cp.array(self._y_new_offt)
        self._px_prev[...] = cp.array(self._px_new)
        self._py_prev[...] = cp.array(self._py_new)
        self._pz_prev[...] = cp.array(self._pz_new)

        self._Ex_prev[...] = self._Ex
        self._Ey_prev[...] = self._Ey
        self._Ez_prev[...] = self._Ez
        self._Bx_prev[...] = self._Bx
        self._By_prev[...] = self._By
        self._Bz_prev[...] = self._Bz
        self._jx_prev[...] = self._jx
        self._jy_prev[...] = self._jy

        numba.cuda.synchronize()


# TODO: try local arrays for bet (on larger grid sizes)?
# TODO: specialize for specific grid sizes?
# TODO: try going syncless


def make_coarse_plasma_grid(steps, step_size, coarseness):
    assert coarseness == int(coarseness)
    plasma_step = step_size * coarseness
    right_half = np.arange(steps // (coarseness * 2)) * plasma_step
    left_half = -right_half[:0:-1]  # invert, reverse, drop zero
    plasma_grid = np.concatenate([left_half, right_half])
    assert(np.array_equal(plasma_grid, -plasma_grid[::-1]))
    return plasma_grid


def make_fine_plasma_grid(steps, step_size, fineness):
    assert fineness == int(fineness)
    plasma_step = step_size / fineness
    if fineness % 2:  # some on zero axes, none on cell corners
        right_half = np.arange(steps // 2 * fineness) * plasma_step
        left_half = -right_half[:0:-1]  # invert, reverse, drop zero
        plasma_grid = np.concatenate([left_half, right_half])
    else:  # none on zero axes, none on cell corners
        right_half = (.5 + np.arange(steps // 2 * fineness)) * plasma_step
        left_half = -right_half[::-1]  # invert, reverse
        plasma_grid = np.concatenate([left_half, right_half])
    assert(np.array_equal(plasma_grid, -plasma_grid[::-1]))
    return plasma_grid


def plasma_make(steps, cell_size, coarseness=2, fineness=2):
    coarse_step = cell_size * coarseness

    # Make two initial grids of plasma particles, coarse and fine.
    # Coarse is the one that will evolve and fine is the one to be bilinearly
    # interpolated from the coarse one based on the initial positions.

    coarse_grid = make_coarse_plasma_grid(steps, cell_size, coarseness)
    coarse_grid_xs, coarse_grid_ys = coarse_grid[:, None], coarse_grid[None, :]

    fine_grid = make_fine_plasma_grid(steps, cell_size, fineness)
    fine_grid_xs, fine_grid_ys = fine_grid[:, None], fine_grid[None, :]

    Nc = len(coarse_grid)

    # Create plasma particles on that grids
    coarse_electrons_x_init = np.broadcast_to(coarse_grid_xs, (Nc, Nc))
    coarse_electrons_y_init = np.broadcast_to(coarse_grid_ys, (Nc, Nc))
    coarse_electrons_x_offt = np.zeros((Nc, Nc))
    coarse_electrons_y_offt = np.zeros((Nc, Nc))
    coarse_electrons_px = np.zeros((Nc, Nc))
    coarse_electrons_py = np.zeros((Nc, Nc))
    coarse_electrons_pz = np.zeros((Nc, Nc))
    coarse_electrons_m = np.ones((Nc, Nc)) * ELECTRON_MASS * coarseness**2
    coarse_electrons_q = np.ones((Nc, Nc)) * ELECTRON_CHARGE * coarseness**2

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

    # TODO: decide on a flat-or-square plasma
    return (coarse_electrons_x_init, coarse_electrons_y_init,
            coarse_electrons_x_offt, coarse_electrons_y_offt,
            coarse_electrons_px, coarse_electrons_py, coarse_electrons_pz,
            coarse_electrons_m, coarse_electrons_q,
            A_weights, B_weights, C_weights, D_weights,
            fine_grid, indices_prev, indices_next)


max_zn = 0
def diags_ro_zn(config, ro):
    global max_zn

    sigma = 0.25 / config.grid_step_size
    blurred = scipy.ndimage.gaussian_filter(ro, sigma=sigma)
    hf = ro - blurred
    zn = np.abs(hf).mean() / 4.23045376e-04
    max_zn = max(max_zn, zn)
    return max_zn


def diags_peak_msg(Ez_00_history):
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


def diagnostics(gpu, config, xi_i, Ez_00_history):
    xi = -xi_i * config.xi_step_size

    Ez_00 = Ez_00_history[-1]
    peak_report = diags_peak_msg(Ez_00_history)

    ro = gpu.ro
    max_zn = diags_ro_zn(config, gpu.ro)
    diags_ro_slice(config, xi_i, xi, ro)

    print(f'xi={xi:+.4f} {Ez_00:+.4e}|{peak_report}|zn={max_zn:.3f}')
    sys.stdout.flush()


def init(config):
    grid = ((np.arange(config.grid_steps) - config.grid_steps // 2)
            * config.grid_step_size)
    xs, ys = grid[:, None], grid[None, :]

    pl_x_init, pl_y_init, pl_x_offt, pl_y_offt, pl_px, pl_py, pl_pz, pl_m, pl_q, *virt_params = plasma_make(
        config.grid_steps - config.plasma_padding_steps * 2,
        config.grid_step_size,
        coarseness=config.plasma_coarseness, fineness=config.plasma_fineness
    )

    gpu = GPUMonolith(config,
                      pl_x_init, pl_y_init, pl_x_offt, pl_y_offt,
                      pl_px, pl_py, pl_pz, pl_m, pl_q, *virt_params)
    gpu.load(0, pl_x_offt, pl_y_offt,
             pl_px, pl_py, pl_pz, pl_m, pl_q,
             0, 0, 0, 0, 0, 0, 0, 0)
    gpu.initial_deposition(pl_x_offt, pl_y_offt,
                           pl_px, pl_py, pl_pz, pl_m, pl_q)

    return gpu, xs, ys


# TODO: fold init, load, initial_deposition into GPUMonolith.__init__?
def main():
    import config
    gpu, xs, ys = init(config)
    Ez_00_history = []

    for xi_i in range(config.xi_steps):
        beam_ro = config.beam(xi_i, xs, ys)

        gpu.step(beam_ro)

        Ez_00 = gpu.Ez[config.grid_steps // 2, config.grid_steps // 2]
        Ez_00_history.append(Ez_00)

        time_for_diags = xi_i % config.diagnostics_each_N_steps == 0
        last_step = xi_i == config.xi_steps - 1
        if time_for_diags or last_step:
            diagnostics(gpu, config, xi_i, Ez_00_history)


if __name__ == '__main__':
    main()

