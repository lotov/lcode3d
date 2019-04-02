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

# Should be detectable with newer cupy (>6.0.0b2) as
# WARP_SIZE = cp.cuda.Device(0).attributes['WarpSize']
# But as of 2019 it's always 32. It's even a hardcoded value in cupy.
WARP_SIZE = 32

ELECTRON_CHARGE = -1
ELECTRON_MASS = 1


# Grouping GPU arrays, with optional transparent RAM<->GPU copying #

class GPUArrays:
    """
    A convenient way to group several GPU arrays and access them with a dot.
    `x = GPUArrays(something=numpy_array, something_else=another_array)`
    will create `x` with `x.something` and `x.something_else` being GPU arrays.
    Do not add more attributes later, specify them all at construction time.
    """
    def __init__(self, **kwargs):
        """
        Convert the keyword arguments to `cupy` arrays and assign them
        to the object attributes.
        Amounts to, e.g., `self.something = cp.asarray(numpy_array)`,
        and `self.something_else = cp.asarray(another_array)`,
        see class doctring.
        """
        for name, array in kwargs.items():
            setattr(self, name, cp.asarray(array))


# NOTE: The implementation may be complicated, but the usage is simple.
class GPUArraysView:
    """
    This is a magical wrapper around GPUArrays that handles GPU-RAM data
    transfer transparently.
    Accessing `view.something` will automatically copy array to host RAM,
    setting `view.something = ...` will copy the changes back to GPU RAM.
    Usage: `view = GPUArraysView(gpu_arrays); view.something`
    Do not add more attributes later, specify them all at construction time.
    NOTE: repeatedly accessing an attribute will result in repeated copying!
    """
    def __init__(self, gpu_arrays):
        """
        Wrap `gpu_arrays` and transparently copy data to/from GPU.
        """
        # Could've been written as `self._arrs = gpu_arrays`
        # if only `__setattr__` was not overwritten!
        # `super(GPUArraysView) is the proper way to obtain the parent class
        # (`object`), which has a regular boring `__setattr__` that we can use.
        super(GPUArraysView, self).__setattr__('_arrs', gpu_arrays)

    def __dir__(self):
        """
        Make `dir()` also show the wrapped `gpu_arrays` attributes.
        """
        # See `GPUArraysView.__init__` for the explanation how we access the
        # parent's plain `__dir__()` implementation (and avoid recursion).
        return list(set(super(GPUArraysView, self).__dir__() +
                        dir(self._arrs)))

    def __getattr__(self, attrname):
        """
        Intercept access to (missing) attributes, access the wrapped object
        attributes instead and copy the arrays from GPU to RAM.
        """
        return getattr(self._arrs, attrname).get()  # auto-copies to host RAM

    def __setattr__(self, attrname, value):
        """
        Intercept setting attributes, access the wrapped object attributes
        instead and reassign their contents, copying the arrays from RAM
        to GPU in the process.
        """
        getattr(self._arrs, attrname)[...] = value  # copies to GPU RAM
        # TODO: just copy+reassign it without preserving identity and shape?


# Solving Laplace equation with Dirichlet boundary conditions (Ez) #

def dst2d(a):
    """
    Calculate DST-Type1-2D, jury-rigged from anti-symmetrically-padded rFFT.
    """
    assert a.shape[0] == a.shape[1]
    N = a.shape[0]
    #                                    / 0  0  0  0  0  0 \
    #  0  0  0  0                       |  0 /1  2\ 0 -2 -1  |
    #  0 /1  2\ 0   anti-symmetrically  |  0 \3  4/ 0 -4 -3  |
    #  0 \3  4/ 0       padded to       |  0  0  0  0  0  0  |
    #  0  0  0  0                       |  0 -3 -4  0 +4 +3  |
    #                                    \ 0 -1 -2  0 +2 +1 /
    p = cp.zeros((2 * N + 2, 2 * N + 2))
    p[1:N+1, 1:N+1], p[1:N+1, N+2:] = a,             -cp.fliplr(a)
    p[N+2:,  1:N+1], p[N+2:,  N+2:] = -cp.flipud(a), +cp.fliplr(cp.flipud(a))

    # after padding: rFFT-2D, cut out the top-left segment, take -real part
    return -cp.fft.rfft2(p)[1:N+1, 1:N+1].real


@cp.memoize()
def dirichlet_matrix(grid_steps, grid_step_size):
    """
    Calculate a magical matrix that solves the Laplace equation
    if you elementwise-multiply the RHS by it "in DST-space".
    See Samarskiy-Nikolaev, p. 187.
    """
    # mul[i, j] = 1 / (lam[i] + lam[j])
    # lam[k] = 4 / h**2 * sin(k * pi * h / (2 * L))**2, where L = h * (N - 1)
    k = cp.arange(1, grid_steps - 1)
    lam = 4 / grid_step_size**2 * cp.sin(k * cp.pi / (2 * (grid_steps - 1)))**2
    lambda_i, lambda_j = lam[:, None], lam[None, :]
    mul = 1 / (lambda_i + lambda_j)
    return mul / (2 * (grid_steps - 1))**2  # additional 2xDST normalization


def calculate_Ez(config, jx, jy):
    """
    Calculate Ez as iDST2D(dirichlet_matrix * DST2D(djx/dx + djy/dy)).
    """
    # 0. Calculate RHS (NOTE: it is smaller by 1 on each side).
    # NOTE: use gradient instead if available (cupy doesn't have gradient yet).
    djx_dx = jx[2:, 1:-1] - jx[:-2, 1:-1]
    djy_dy = jy[1:-1, 2:] - jy[1:-1, :-2]
    rhs_inner = -(djx_dx + djy_dy) / (config.grid_step_size * 2)  # -?

    # 1. Apply DST-Type1-2D (Discrete Sine Transform Type 1 2D) to the RHS.
    f = dst2d(rhs_inner)

    # 2. Multiply f by the special matrix that does the job and normalizes.
    f *= dirichlet_matrix(config.grid_steps, config.grid_step_size)

    # 3. Apply iDST-Type1-2D (Inverse Discrete Sine Transform Type 1 2D).
    #    We don't have to define a separate iDST function, because
    #    unnormalized DST-Type1 is its own inverse, up to a factor 2(N+1)
    #    and we take all scaling matters into account with a single factor
    #    hidden inside dirichlet_matrix.
    Ez_inner = dst2d(f)
    Ez = cp.pad(Ez_inner, 1, 'constant', constant_values=0)
    numba.cuda.synchronize()
    return Ez


# Solving Laplace or Helmholtz equation with mixed boundary conditions #

# jury-rigged from padded rFFT
def mix2d(a):
    """
    Calculate a DST-DCT-hybrid transform
    (DST in first direction, DCT in second one),
    jury-rigged from padded rFFT
    (anti-symmetrically in first direction, symmetrically in second direction).
    """
    # NOTE: LCODE 3D uses x as the first direction, thus the confision below.
    M, N = a.shape
    #                                  /(0  1  2  0)-2 -1 \      +---->  x
    #  / 1  2 \                       | (0  3  4  0)-4 -3  |     |      (M)
    #  | 3  4 |  mixed-symmetrically  | (0  5  6  0)-6 -5  |     |
    #  | 5  6 |       padded to       | (0  7  8  0)-8 -7  |     v
    #  \ 7  8 /                       |  0 +5 +6  0 -6 -5  |
    #                                  \ 0 +3 +4  0 -4 -3 /      y (N)
    p = cp.zeros((2 * M + 2, 2 * N - 2))  # wider than before
    p[1:M+1, :N] = a
    p[M+2:2*M+2, :N] = -cp.flipud(a)  # flip to right on drawing above
    p[1:M+1, N-1:2*N-2] = cp.fliplr(a)[:, :-1]  # flip down on drawing above
    p[M+2:2*M+2, N-1:2*N-2] = -cp.flipud(cp.fliplr(a))[:, :-1]
    # Note: the returned array is wider than the input array, it is padded
    # with zeroes (depicted above as a square region marked with round braces).
    return -cp.fft.rfft2(p)[:M+2, :N].imag  # FFT, cut a corner with 0s, -imag


@cp.memoize()
def mixed_matrix(grid_steps, grid_step_size, subtraction_trick):
    """
    Calculate a magical matrix that solves the Laplace or Helmholtz equation
    (subtraction_trick=True and subtraction_trick=False correspondingly)
    if you elementwise-multiply the RHS by it "in DST-DCT-transformed-space".
    See Samarskiy-Nikolaev, p. 189 and around.
    """
    # mul[i, j] = 1 / (lam[i] + lam[j])
    # lam[k] = 4 / h**2 * sin(k * pi * h / (2 * L))**2, where L = h * (N - 1)
    # but k for lam_i spans from 1..N-2, while k for lam_j covers 0..N-1
    ki, kj = cp.arange(1, grid_steps - 1), cp.arange(grid_steps)
    li = 4 / grid_step_size**2 * cp.sin(ki * cp.pi / (2 * (grid_steps - 1)))**2
    lj = 4 / grid_step_size**2 * cp.sin(kj * cp.pi / (2 * (grid_steps - 1)))**2
    lambda_i, lambda_j = li[:, None], lj[None, :]
    mul = 1 / (lambda_i + lambda_j + (1 if subtraction_trick else 0))
    return mul / (2 * (grid_steps - 1))**2  # additional 2xDST normalization


def dx_dy(arr, grid_step_size):
    """
    Calculate x and y derivatives simultaneously (like np.gradient does).
    NOTE: use gradient instead if available (cupy doesn't have gradient yet).
    NOTE: arrays are assumed to have zeros on the perimeter.
    """
    dx, dy = cp.zeros_like(arr), cp.zeros_like(arr)
    dx[1:-1, 1:-1] = arr[2:, 1:-1] - arr[:-2, 1:-1]  # arrays have 0s
    dy[1:-1, 1:-1] = arr[1:-1, 2:] - arr[1:-1, :-2]  # on the perimeter
    return dx / (grid_step_size * 2), dy / (grid_step_size * 2)


def calculate_Ex_Ey_Bx_By(config, Ex_avg, Ey_avg, Bx_avg, By_avg,
                          beam_ro, ro, jx, jy, jz, jx_prev, jy_prev):
    """
    Calculate transverse fields as iDST-DCT(mixed_matrix * DST-DCT(RHS.T)).T,
    with and without transposition depending on the field component.
    NOTE: density and currents are assumed to be zero on the perimeter
          (no plasma particles must reach the wall, so the reflection boundary
           must be closer to the center than the simulation window boundary
           minus the coarse plasma particle cloud width).
    """
    # 0. Calculate gradients and RHS.
    dro_dx, dro_dy = dx_dy(ro + beam_ro, config.grid_step_size)
    djz_dx, djz_dy = dx_dy(jz + beam_ro, config.grid_step_size)
    djx_dxi = (jx_prev - jx) / config.xi_step_size  # - ?
    djy_dxi = (jy_prev - jy) / config.xi_step_size  # - ?

    # Are we solving a Laplace equation or a Helmholtz one?
    subtraction_trick = config.field_solver_subtraction_trick
    Ex_rhs = -((dro_dx - djx_dxi) - Ex_avg * subtraction_trick)  # -?
    Ey_rhs = -((dro_dy - djy_dxi) - Ey_avg * subtraction_trick)
    Bx_rhs = +((djz_dy - djy_dxi) + Bx_avg * subtraction_trick)
    By_rhs = -((djz_dx - djx_dxi) - By_avg * subtraction_trick)

    # Boundary conditions application (for future reference, ours are zero):
    # rhs[:, 0] -= bound_bottom[:] * (2 / grid_step_size)
    # rhs[:, -1] += bound_top[:] * (2 / grid_step_size)

    # 1. Apply our mixed DCT-DST transform to RHS.
    Ey_f = mix2d(Ey_rhs[1:-1, :])[1:-1, :]

    # 2. Multiply f by the magic matrix.
    mix_mat = mixed_matrix(config.grid_steps, config.grid_step_size,
                           config.field_solver_subtraction_trick)
    Ey_f *= mix_mat

    # 3. Apply our mixed DCT-DST transform again.
    Ey = mix2d(Ey_f)

    # Likewise for other fields:
    Bx = mix2d(mix_mat * mix2d(Bx_rhs[1:-1, :])[1:-1, :])
    By = mix2d(mix_mat * mix2d(By_rhs.T[1:-1, :])[1:-1, :]).T
    Ex = mix2d(mix_mat * mix2d(Ex_rhs.T[1:-1, :])[1:-1, :]).T

    return Ex, Ey, Bx, By


# Pushing particles without any fields (used for initial halfstep estimation) #


def move_estimate_wo_fields(config,
                            m, x_init, y_init, prev_x_offt, prev_y_offt,
                            px, py, pz):
    """
    Move coarse plasma particles as if there were no fields.
    Also reflect the particles from `+-reflect_boundary`.
    """
    x, y = x_init + prev_x_offt, y_init + prev_y_offt
    gamma_m = cp.sqrt(m**2 + pz**2 + px**2 + py**2)

    x += px / (gamma_m - pz) * config.xi_step_size
    y += py / (gamma_m - pz) * config.xi_step_size

    reflect = config.reflect_boundary
    x[x >= +reflect] = +2 * reflect - x[x >= +reflect]
    x[x <= -reflect] = -2 * reflect - x[x <= -reflect]
    y[y >= +reflect] = +2 * reflect - y[y >= +reflect]
    y[y <= -reflect] = -2 * reflect - y[y <= -reflect]

    x_offt, y_offt = x - x_init, y - y_init

    numba.cuda.synchronize()
    return x_offt, y_offt


# Deposition and interpolation helper functions #

@numba.jit(inline=True)
def weights(x, y, grid_steps, grid_step_size):
    """
    Calculate the indices of a cell corresponding to the coordinates,
    and the coefficients of interpolation and deposition for this cell
    and 8 surrounding cells.
    The weights correspond to 2D triangluar shaped cloud (TSC2D).
    """
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
    """
    Collect value from a cell and 8 surrounding cells (using `weights` output).
    """
    return (
        a[i - 1, j + 1] * wMP + a[i + 0, j + 1] * w0P + a[i + 1, j + 1] * wPP +
        a[i - 1, j + 0] * wM0 + a[i + 0, j + 0] * w00 + a[i + 1, j + 0] * wP0 +
        a[i - 1, j - 1] * wMM + a[i + 0, j - 1] * w0M + a[i + 1, j - 1] * wPM
    )


@numba.jit(inline=True)
def deposit9(a, i, j, val, wMP, w0P, wPP, wM0, w00, wP0, wMM, w0M, wPM):
    """
    Deposit value into a cell and 8 surrounding cells (using `weights` output).
    """
    # This is like a[i - 1, j + 1] += val * wMP, except it is atomic
    # and incrementing the same cell by several threads will add up correctly.
    # CUDA Compute Capability 6.0+ is recommended for hardware atomics support.
    numba.cuda.atomic.add(a, (i - 1, j + 1), val * wMP)
    numba.cuda.atomic.add(a, (i + 0, j + 1), val * w0P)
    numba.cuda.atomic.add(a, (i + 1, j + 1), val * wPP)
    numba.cuda.atomic.add(a, (i - 1, j + 0), val * wM0)
    numba.cuda.atomic.add(a, (i + 0, j + 0), val * w00)
    numba.cuda.atomic.add(a, (i + 1, j + 0), val * wP0)
    numba.cuda.atomic.add(a, (i - 1, j - 1), val * wMM)
    numba.cuda.atomic.add(a, (i + 0, j - 1), val * w0M)
    numba.cuda.atomic.add(a, (i + 1, j - 1), val * wPM)


# Coarse and fine plasma initialization #

def make_coarse_plasma_grid(steps, step_size, coarseness):
    """
    Create initial coarse plasma particles coordinates
    (a single 1D grid for both x and y).
    """
    assert coarseness == int(coarseness)  # TODO: why?
    plasma_step = step_size * coarseness
    right_half = np.arange(steps // (coarseness * 2)) * plasma_step
    left_half = -right_half[:0:-1]  # invert, reverse, drop zero
    plasma_grid = np.concatenate([left_half, right_half])
    assert(np.array_equal(plasma_grid, -plasma_grid[::-1]))
    return plasma_grid


def make_fine_plasma_grid(steps, step_size, fineness):
    """
    Create initial fine plasma particles coordinates
    (a single 1D grid for both x and y).
    Avoids positioning particles at the cell edges and boundaries, example:
    `fineness=3` (and `coarseness=2`):
        +-----------+-----------+-----------+-----------+
        | .   .   . | .   .   . | .   .   . | .   .   . |
        |           |           |           |           |   . - fine particle
        | .   .   . | .   *   . | .   .   . | .   *   . |
        |           |           |           |           |   * - coarse particle
        | .   .   . | .   .   . | .   .   . | .   .   . |
        +-----------+-----------+-----------+-----------+
    `fineness=2` (and `coarseness=2`):
        +-------+-------+-------+-------+-------+
        | .   . | .   . | .   . | .   . | .   . |           . - fine particle
        |       |   *   |       |   *   |       |
        | .   . | .   . | .   . | .   . | .   . |           * - coarse particle
        +-------+-------+-------+-------+-------+
    """
    assert fineness == int(fineness)
    plasma_step = step_size / fineness
    if fineness % 2:  # some on zero axes, none on cell corners
        right_half = np.arange(steps // 2 * fineness) * plasma_step
        left_half = -right_half[:0:-1]  # invert, reverse, drop zero
    else:  # none on zero axes, none on cell corners
        right_half = (.5 + np.arange(steps // 2 * fineness)) * plasma_step
        left_half = -right_half[::-1]  # invert, reverse
    plasma_grid = np.concatenate([left_half, right_half])
    assert(np.array_equal(plasma_grid, -plasma_grid[::-1]))
    return plasma_grid


def plasma_make(steps, cell_size, coarseness=2, fineness=2):
    """
    Make coarse plasma initial state arrays and the arrays needed to intepolate
    coarse plasma into fine plasma (`virt_params`).
    Coarse is the one that will evolve and fine is the one to be bilinearly
    interpolated from the coarse one based on the initial positions
    (using 1 to 4 coarse plasma particles that initially were the closest).
    """
    coarse_step = cell_size * coarseness

    # Make two initial grids of plasma particles, coarse and fine.
    # Coarse is the one that will evolve and fine is the one to be bilinearly
    # interpolated from the coarse one based on the initial positions.

    coarse_grid = make_coarse_plasma_grid(steps, cell_size, coarseness)
    coarse_grid_xs, coarse_grid_ys = coarse_grid[:, None], coarse_grid[None, :]

    fine_grid = make_fine_plasma_grid(steps, cell_size, fineness)

    Nc = len(coarse_grid)

    # Create plasma particles on the coarse grid, the ones that really move
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

    # Neighbour indices array, 1D, same in both x and y direction.
    indices = np.searchsorted(coarse_grid, fine_grid)
    # example:
    #     coarse:  [-2., -1.,  0.,  1.,  2.]
    #     fine:    [-2.4, -1.8, -1.2, -0.6,  0. ,  0.6,  1.2,  1.8,  2.4]
    #     indices: [ 0  ,  1  ,  1  ,  2  ,  2  ,  3  ,  4  ,  4  ,  5 ]
    # There is no coarse particle with index 5, so clip it to 4:
    indices_next = np.clip(indices, 0, Nc - 1)  # [0, 1, 1, 2, 2, 3, 4, 4, 4]
    # Clip to zero for indices of prev particles as well:
    indices_prev = np.clip(indices - 1, 0, Nc - 1)  # [0, 0, 0, 1 ... 3, 3, 4]
    # mixed from: [ 0&0 , 0&1 , 0&1 , 1&2 , 1&2 , 2&3 , 3&4 , 3&4, 4&4 ]

    # Calculate weights for coarse->fine interpolation from initial positions.
    # The further the fine particle is from closest right coarse particles,
    # the more influence the left ones have.
    influence_prev = (coarse_grid[indices_next] - fine_grid) / coarse_step
    influence_next = (fine_grid - coarse_grid[indices_prev]) / coarse_step
    # Fix for boundary cases of missing cornering particles.
    influence_prev[indices_next == 0] = 0   # nothing on the left?
    influence_next[indices_next == 0] = 1   # use right
    influence_next[indices_prev == Nc - 1] = 0  # nothing on the right?
    influence_prev[indices_prev == Nc - 1] = 1  # use left
    # Same arrays are used for interpolating in y-direction.

    # The virtualization formula is thus
    # influence_prev[pi] * influence_prev[pj] * <bottom-left neighbour value> +
    # influence_prev[pi] * influence_next[nj] * <top-left neighbour value> +
    # influence_next[ni] * influence_prev[pj] * <bottom-right neighbour val> +
    # influence_next[ni] * influence_next[nj] * <top-right neighbour value>
    # where pi, pj are indices_prev[i], indices_prev[j],
    #       ni, nj are indices_next[i], indices_next[j] and
    #       i, j are indices of fine virtual particles

    # This is what is employed inside mix() and deposit_kernel().

    # An equivalent formula would be
    # inf_prev[pi] * (inf_prev[pj] * <bot-left> + inf_next[nj] * <bot-right>) +
    # inf_next[ni] * (inf_prev[pj] * <top-left> + inf_next[nj] * <top-right>)

    # Values of m, q, px, py, pz should be scaled by 1/(fineness*coarseness)**2

    virt_params = GPUArrays(
        influence_prev=influence_prev, influence_next=influence_next,
        indices_prev=indices_prev, indices_next=indices_next,
        fine_grid=fine_grid,
    )

    return (coarse_electrons_x_init, coarse_electrons_y_init,
            coarse_electrons_x_offt, coarse_electrons_y_offt,
            coarse_electrons_px, coarse_electrons_py, coarse_electrons_pz,
            coarse_electrons_m, coarse_electrons_q, virt_params)


@numba.jit(inline=True)
def mix(coarse, A, B, C, D, pi, ni, pj, nj):
    """
    Bilinearly interpolate fine plasma properties from four
    historically-neighbouring plasma particle property values.
     B    D  #  y ^         A - bottom-left  neighbour, indices: pi, pj
        .    #    |         B - top-left     neighbour, indices: pi, nj
             #    +---->    C - bottom-right neighbour, indices: ni, pj
     A    C  #         x    D - top-right    neighbour, indices: ni, nj
    See the rest of the deposition and plasma creation for more info.
    """
    return (A * coarse[pi, pj] + B * coarse[pi, nj] +
            C * coarse[ni, pj] + D * coarse[ni, nj])


# Deposition #

# TODO: try to get rid of the CUDA kernel
@numba.cuda.jit
def deposit_kernel(grid_steps, grid_step_size, virtplasma_smallness_factor,
                   c_x_offt, c_y_offt, c_m, c_q, c_px, c_py, c_pz,  # coarse
                   fine_grid,
                   influence_prev, influence_next, indices_prev, indices_next,
                   out_ro, out_jx, out_jy, out_jz):
    """
    Interpolate coarse plasma into fine plasma and deposit it on the
    charge density and current grids.
    """
    # Do nothing if our thread does not have a fine particle to deposit.
    fk = numba.cuda.grid(1)
    if fk >= fine_grid.size**2:
        return
    fi, fj = fk // fine_grid.size, fk % fine_grid.size

    # Calculate the weights of the historically-neighbouring coarse particles
    A = influence_prev[fi] * influence_prev[fj]
    B = influence_prev[fi] * influence_next[fj]
    C = influence_next[fi] * influence_prev[fj]
    D = influence_next[fi] * influence_next[fj]
    # and retrieve their indices.
    pi, ni = indices_prev[fi], indices_next[fi]
    pj, nj = indices_prev[fj], indices_next[fj]

    # Now we're ready to mix the fine particle characteristics
    x_offt = mix(c_x_offt, A, B, C, D, pi, ni, pj, nj)
    y_offt = mix(c_y_offt, A, B, C, D, pi, ni, pj, nj)
    x = fine_grid[fi] + x_offt  # x_fine_init
    y = fine_grid[fj] + y_offt  # y_fine_init

    # TODO: const m and q
    m = virtplasma_smallness_factor * mix(c_m, A, B, C, D, pi, ni, pj, nj)
    q = virtplasma_smallness_factor * mix(c_q, A, B, C, D, pi, ni, pj, nj)

    px = virtplasma_smallness_factor * mix(c_px, A, B, C, D, pi, ni, pj, nj)
    py = virtplasma_smallness_factor * mix(c_py, A, B, C, D, pi, ni, pj, nj)
    pz = virtplasma_smallness_factor * mix(c_pz, A, B, C, D, pi, ni, pj, nj)

    # and deposit the resulting fine particle on ro/j grids.
    gamma_m = sqrt(m**2 + px**2 + py**2 + pz**2)
    dro = q / (1 - pz / gamma_m)
    djx = px * (dro / gamma_m)
    djy = py * (dro / gamma_m)
    djz = pz * (dro / gamma_m)

    i, j, wMP, w0P, wPP, wM0, w00, wP0, wMM, w0M, wPM = weights(
        x, y, grid_steps, grid_step_size
    )
    deposit9(out_ro, i, j, dro, wMP, w0P, wPP, wM0, w00, wP0, wMM, w0M, wPM)
    deposit9(out_jx, i, j, djx, wMP, w0P, wPP, wM0, w00, wP0, wMM, w0M, wPM)
    deposit9(out_jy, i, j, djy, wMP, w0P, wPP, wM0, w00, wP0, wMM, w0M, wPM)
    deposit9(out_jz, i, j, djz, wMP, w0P, wPP, wM0, w00, wP0, wMM, w0M, wPM)


def deposit(config, ro_initial, x_offt, y_offt, m, q, px, py, pz, virt_params):
    """
    Interpolate coarse plasma into fine plasma and deposit it on the
    charge density and current grids.
    This is a convenience wrapper around the `deposit_kernel` CUDA kernel.
    """
    virtplasma_smallness_factor = 1 / (config.plasma_coarseness *
                                       config.plasma_fineness)**2
    ro = cp.zeros((config.grid_steps, config.grid_steps))
    jx = cp.zeros((config.grid_steps, config.grid_steps))
    jy = cp.zeros((config.grid_steps, config.grid_steps))
    jz = cp.zeros((config.grid_steps, config.grid_steps))
    cfg = int(np.ceil(virt_params.fine_grid.size**2 / WARP_SIZE)), WARP_SIZE
    deposit_kernel[cfg](config.grid_steps, config.grid_step_size,
                        virtplasma_smallness_factor,
                        x_offt, y_offt, m, q, px, py, pz,
                        virt_params.fine_grid,
                        virt_params.influence_prev, virt_params.influence_next,
                        virt_params.indices_prev, virt_params.indices_next,
                        ro, jx, jy, jz)
    # Also add the background ion charge density.
    ro += ro_initial  # Do it last to preserve more float precision
    numba.cuda.synchronize()
    return ro, jx, jy, jz


def initial_deposition(config, x_offt, y_offt, px, py, pz, m, q, virt_params):
    """
    Determine the background ion charge density by depositing the electrons
    with their initial parameters and negating the result.
    """
    # Don't allow initial speeds for calculations with background ions
    assert all([np.array_equiv(p, 0) for p in [px, py, pz]])

    ro_electrons_initial, _, _, _ = deposit(config, 0, x_offt, y_offt,
                                            m, q, px, py, pz, virt_params)
    return -ro_electrons_initial  # Right on the GPU, huh


# Field interpolation and particle movement (fused) #

# TODO: try to get rid of the kernel
@numba.cuda.jit
def move_smart_kernel(xi_step_size, reflect_boundary,
                      grid_step_size, grid_steps,
                      ms, qs,
                      x_init, y_init,
                      prev_x_offt, prev_y_offt,
                      estimated_x_offt, estimated_y_offt,
                      prev_px, prev_py, prev_pz,
                      Ex_avg, Ey_avg, Ez_avg, Bx_avg, By_avg, Bz_avg,
                      new_x_offt, new_y_offt, new_px, new_py, new_pz):
    """
    Update plasma particle coordinates and momenta according to the field
    values interpolated halfway between the previous plasma particle location
    and the the best estimation of its next location currently available to us.
    Also reflect the particles from `+-reflect_boundary`.
    """
    # Do nothing if our thread does not have a coarse particle to move.
    k = numba.cuda.grid(1)
    if k >= ms.size:
        return

    m, q = ms[k], qs[k]

    opx, opy, opz = prev_px[k], prev_py[k], prev_pz[k]
    px, py, pz = opx, opy, opz
    x_offt, y_offt = prev_x_offt[k], prev_y_offt[k]

    # Calculate midstep positions and fields in them.
    x_halfstep = x_init[k] + (prev_x_offt[k] + estimated_x_offt[k]) / 2
    y_halfstep = y_init[k] + (prev_y_offt[k] + estimated_y_offt[k]) / 2
    i, j, wMP, w0P, wPP, wM0, w00, wP0, wMM, w0M, wPM = weights(
        x_halfstep, y_halfstep, grid_steps, grid_step_size
    )
    Ex = interp9(Ex_avg, i, j, wMP, w0P, wPP, wM0, w00, wP0, wMM, w0M, wPM)
    Ey = interp9(Ey_avg, i, j, wMP, w0P, wPP, wM0, w00, wP0, wMM, w0M, wPM)
    Ez = interp9(Ez_avg, i, j, wMP, w0P, wPP, wM0, w00, wP0, wMM, w0M, wPM)
    Bx = interp9(Bx_avg, i, j, wMP, w0P, wPP, wM0, w00, wP0, wMM, w0M, wPM)
    By = interp9(By_avg, i, j, wMP, w0P, wPP, wM0, w00, wP0, wMM, w0M, wPM)
    Bz = 0  # Bz = 0 for now

    # Move the particles according the the fields
    gamma_m = sqrt(m**2 + pz**2 + px**2 + py**2)
    vx, vy, vz = px / gamma_m, py / gamma_m, pz / gamma_m
    factor_1 = q * xi_step_size / (1 - pz / gamma_m)
    dpx = factor_1 * (Ex + vy * Bz - vz * By)
    dpy = factor_1 * (Ey - vx * Bz + vz * Bx)
    dpz = factor_1 * (Ez + vx * By - vy * Bx)
    px, py, pz = opx + dpx / 2, opy + dpy / 2, opz + dpz / 2

    # Move the particles according the the fields again using updated momenta
    gamma_m = sqrt(m**2 + pz**2 + px**2 + py**2)
    vx, vy, vz = px / gamma_m, py / gamma_m, pz / gamma_m
    factor_1 = q * xi_step_size / (1 - pz / gamma_m)
    dpx = factor_1 * (Ex + vy * Bz - vz * By)
    dpy = factor_1 * (Ey - vx * Bz + vz * Bx)
    dpz = factor_1 * (Ez + vx * By - vy * Bx)
    px, py, pz = opx + dpx / 2, opy + dpy / 2, opz + dpz / 2

    # Apply the coordinate and momenta increments
    gamma_m = sqrt(m**2 + pz**2 + px**2 + py**2)

    x_offt += px / (gamma_m - pz) * xi_step_size  # no mixing with x_init
    y_offt += py / (gamma_m - pz) * xi_step_size  # no mixing with y_init

    px, py, pz = opx + dpx, opy + dpy, opz + dpz

    # Reflect the particles from `+-reflect_boundary`.
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

    # Save the results into the output arrays  # TODO: get rid of that
    new_x_offt[k], new_y_offt[k] = x_offt, y_offt
    new_px[k], new_py[k], new_pz[k] = px, py, pz


def move_smart(config,
               m, q, x_init, y_init, x_prev_offt, y_prev_offt,
               estimated_x_offt, estimated_y_offt, px_prev, py_prev, pz_prev,
               Ex_avg, Ey_avg, Ez_avg, Bx_avg, By_avg, Bz_avg):
    """
    Update plasma particle coordinates and momenta according to the field
    values interpolated halfway between the previous plasma particle location
    and the the best estimation of its next location currently available to us.
    This is a convenience wrapper around the `move_smart_kernel` CUDA kernel.
    """
    x_offt_new = cp.zeros_like(x_prev_offt)
    y_offt_new = cp.zeros_like(y_prev_offt)
    px_new = cp.zeros_like(px_prev)
    py_new = cp.zeros_like(py_prev)
    pz_new = cp.zeros_like(pz_prev)
    cfg = int(np.ceil(x_init.size / WARP_SIZE)), WARP_SIZE
    move_smart_kernel[cfg](config.xi_step_size, config.reflect_boundary,
                           config.grid_step_size, config.grid_steps,
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


# The scheme of a single step in xi #

def step(config, const, virt_params, prev, beam_ro):
    """
    Calculate the next iteration of plasma evolution and response.
    Returns the new state with the following attributes:
    `x_offt, y_offt, px, py, pz, Ex, Ey, Ez, Bx, By, Bz, ro, jx, jy, jz`.
    Pass the returned value as `prev` for the next iteration.
    """
    beam_ro = cp.asarray(beam_ro)  # copy the array is on GPU if it's not there

    Bz = cp.zeros_like(prev.Bz)  # Bz = 0 for now

    # Estimate the midpoint particle position without knowing the fields yet
    # TODO: use regular pusher and pass zero fields? previous fields?
    x_offt, y_offt = move_estimate_wo_fields(config, const.m,
                                             const.x_init, const.y_init,
                                             prev.x_offt, prev.y_offt,
                                             prev.px, prev.py, prev.pz)

    # Interpolate fields in midpoint and move particles with previous fields.
    x_offt, y_offt, px, py, pz = move_smart(
        config, const.m, const.q, const.x_init, const.y_init,
        prev.x_offt, prev.y_offt, x_offt, y_offt, prev.px, prev.py, prev.pz,
        # no halfstep-averaged fields yet
        prev.Ex, prev.Ey, prev.Ez, prev.Bx, prev.By, Bz_avg=0
    )
    # Recalculate the plasma density and currents.
    ro, jx, jy, jz = deposit(
        config, const.ro_initial, x_offt, y_offt, const.m, const.q, px, py, pz,
        virt_params
    )

    # Calculate the fields.
    ro_in = ro if not config.field_solver_variant_A else (ro + prev.ro) / 2
    jz_in = jz if not config.field_solver_variant_A else (jz + prev.jz) / 2
    Ex, Ey, Bx, By = calculate_Ex_Ey_Bx_By(config,
                                           prev.Ex, prev.Ey, prev.Bx, prev.By,
                                           # no halfstep-averaged fields yet
                                           beam_ro, ro_in, jx, jy, jz_in,
                                           prev.jx, prev.jy)
    if config.field_solver_variant_A:
        Ex, Ey = 2 * Ex - prev.Ex, 2 * Ey - prev.Ey
        Bx, By = 2 * Bx - prev.Bx, 2 * By - prev.By

    Ez = calculate_Ez(config, jx, jy)
    # Bz = 0 for now
    Ex_avg = (Ex + prev.Ex) / 2
    Ey_avg = (Ey + prev.Ey) / 2
    Ez_avg = (Ez + prev.Ez) / 2
    Bx_avg = (Bx + prev.Bx) / 2
    By_avg = (By + prev.By) / 2

    # Repeat the previous procedure using averaged fields.
    x_offt, y_offt, px, py, pz = move_smart(
        config, const.m, const.q, const.x_init, const.y_init,
        prev.x_offt, prev.y_offt, x_offt, y_offt,
        prev.px, prev.py, prev.pz,
        Ex_avg, Ey_avg, Ez_avg, Bx_avg, By_avg, Bz_avg=0
    )
    ro, jx, jy, jz = deposit(config, const.ro_initial, x_offt, y_offt,
                             const.m, const.q, px, py, pz, virt_params)

    ro_in = ro if not config.field_solver_variant_A else (ro + prev.ro) / 2
    jz_in = jz if not config.field_solver_variant_A else (jz + prev.jz) / 2
    Ex, Ey, Bx, By = calculate_Ex_Ey_Bx_By(config,
                                           Ex_avg, Ey_avg, Bx_avg, By_avg,
                                           beam_ro, ro_in, jx, jy, jz_in,
                                           prev.jx, prev.jy)
    if config.field_solver_variant_A:
        Ex, Ey = 2 * Ex - prev.Ex, 2 * Ey - prev.Ey
        Bx, By = 2 * Bx - prev.Bx, 2 * By - prev.By

    Ez = calculate_Ez(config, jx, jy)
    # Bz = 0 for now
    Ex_avg = (Ex + prev.Ex) / 2
    Ey_avg = (Ey + prev.Ey) / 2
    Ez_avg = (Ez + prev.Ez) / 2
    Bx_avg = (Bx + prev.Bx) / 2
    By_avg = (By + prev.By) / 2

    # Repeat the previous procedure using averaged fields once again.
    x_offt, y_offt, px, py, pz = move_smart(
        config, const.m, const.q, const.x_init, const.y_init,
        prev.x_offt, prev.y_offt, x_offt, y_offt,
        prev.px, prev.py, prev.pz,
        Ex_avg, Ey_avg, Ez_avg, Bx_avg, By_avg, Bz_avg=0
    )
    ro, jx, jy, jz = deposit(config, const.ro_initial, x_offt, y_offt,
                             const.m, const.q, px, py, pz, virt_params)

    # TODO: what do we need that roj_new for, jx_prev/jy_prev only?

    # Return the array collection that would serve as `prev` for the next step.
    new_state = GPUArrays(x_offt=x_offt, y_offt=y_offt, px=px, py=py, pz=pz,
                          Ex=Ex.copy(), Ey=Ey.copy(), Ez=Ez.copy(),
                          Bx=Bx.copy(), By=By.copy(), Bz=Bz.copy(),
                          ro=ro, jx=jx, jy=jy, jz=jz)

    return new_state


# Array initialization #

def init(config):
    """
    Initialize all the arrays needed for `step` and `config.beam`.
    """

    assert config.grid_steps % 2 == 1

    # virtual particles should not reach the window pre-boundary cells
    assert config.reflect_padding_steps > config.plasma_coarseness + 1
    # the (costly) alternative is to reflect after plasma virtualization

    config.reflect_boundary = config.grid_step_size * (
        config.grid_steps / 2 - config.reflect_padding_steps
    )

    grid = ((np.arange(config.grid_steps) - config.grid_steps // 2)
            * config.grid_step_size)
    xs, ys = grid[:, None], grid[None, :]

    x_init, y_init, x_offt, y_offt, px, py, pz, m, q, virt_params = \
        plasma_make(config.grid_steps - config.plasma_padding_steps * 2,
                    config.grid_step_size,
                    coarseness=config.plasma_coarseness,
                    fineness=config.plasma_fineness)

    ro_initial = initial_deposition(config, x_offt, y_offt,
                                    px, py, pz, m, q, virt_params)

    const = GPUArrays(m=m, q=q, x_init=x_init, y_init=y_init,
                      ro_initial=ro_initial)

    def zeros():
        return cp.zeros((config.grid_steps, config.grid_steps))

    state = GPUArrays(x_offt=x_offt, y_offt=y_offt, px=px, py=py, pz=pz,
                      Ex=zeros(), Ey=zeros(), Ez=zeros(),
                      Bx=zeros(), By=zeros(), Bz=zeros(),
                      ro=zeros(), jx=zeros(), jy=zeros(), jz=zeros())

    return xs, ys, const, virt_params, state



# Some really sloppy diagnostics #

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


def diagnostics(view_state, config, xi_i, Ez_00_history):
    xi = -xi_i * config.xi_step_size

    Ez_00 = Ez_00_history[-1]
    peak_report = diags_peak_msg(Ez_00_history)

    ro = view_state.ro
    max_zn = diags_ro_zn(config, ro)
    diags_ro_slice(config, xi_i, xi, ro)

    print(f'xi={xi:+.4f} {Ez_00:+.4e}|{peak_report}|zn={max_zn:.3f}')
    sys.stdout.flush()


# Main loop #

def main():
    import config
    xs, ys, const, virt_params, state = init(config)
    Ez_00_history = []

    for xi_i in range(config.xi_steps):
        beam_ro = config.beam(xi_i, xs, ys)

        state = step(config, const, virt_params, state, beam_ro)
        view_state = GPUArraysView(state)

        Ez_00 = view_state.Ez[config.grid_steps // 2, config.grid_steps // 2]
        Ez_00_history.append(Ez_00)

        time_for_diags = xi_i % config.diagnostics_each_N_steps == 0
        last_step = xi_i == config.xi_steps - 1
        if time_for_diags or last_step:
            diagnostics(view_state, config, xi_i, Ez_00_history)


if __name__ == '__main__':
    main()
