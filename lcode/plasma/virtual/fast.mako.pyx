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


"""
A bilinear fine->coarse plasma initializer and interpolator.
Primary author: A. P. Sosedkin <A.P.Sosedkin@inp.nsk.su>
"""


import numpy as np
cimport numpy as np

from ... import plasma_particle
from ... cimport plasma_particle


cpdef void preweighted_interpolator(
        np.ndarray[plasma_particle.t, ndim=2] coarse_plasma,
        np.ndarray[long] indices_prev,  # in both x and y
        np.ndarray[long] indices_next,  # in both x and y
        np.ndarray[double, ndim=2] A_weights,
        np.ndarray[double, ndim=2] B_weights,
        np.ndarray[double, ndim=2] C_weights,
        np.ndarray[double, ndim=2] D_weights,
        np.ndarray[plasma_particle.t, ndim=2] out_fine_plasma):
    cdef long i, j, px, nx, py, ny
    cdef double A_w, B_w, C_w, D_w
    cdef plasma_particle.t A, B, C, D

    for i in range(out_fine_plasma.shape[0]):
        px, nx = indices_prev[i], indices_next[i]
        for j in range(out_fine_plasma.shape[1]):
            py, ny = indices_prev[j], indices_next[j]
            A = coarse_plasma[px, py]
            B = coarse_plasma[nx, py]
            C = coarse_plasma[px, ny]
            D = coarse_plasma[nx, ny]
            A_w = A_weights[i, j]
            B_w = B_weights[i, j]
            C_w = C_weights[i, j]
            D_w = D_weights[i, j]
            <% cs = 'x', 'y', 'p[0]', 'p[1]', 'p[2]', 'v[0]', 'v[1]', 'v[2]' %>
            % for component in cs:
            out_fine_plasma[i, j].${component} = (
                A.${component} * A_w +
                B.${component} * B_w +
                C.${component} * C_w +
                D.${component} * D_w
            )
            % endfor


def make(window_width, steps, coarseness=2, fineness=2):
    cell_size = window_width / steps
    half_width = window_width / 2
    coarse_step, fine_step = cell_size * coarseness, cell_size / fineness

    # Make two initial grids of plasma particles, coarse and fine.
    # Coarse is the one that will evolve and fine is the one to be bilinearly
    # interpolated from the coarse one based on the initial positions.

    coarse_grid_half = np.arange(coarse_step / 2, half_width, coarse_step)
    coarse_grid = np.concatenate([-coarse_grid_half[::-1], coarse_grid_half])
    coarse_grid_xs, coarse_grid_ys = coarse_grid[:, None], coarse_grid[None, :]

    fine_grid = np.arange(-half_width + fine_step / 2, half_width, fine_step)
    fine_grid_xs, fine_grid_ys = fine_grid[:, None], fine_grid[None, :]

    Nc, Nf = len(coarse_grid), len(fine_grid)

    # Create plasma particles on that grids

    coarse_plasma = np.zeros(2 * Nc**2, plasma_particle.dtype)
    coarse_plasma['N'] = np.arange(coarse_plasma.size)
    coarse_plasma['N'] = np.arange(coarse_plasma.size)
    coarse_ions = coarse_plasma[::2].reshape(Nc, Nc)
    coarse_electrons = coarse_plasma[1::2].reshape(Nc, Nc)
    coarse_ions['x'] = coarse_electrons['x'] = coarse_grid_xs
    coarse_ions['y'] = coarse_electrons['y'] = coarse_grid_ys
    coarse_ions['m'] = plasma_particle.USUAL_ION_MASS * coarseness**2
    coarse_ions['q'] = plasma_particle.USUAL_ION_CHARGE * coarseness**2
    coarse_electrons['m'] = plasma_particle.USUAL_ELECTRON_MASS * coarseness**2
    coarse_electrons['q'] = (plasma_particle.USUAL_ELECTRON_CHARGE *
                             coarseness**2)
    # v, p == 0

    fine_plasma = np.zeros(2 * Nf**2, plasma_particle.dtype)
    fine_plasma['N'] = np.arange(fine_plasma.size)       # not really needed
    fine_ions = fine_plasma[::2].reshape(Nf, Nf)
    fine_electrons = fine_plasma[1::2].reshape(Nf, Nf)
    fine_ions['x'] = fine_electrons['x'] = fine_grid_xs  # not really needed
    fine_ions['y'] = fine_electrons['y'] = fine_grid_ys  # not really needed
    fine_ions['m'] = plasma_particle.USUAL_ION_MASS / fineness**2
    fine_ions['q'] = plasma_particle.USUAL_ION_CHARGE / fineness**2
    fine_electrons['m'] = plasma_particle.USUAL_ELECTRON_MASS / fineness**2
    fine_electrons['q'] = (plasma_particle.USUAL_ELECTRON_CHARGE / fineness**2)

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

    # Finally, writing a virtualizer is trivial now

    # A performance trick: a reusable array is stored in the closure.
    # Could've made a copy inside virtualize(...), but this way is faster.
    evolved_fine = fine_plasma.copy()
    evolved_fine_ions = evolved_fine[::2].reshape(Nf, Nf)
    evolved_fine_electrons = evolved_fine[1::2].reshape(Nf, Nf)
    # But it forces us to trust the caller not to mess up the array

    def virtualize(evolved_coarse):
        # This function will get called a lot (several times per xi step)
        evolved_coarse_ions = evolved_coarse[::2].reshape(Nc, Nc)
        evolved_coarse_electrons = evolved_coarse[1::2].reshape(Nc, Nc)

        preweighted_interpolator(evolved_coarse_ions,
                                 indices_prev, indices_next,
                                 A_weights, B_weights, C_weights, D_weights,
                                 out_fine_plasma=evolved_fine_ions)

        preweighted_interpolator(evolved_coarse_electrons,
                                 indices_prev, indices_next,
                                 A_weights, B_weights, C_weights, D_weights,
                                 out_fine_plasma=evolved_fine_electrons)

        return evolved_fine

    return coarse_plasma.ravel(), virtualize
