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

from math import sqrt, floor

import numpy as np

import numba
import numba.cuda

RoJ_dtype = np.dtype([
    ('ro', np.double),
    ('jz', np.double),
    ('jx', np.double),
    ('jy', np.double),
], align=False)


@numba.cuda.jit
def zerofill_kernel(arr1d):
    index = numba.cuda.grid(1)
    stride = numba.cuda.blockDim.x * numba.cuda.gridDim.x
    for k in range(index, arr1d.size, stride):
        arr1d[k] = 0


@numba.cuda.jit
def deposit_kernel(n_dim, h,
                   c_x, c_y, c_m, c_q, c_p_x, c_p_y, c_p_z,  # coarse
                   A_weights, B_weights, C_weights, D_weights,
                   indices_prev, indices_next, smallness_factor,
                   out_ro, out_jx, out_jy, out_jz):
    index = numba.cuda.grid(1)
    stride = numba.cuda.blockDim.x * numba.cuda.gridDim.x
    for pk in range(index, A_weights.size, stride):
        pi, pj = pk // A_weights.shape[0], pk % A_weights.shape[0]
    #for pi in range(index, A_weights.shape[0], stride):
    #    for pj in range(A_weights.shape[1]):
        px, nx = indices_prev[pi], indices_next[pi]
        py, ny = indices_prev[pj], indices_next[pj]

        A = A_weights[pi, pj]
        B = B_weights[pi, pj]
        C = C_weights[pi, pj]
        D = D_weights[pi, pj]

        x = A * c_x[px, py] + B * c_x[nx, py] + C * c_x[px, ny] + D * c_x[nx, ny]
        y = A * c_y[px, py] + B * c_y[nx, py] + C * c_y[px, ny] + D * c_y[nx, ny]
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

        m_sq = m**2
        p_x_sq = p_x**2
        p_y_sq = p_y**2
        p_z_sq = p_z**2
        gamma_m = sqrt(m_sq + p_x_sq + p_y_sq + p_z_sq)
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

gpu_A_weights = gpu_B_weights = gpu_C_weights = gpu_D_weights = None
gpu_indices_prev = gpu_indices_next = None
gpu_ro = gpu_jx = gpu_jy = gpu_jz = None

def deposit(config, plasma, ion_initial_ro):
    global gpu_A_weights, gpu_B_weights, gpu_C_weights, gpu_D_weights
    global gpu_indices_prev, gpu_indices_next
    global gpu_ro, gpu_jx, gpu_jy, gpu_jz
    #plasma_virtualized = config.virtualize(plasma)

    roj = np.zeros((config.n_dim, config.n_dim), dtype=RoJ_dtype)

    T = plasma.shape[0]
    N = int(sqrt(T))
    assert N**2 == T
    xs_ = np.ascontiguousarray(plasma['x'].reshape(N, N))
    ys_ = np.ascontiguousarray(plasma['y'].reshape(N, N))
    p_x_ = np.ascontiguousarray(plasma['p'][:, 1].reshape(N, N))
    p_y_ = np.ascontiguousarray(plasma['p'][:, 2].reshape(N, N))
    p_z_ = np.ascontiguousarray(plasma['p'][:, 0].reshape(N, N))
    m_ = np.ascontiguousarray(plasma['m'].reshape(N, N))
    q_ = np.ascontiguousarray(plasma['q'].reshape(N, N))
    gpu_m = numba.cuda.to_device(m_)
    gpu_q = numba.cuda.to_device(q_)
    gpu_x = numba.cuda.to_device(xs_)
    gpu_y = numba.cuda.to_device(ys_)
    gpu_px = numba.cuda.to_device(p_x_)
    gpu_py = numba.cuda.to_device(p_y_)
    gpu_pz = numba.cuda.to_device(p_z_)
    # TODO: const once
    if gpu_A_weights is None:
        gpu_A_weights = numba.cuda.to_device(config.virtualize.A_weights)
        gpu_B_weights = numba.cuda.to_device(config.virtualize.B_weights)
        gpu_C_weights = numba.cuda.to_device(config.virtualize.C_weights)
        gpu_D_weights = numba.cuda.to_device(config.virtualize.D_weights)
        gpu_indices_prev = numba.cuda.to_device(config.virtualize.indices_prev)
        gpu_indices_next = numba.cuda.to_device(config.virtualize.indices_next)
        gpu_ro = numba.cuda.to_device(np.ascontiguousarray(roj['ro']))
        gpu_jx = numba.cuda.to_device(np.ascontiguousarray(roj['jx']))
        gpu_jy = numba.cuda.to_device(np.ascontiguousarray(roj['jy']))
        gpu_jz = numba.cuda.to_device(np.ascontiguousarray(roj['jz']))
    gpu_ro[:, :] = ion_initial_ro  # background_ions
    zerofill_kernel[19, 192](gpu_jx.ravel())
    zerofill_kernel[19, 192](gpu_jy.ravel())
    zerofill_kernel[19, 192](gpu_jz.ravel())
    #numba.cuda.synchronize()


    # 3:04 at both 19x128 and 19x256
    # 3:01 at 19x192
    # 3:00 at 38x96
    # 3:01 at 76x48
    # 3:04 at 38x64
    # 2:59 at 38x128
    # 2:58 at 38x192
    # 2:58 at 38x256
    # 2:58 at 76x128
    deposit_kernel[19, 192](config.n_dim, config.h,
                           gpu_x, gpu_y, gpu_m, gpu_q, gpu_px, gpu_py, gpu_pz,
                           gpu_A_weights, gpu_B_weights, gpu_C_weights, gpu_D_weights,
                           gpu_indices_prev, gpu_indices_next,
                           1 / config.virtualize.ratio,
                           gpu_ro, gpu_jx, gpu_jy, gpu_jz)
    #numba.cuda.synchronize()

    roj['ro'] = gpu_ro.copy_to_host()
    roj['jx'] = gpu_jx.copy_to_host()
    roj['jy'] = gpu_jy.copy_to_host()
    roj['jz'] = gpu_jz.copy_to_host()
    #numba.cuda.synchronize()
    return roj
