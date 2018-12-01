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
def roj_init_kernel(ro, jx, jy, jz, ro_initial):
    index = numba.cuda.grid(1)
    stride = numba.cuda.blockDim.x * numba.cuda.gridDim.x
    for k in range(index, ro.size, stride):
        ro[k] = ro_initial[k]
        jx[k] = jy[k] = jz[k] = 0


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


class GPUMonolith:
    cfg = (19, 192)  # empirical guess for a GTX 1070 Ti

    def __init__(self, config):
        P = config.plasma.size
        Q = int(sqrt(P))
        assert Q**2 == P
        N = config.grid_steps
        self.Q = Q

        self._m = numba.cuda.device_array((Q, Q))
        self._q = numba.cuda.device_array((Q, Q))
        self._x = numba.cuda.device_array((Q, Q))
        self._y = numba.cuda.device_array((Q, Q))
        self._px = numba.cuda.device_array((Q, Q))
        self._py = numba.cuda.device_array((Q, Q))
        self._pz = numba.cuda.device_array((Q, Q))

        self._A_weights = numba.cuda.to_device(config.virtualize.A_weights)
        self._B_weights = numba.cuda.to_device(config.virtualize.B_weights)
        self._C_weights = numba.cuda.to_device(config.virtualize.C_weights)
        self._D_weights = numba.cuda.to_device(config.virtualize.D_weights)
        self._indices_prev = numba.cuda.to_device(config.virtualize.indices_prev)
        self._indices_next = numba.cuda.to_device(config.virtualize.indices_next)

        self._ro_initial = numba.cuda.device_array((N, N))
        self._ro = numba.cuda.device_array((N, N))
        self._jx = numba.cuda.device_array((N, N))
        self._jy = numba.cuda.device_array((N, N))
        self._jz = numba.cuda.device_array((N, N))

    def initial_deposition(self, config, plasma_initial):
        self.load(plasma_initial)
        zerofill_kernel[self.cfg](self._ro.ravel())
        deposit_kernel[self.cfg](config.n_dim, config.h,
                                 self._x, self._y, self._m, self._q,
                                 self._px, self._py, self._pz,
                                 self._A_weights, self._B_weights,
                                 self._C_weights, self._D_weights,
                                 self._indices_prev, self._indices_next,
                                 1 / config.virtualize.ratio,
                                 self._ro, self._jx, self._jy, self._jz)
        self._ro_initial[:, :] = -self._ro.copy_to_host()  # ion background


    def load(self, plasma):
        Q = self.Q
        self._m[:, :] = np.ascontiguousarray(plasma['m'].reshape(Q, Q))
        self._q[:, :] = np.ascontiguousarray(plasma['q'].reshape(Q, Q))
        self._x[:, :] = np.ascontiguousarray(plasma['x'].reshape(Q, Q))
        self._y[:, :] = np.ascontiguousarray(plasma['y'].reshape(Q, Q))
        self._px[:, :] = np.ascontiguousarray(plasma['p'][:, 1].reshape(Q, Q))
        self._py[:, :] = np.ascontiguousarray(plasma['p'][:, 2].reshape(Q, Q))
        self._pz[:, :] = np.ascontiguousarray(plasma['p'][:, 0].reshape(Q, Q))

        roj_init_kernel[self.cfg](self._ro.ravel(), self._jx.ravel(),
                                  self._jy.ravel(), self._jz.ravel(),
                                  self._ro_initial.ravel())

    def step(self, config, plasma):
        self.load(plasma)
        deposit_kernel[self.cfg](config.n_dim, config.h,
                                 self._x, self._y, self._m, self._q,
                                 self._px, self._py, self._pz,
                                 self._A_weights, self._B_weights,
                                 self._C_weights, self._D_weights,
                                 self._indices_prev, self._indices_next,
                                 1 / config.virtualize.ratio,
                                 self._ro, self._jx, self._jy, self._jz)
        return self.unload(config)

    def unload(self, config):
        roj = np.zeros((config.n_dim, config.n_dim), dtype=RoJ_dtype)
        roj['ro'] = self._ro.copy_to_host()
        roj['jx'] = self._jx.copy_to_host()
        roj['jy'] = self._jy.copy_to_host()
        roj['jz'] = self._jz.copy_to_host()
        return roj
