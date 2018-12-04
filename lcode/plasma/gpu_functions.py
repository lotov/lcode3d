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

import pyculib.fft

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

@numba.cuda.jit
def calculate_RHS_kernel(Ex_sub, beam_ro, ro, jx, jx_prev,
                         grid_step_size, xi_step_size, subtraction_trick,
                         Ex_rhs, Ex_dct1_in):
    N = Ex_rhs.shape[0]
    index = numba.cuda.grid(1)
    stride = numba.cuda.blockDim.x * numba.cuda.gridDim.x
    for k in range(index, Ex_rhs.size, stride):
        i, j = k // N, k % N

        dro_dx = (((+ro[i + 1, j] + beam_ro[i + 1, j]
                    -ro[i - 1, j] - beam_ro[i - 1, j])
                  ) / (2 * grid_step_size)  # - ?
                  if 0 < i < N - 1 else 0)
        djx_dxi = (jx_prev[i, j] - jx[i, j]) / xi_step_size               # - ?

        # TODO: get rid of that array and go straight for dct_in?
        Ex_rhs[i, j] = -((dro_dx - djx_dxi) - Ex_sub[i, j] * subtraction_trick)
        Ex_dct1_in[j, i] = Ex_rhs[i, j]
        # symmetrically pad dct1_in to apply DCT-via-FFT later
        ii = max(i, 1)  # avoid writing to dct_in[:, 2 * N - 2], w/o branching
        Ex_dct1_in[j, 2 * N - 2 - ii] = Ex_dct1_in[j, ii]

        # applying non-zero boundary conditions to the RHS would be:
        # for i in range(self.N):
            # rhs_fixed[i, 0] += top[i] * (2 / self.grid_step_size)
            # rhs_fixed[i, self.N - 1] += bot[i] * (2 / self.grid_step_size)
            ## rhs_fixed[0, i] = rhs_fixed[self.N - 1, i] = 0
            ### changes nothing, as there's a particle-free padding zone?


@numba.cuda.jit
def mid_dct_transform_Ex(Ex_dct1_out, Ex_dct2_in, alf, bet, mul):
    N = Ex_dct1_out.shape[0]
    index = numba.cuda.grid(1)
    stride = numba.cuda.blockDim.x * numba.cuda.gridDim.x

    # Solve tridiagonal matrix equation for each spectral column with Thomas method:
    # A @ tmp_2[k, :] = tmp_1[k, :]
    # A has -1 on superdiagonal, -1 on subdiagonal and aa[k] at the main diagonal
    # The edge elements of each column are forced to 0!
    for i in range(index, N, stride):
        bet[i, 0] = 0
        for j in range(1, N - 1):
            # Note the transposition for dct1_out!
            bet[i, j + 1] = (mul * Ex_dct1_out[j, i].real + bet[i, j]) * alf[i, j + 1]
        # Note the transposition for dct2_in!
        Ex_dct2_in[N - 1, i] = 0  # Note the forced zero
        for j in range(N - 2, 0 - 1, -1):
            Ex_dct2_in[j, i] = alf[i, j + 1] * Ex_dct2_in[j + 1, i] + bet[i, j + 1]
            # also symmetrical-fill the array in preparation for a second DCT
            ii = max(i, 1)  # avoid writing to dct_in[:, 2 * N - 2], w/o branching
            Ex_dct2_in[j, 2 * N - 2 - ii] = Ex_dct2_in[j, ii]
        # dct2_in[:, 0] == 0  # happens by itself


@numba.cuda.jit
def unpack_resulting_Ex(Ex_dct2_out, Ex):
    N = Ex.shape[0]
    index = numba.cuda.grid(1)
    stride = numba.cuda.blockDim.x * numba.cuda.gridDim.x
    for k in range(index, Ex.size, stride):
        i, j = k // N, k % N
        Ex[i, j] = Ex_dct2_out[j, i].real


class GPUMonolith:
    cfg = (19, 192)  # empirical guess for a GTX 1070 Ti

    def __init__(self, config):
        self._Nc = Nc = int(sqrt(config.plasma.size))
        assert Nc**2 == config.plasma.size

        self.grid_steps = N = config.grid_steps
        self.xi_step_size = config.xi_step_size
        self.grid_step_size = config.window_width / config.grid_steps
        self.subtraction_trick = config.field_solver_subtraction_trick

        self.virtplasma_smallness_factor = 1 / config.virtualize.ratio

        self._m = numba.cuda.device_array((Nc, Nc))
        self._q = numba.cuda.device_array((Nc, Nc))
        self._x = numba.cuda.device_array((Nc, Nc))
        self._y = numba.cuda.device_array((Nc, Nc))
        self._px = numba.cuda.device_array((Nc, Nc))
        self._py = numba.cuda.device_array((Nc, Nc))
        self._pz = numba.cuda.device_array((Nc, Nc))

        self._A_weights = numba.cuda.to_device(config.virtualize.A_weights)
        self._B_weights = numba.cuda.to_device(config.virtualize.B_weights)
        self._C_weights = numba.cuda.to_device(config.virtualize.C_weights)
        self._D_weights = numba.cuda.to_device(config.virtualize.D_weights)
        self._indices_prev = numba.cuda.to_device(config.virtualize.indices_prev)
        self._indices_next = numba.cuda.to_device(config.virtualize.indices_next)

        # Constant array for mixed boundary conditions solver
        # * diagonal matrix elements (used in the next one)
        aa = 2 + 4 * np.sin(np.arange(0, N) * np.pi / (2 * (N - 1)))**2
        if self.subtraction_trick:
            aa += self.grid_step_size**2 * self.subtraction_trick
        alf = np.zeros((N, N + 1))
        # * precalculated internal coefficients for tridiagonal solving
        for i in range(1, N):
            alf[:, i + 1] = 1 / (aa - alf[:, i])
        self._mix_alf = numba.cuda.to_device(alf)
        # * scratchpad array for mixed boundary conditions solver
        self._mix_bet = numba.cuda.device_array((N, N))

        self.dct_plan = pyculib.fft.FFTPlan(shape=(2 * N - 2,),
                                            itype=np.float64,
                                            otype=np.complex128,
                                            batch=N)
        # (2 * N - 2) // 2 + 1 == (N - 1) + 1 == N
        self._Ex_dct1_in = numba.cuda.device_array((N, 2 * N - 2))
        self._Ex_dct1_out = numba.cuda.device_array((N, N), dtype=np.complex128)
        self._Ex_dct2_in = numba.cuda.device_array((N, 2 * N - 2))
        self._Ex_dct2_out = numba.cuda.device_array((N, N), dtype=np.complex128)
        self._Ex = numba.cuda.device_array((N, N))

        # total multiplier to compensate for the iDCT+DCT transforms
        self.mix_mul = self.grid_step_size**2
        self.mix_mul /= 2 * N - 2  # don't ask

        self._ro_initial = numba.cuda.device_array((N, N))
        self._ro = numba.cuda.device_array((N, N))
        self._jx = numba.cuda.device_array((N, N))
        self._jy = numba.cuda.device_array((N, N))
        self._jz = numba.cuda.device_array((N, N))

        self._beam_ro = numba.cuda.device_array((N, N))
        self._Ex_prev = numba.cuda.device_array((N, N))
        self._jx_prev = numba.cuda.device_array((N, N))

        self._Ex_rhs = numba.cuda.device_array((N, N))

    def load(self, plasma, beam_ro, Ex_prev, jx_prev):
        Nc = self._Nc
        self._m[:, :] = np.ascontiguousarray(plasma['m'].reshape(Nc, Nc))
        self._q[:, :] = np.ascontiguousarray(plasma['q'].reshape(Nc, Nc))
        self._x[:, :] = np.ascontiguousarray(plasma['x'].reshape(Nc, Nc))
        self._y[:, :] = np.ascontiguousarray(plasma['y'].reshape(Nc, Nc))
        self._px[:, :] = np.ascontiguousarray(plasma['p'][:, 1].reshape(Nc, Nc))
        self._py[:, :] = np.ascontiguousarray(plasma['p'][:, 2].reshape(Nc, Nc))
        self._pz[:, :] = np.ascontiguousarray(plasma['p'][:, 0].reshape(Nc, Nc))

        roj_init_kernel[self.cfg](self._ro.ravel(), self._jx.ravel(),
                                  self._jy.ravel(), self._jz.ravel(),
                                  self._ro_initial.ravel())
        numba.cuda.synchronize()

        self._beam_ro[:, :] = np.ascontiguousarray(beam_ro)
        self._Ex_prev[:, :] = np.ascontiguousarray(Ex_prev)
        self._jx_prev[:, :] = np.ascontiguousarray(jx_prev)

    def deposit(self):
        deposit_kernel[self.cfg](self.grid_steps, self.grid_step_size,
                                 self._x, self._y, self._m, self._q,
                                 self._px, self._py, self._pz,
                                 self._A_weights, self._B_weights,
                                 self._C_weights, self._D_weights,
                                 self._indices_prev, self._indices_next,
                                 self.virtplasma_smallness_factor,
                                 self._ro, self._jx, self._jy, self._jz)
        numba.cuda.synchronize()

    def initial_deposition(self, config, plasma_initial):
        self.load(plasma_initial, 0, 0, 0)
        zerofill_kernel[self.cfg](self._ro.ravel())
        numba.cuda.synchronize()
        self.deposit()
        self._ro_initial[:, :] = -np.array(self._ro.copy_to_host())

    def calculate_RHS(self):
        calculate_RHS_kernel[self.cfg](self._Ex_prev,
                                       self._beam_ro,
                                       self._ro, self._jx,
                                       self._jx_prev,
                                       self.grid_step_size, self.xi_step_size,
                                       self.subtraction_trick,
                                       self._Ex_rhs, self._Ex_dct1_in)
        numba.cuda.synchronize()

    def calculate_Ex(self):
        # The grand plan: mul * iDCT(SPECTRAL_MAGIC(DCT(in.T).T)).T).T
        # where iDCT is DCT;
        # and DCT is jury-rigged from symmetrically-padded DFT
        self.calculate_Ex_1()
        self.calculate_Ex_2()
        self.calculate_Ex_3()
        self.calculate_Ex_4()

    def calculate_Ex_1(self):
        # 1. Apply iDCT-1 (Discrete Cosine Transform Type 1) to the RHS
        # iDCT-1 is just DCT-1 in cuFFT
        self.dct_plan.forward(self._Ex_dct1_in.ravel(), self._Ex_dct1_out.ravel())
        numba.cuda.synchronize()
        # This implementation of DCT is real-to-complex, so scrapping the i, j
        # element of the transposed answer would be dct1_out[j, i].real

    def calculate_Ex_2(self):
        # 2. Solve tridiagonal matrix equation for each spectral column with Thomas method:
        mid_dct_transform_Ex[self.cfg](self._Ex_dct1_out, self._Ex_dct2_in,
                                       self._mix_alf, self._mix_bet,
                                       self.mix_mul)
        numba.cuda.synchronize()

    def calculate_Ex_3(self):
        # 3. Apply DCT-1 (Discrete Cosine Transform Type 1) to the transformed spectra
        self.dct_plan.forward(self._Ex_dct2_in.ravel(), self._Ex_dct2_out.ravel())
        numba.cuda.synchronize()

    def calculate_Ex_4(self):
        # 4. Transpose the resulting Ex (TODO: fuse this step into later steps?)
        unpack_resulting_Ex[self.cfg](self._Ex_dct2_out, self._Ex)
        numba.cuda.synchronize()


    def step(self, config, plasma, beam_ro, Ex_prev, jx_prev):
        self.load(plasma, beam_ro, Ex_prev, jx_prev)

        self.deposit()
        self.calculate_RHS()

        self.calculate_Ex()

        return self.unload(config)

    def unload(self, config):
        roj = np.zeros((config.n_dim, config.n_dim), dtype=RoJ_dtype)
        roj['ro'] = self._ro.copy_to_host()
        roj['jx'] = self._jx.copy_to_host()
        roj['jy'] = self._jy.copy_to_host()
        roj['jz'] = self._jz.copy_to_host()

        Ex_rhs = self._Ex_rhs.copy_to_host()

        Ex = self._Ex.copy_to_host()

        numba.cuda.synchronize()

        return roj, Ex_rhs, Ex
