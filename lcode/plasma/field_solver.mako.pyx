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
3D field solver for LCODE.
Primary author: I. A. Shalimova <ias@osmf.sscc.ru>
Secondary author: A. P. Sosedkin <A.P.Sosedkin@inp.nsk.su>
'''


from libc.math cimport sin, cos, exp
from libc.math cimport M_PI as pi  # 3.141592653589793 on my machine

import numpy as np
cimport numpy as np

import cython
cimport cython
from cython.parallel import prange, parallel
cimport openmp

import scipy.fftpack

#from trig_mkl import TrigTransform
#from trig_mkl cimport TrigTransform

RoJ_dtype = np.dtype([
    ('ro', np.double),
    ('jz', np.double),
    ('jx', np.double),
    ('jy', np.double),
], align=False)


cdef class ThreadLocalStorage:
    def __init__(self, n_dim, h):
        self.alf = np.zeros(n_dim)
        self.bet = np.zeros(n_dim + 1)
        self.RedFi = np.zeros(n_dim)
        self.Svl = np.zeros(n_dim)
        self.PrF = np.zeros(n_dim)
        self.PrV = np.zeros(n_dim)
        self.Psi = np.zeros(n_dim)
        self.p = np.zeros((n_dim, n_dim))
        self.v = np.zeros((n_dim, n_dim))
        self.v1 = np.zeros((n_dim, n_dim))
        self.p_km1 = np.zeros((n_dim, n_dim))
        self.rhs = np.zeros((n_dim, n_dim))
        self.grad1 = np.zeros((n_dim, n_dim))
        self.grad2 = np.zeros((n_dim, n_dim))

        self.Ez_a = 2 + 4 * np.sin(np.arange(1, n_dim) * np.pi / (2 * n_dim))**2  #  +  h**2  # only used with xi derivatives
        self.Ez_bet = np.zeros((n_dim - 1, n_dim))
        self.Ez_alf = np.zeros((n_dim - 1, n_dim))
        self.Ez_alf[:, 0] = 0
        for k in range(n_dim - 1):
            for i in range(n_dim - 1):
                self.Ez_alf[k, i + 1] = 1 / (self.Ez_a[k] - self.Ez_alf[k, i])
        self.Ez_PrFi = np.zeros((n_dim - 1, n_dim - 1))
        self.Ez_P = np.zeros((n_dim - 1, n_dim - 1))


@cython.boundscheck(False)
cpdef void pader_x(double[:, :] in_, double[:, :] out, double h, int n_dim,
                   int num_threads) nogil:
    cdef int i, j
    cdef double h2 = h * 2
    for j in prange(n_dim, num_threads=num_threads, nogil=True):
        for i in range(1, n_dim - 1):
            out[i, j] = (in_[i + 1, j] - in_[i - 1, j]) / h2
        out[0, j] = out[n_dim - 1, j] = 0
        # WRONG
        ##out[0, j] = 2 * (in_[1, j] - in_[0, j]) / h
        ##out[-1, j] = 2 * (in_[-1, j] - 2 * in_[-2, j]) / h


@cython.boundscheck(False)
cpdef void pader_y(double[:, :] in_, double[:, :] out, double h, int n_dim,
                   int num_threads) nogil:
    cdef int i, j
    cdef double h2 = h * 2
    for i in prange(n_dim, num_threads=num_threads, nogil=True):
        for j in range(1, n_dim - 1):
            out[i, j] = (in_[i, j + 1] - in_[i, j - 1]) / h2
        out[i, 0] = out[i, n_dim - 1] = 0
        # WRONG
        ##out[i, 0] = 2 * (in_[i, 1] - in_[i, 0]) / h
        ##out[i, -1] = 2 * (in_[i, 1] - in_[i, -2]) / h


@cython.boundscheck(False)
cpdef void pader_xi(double[:, :] in_prev, double[:, :] in_cur,
                    double[:, :] out, double h3, int n_dim,
                    int num_threads) nogil:
    cdef int i, j
    for i in prange(n_dim, num_threads=num_threads, nogil=True):
        for j in range(n_dim):
            out[i, j] = (in_prev[i, j] - in_cur[i, j]) / h3


cdef class MixedSolver:
    def __init__(MixedSolver self, int N, double h,
                 double subtraction_trick=1, int num_threads=1):
        self.h, self.N = h, N
        self.subtraction_trick = subtraction_trick
        #self.mul = h**2  / (2 * (N - 1))  # total multiplier to compensate for the iDCT+DCT transforms
        #self.mul *= 2 * (N - 1)  # for use with unnormed Intel MKL, not scipy
        self.mul = h**2  # total multiplier to compensate for the iDCT+DCT transforms (MKL!)

        aa = 2 + 4 * np.sin(np.arange(0, N) * np.pi / (2 * (N - 1)))**2  # diagonal matrix elements
        if subtraction_trick:
            aa += self.h**2 * subtraction_trick
        alf = np.zeros((N, N + 1))  # precalculated internal coefficients for tridiagonal solving
        for i in range(1, N):
            alf[:, i + 1] = 1 / (aa - alf[:, i])
        self.alf = alf

        self.bet = np.zeros((N, N))
        #self.rhs_fixed = np.zeros((N, N))
        #self.tmp1 = np.zeros((N, N))
        #self.tmp2 = np.zeros((N, N))
        self.num_threads = num_threads
        self.tt1 = TrigTransform(self.N, tt_type='dct', num_threads=self.num_threads)
        self.tt2 = TrigTransform(self.N, tt_type='dct', num_threads=self.num_threads)

    cpdef solve(MixedSolver self, double[:, :] rhs, double[:] bound_top, double[:] bound_bot, double[:, :] out):
        # Solve Laplace x = (-)? RHS for x with mixed boundary conditions using DCT-1
        cdef int i, j
        assert rhs.shape[0] == rhs.shape[1] == self.N

        # 1. Apply boundary conditions to the rhs
        cdef double[:, :] rhs_fixed = self.tt1.array  # a view, not a copy!
        rhs_fixed[...] = rhs.T
        for i in range(self.N):
            rhs_fixed[i, 0] += bound_top[i] * (2 / self.h)
            rhs_fixed[i, self.N - 1] += bound_bot[i] * (2 / self.h)
            # rhs_fixed[0, i] = rhs_fixed[self.N - 1, i] = 0  # changes nothing???

        # 2. Apply iDCT-1 (inverse Discrete Cosine Transform Type 1) to the RHS
        # TODO: accelerated version using fftw with extra codelets via ctypes?
        #cdef double[:, :] tmp1 = scipy.fftpack.idct(x=self.rhs_fixed, type=1, overwrite_x=True).T
        #cdef double[:, :] tmp1 = self.forward.transform(x=self.rhs_fixed).T
        #cdef double[:, :] tmp1 = self.tt.idct_2d(self.rhs_fixed).T
        #self.tmp1[...] = tmp1
        self.tt1.idct_2d()
        cdef double[:, :] tmp1 = self.tt1.array.T
        cdef double[:, :] tmp2 = self.tt2.array.T

        # 3. Solve tridiagonal matrix equation for each spectral column with Thomas method:
        # A @ tmp_2[k, :] = tmp_1[k, :]
        # A has -1 on superdiagonal, -1 on subdiagonal and aa[k] at the main diagonal
        # The edge elements of each column are forced to 0!
        for i in prange(self.N, num_threads=self.num_threads, nogil=True):
            self.bet[i, 0] = 0
            for j in range(1, self.N - 1):
                self.bet[i, j + 1] = (self.mul * tmp1[i, j] + self.bet[i, j]) * self.alf[i, j + 1]
            tmp2[i, self.N - 1] = 0  # note the forced zero
            for j in range(self.N - 2, 0 - 1, -1):
                tmp2[i, j] = self.alf[i, j + 1] * tmp2[i, j + 1] + self.bet[i, j + 1]
            # tmp2[:, 0] == 0, it happens by itself

        # EXTRA: suppression
        #for i in range(self.N):
        #    for j in range(self.N):
        #        #self.tmp2[i, j] *= exp(-((i+1)/self.N + .1)**10)
        #        self.tmp2[i, j] *= 1 - (i/self.N)**2

        # 4. Apply DCT-1 (Discrete Cosine Transform Type 1) to the transformed spectra
        #cdef double[:, :] tmp_out = scipy.fftpack.dct(self.tmp2.T, type=1, overwrite_x=True)
        #cdef double[:, :] tmp_out = self.tt.dct_2d(self.tmp2.T)
        self.tt2.dct_2d()  # .T of argument implied, see above
        out[...] = self.tt2.array.T


cpdef void calculate_Ex(double[:, :] in_Ex, double[:, :] out_Ex,
                        double[:, :] ro,
                        double[:, :] jx, double[:, :] jx_prev,
                        ThreadLocalStorage tls,
                        MixedSolver mxs,
                        unsigned int n_dim, double h, double h3,
                        unsigned int npq,
                        double[:] zz,
                        bint variant_A,
                        int num_threads,
                        ):
    cdef int i, j
    cdef double[:, :] dro_dx = tls.grad1
    cdef double[:, :] djx_dxi = tls.grad2
    pader_x(ro, dro_dx, h, n_dim, num_threads)
    pader_xi(jx_prev, jx, djx_dxi, h3, n_dim, num_threads)
    for i in prange(n_dim, num_threads=num_threads, nogil=True):
        for j in range(n_dim):
            out_Ex[i, j] = in_Ex[i, j]  # start from an approximation
            tls.rhs[i, j] = -(dro_dx[i, j] - djx_dxi[i, j])
            if mxs.subtraction_trick:
                tls.rhs[i, j] += in_Ex[i, j] * mxs.subtraction_trick
            #tls.rhs[i, j] = - (dro_dx[i, j] - djx_dxi[i, j])
    #Posson_reduct_12(zz, zz, tls.rhs, out_Ex, tls, n_dim, h, npq)
    mxs.solve(tls.rhs, zz, zz, out_Ex)

    if variant_A:
        for i in range(n_dim):
            for j in range(n_dim):
                out_Ex[i, j] = 2 * out_Ex[i, j] - in_Ex[i, j]


cpdef void calculate_Ey(double[:, :] in_Ey, double[:, :] out_Ey_T,
                        double[:, :] ro,
                        double[:, :] jy, double[:, :] jy_prev,
                        ThreadLocalStorage tls,
                        MixedSolver mxs,
                        unsigned int n_dim, double h, double h3,
                        unsigned int npq,
                        double[:] zz,
                        bint variant_A,
                        int num_threads,
                        ):
    cdef int i, j
    cdef double[:, :] dro_dy = tls.grad1
    cdef double[:, :] djy_dxi = tls.grad2
    pader_y(ro, dro_dy, h, n_dim, num_threads)
    pader_xi(jy_prev, jy, djy_dxi, h3, n_dim, num_threads)
    for i in prange(n_dim, num_threads=num_threads, nogil=True):
        for j in range(n_dim):
            out_Ey_T[j, i] = in_Ey[i, j]  # start from an approximation
            tls.rhs[j, i] = -(dro_dy[i, j] - djy_dxi[i, j])
            if mxs.subtraction_trick:
                tls.rhs[j, i] += in_Ey[i, j] * mxs.subtraction_trick
            #tls.rhs[j, i] = -(dro_dy[i, j] - djy_dxi[i, j])
    #Posson_reduct_12(zz, zz, tls.rhs, out_Ey_T, tls, n_dim, h, npq)
    mxs.solve(tls.rhs, zz, zz, out_Ey_T)

    if variant_A:
        for i in range(n_dim):
            for j in range(n_dim):
                out_Ey_T[j, i] = 2 * out_Ey_T[j, i] - in_Ey[i, j]


cpdef void calculate_Bx(double[:, :] in_Bx, double[:, :] out_Bx_T,
                        double[:, :] jz,
                        double[:, :] jy, double[:, :] jy_prev,
                        ThreadLocalStorage tls,
                        MixedSolver mxs,
                        unsigned int n_dim, double h, double h3,
                        unsigned int npq,
                        double[:] zz,
                        bint variant_A,
                        int num_threads,
                        ):
    cdef int i, j
    cdef double[:, :] djz_dy = tls.grad1
    cdef double[:, :] djy_dxi = tls.grad2
    pader_y(jz, djz_dy, h, n_dim, num_threads)
    pader_xi(jy_prev, jy, djy_dxi, h3, n_dim, num_threads)
    for i in prange(n_dim, num_threads=num_threads, nogil=True):
        for j in range(n_dim):
            out_Bx_T[j, i] = in_Bx[i, j]  # start from an approximation
            tls.rhs[j, i] = +(djz_dy[i, j] - djy_dxi[i, j])
            if mxs.subtraction_trick:
                tls.rhs[j, i] += in_Bx[i, j] * mxs.subtraction_trick
            #tls.rhs[j, i] = +(djz_dy[i, j] - djy_dxi[i, j])
    #Posson_reduct_12(zz, zz, tls.rhs, out_Bx_T, tls, n_dim, h, npq)
    mxs.solve(tls.rhs, zz, zz, out_Bx_T)

    if variant_A:
        for i in range(n_dim):
            for j in range(n_dim):
                out_Bx_T[j, i] = 2 * out_Bx_T[j, i] - in_Bx[i, j]


cpdef void calculate_By(double[:, :] in_By, double[:, :] out_By,
                        double[:, :] jz,
                        double[:, :] jx, double[:, :] jx_prev,
                        ThreadLocalStorage tls,
                        MixedSolver mxs,
                        unsigned int n_dim, double h, double h3,
                        unsigned int npq,
                        double[:] zz,
                        bint variant_A,
                        int num_threads,
                        ):
    cdef int i, j
    cdef double[:, :] djz_dx = tls.grad1
    cdef double[:, :] djx_dxi = tls.grad2
    pader_x(jz, djz_dx, h, n_dim, num_threads)
    pader_xi(jx_prev, jx, djx_dxi, h3, n_dim, num_threads)
    for i in prange(n_dim, num_threads=num_threads, nogil=True):
        for j in range(n_dim):
            out_By[i, j] = in_By[i, j]  # start from an approximation
            tls.rhs[i, j] = -(djz_dx[i, j] - djx_dxi[i, j])
            if mxs.subtraction_trick:
                tls.rhs[i, j] += in_By[i, j] * mxs.subtraction_trick
            #tls.rhs[i, j] = -(djz_dx[i, j] - djx_dxi[i, j])
    #Posson_reduct_12(zz, zz, tls.rhs, out_By, tls, n_dim, h, npq)
    mxs.solve(tls.rhs, zz, zz, out_By)

    if variant_A:
        for i in range(n_dim):
            for j in range(n_dim):
                out_By[i, j] = 2 * out_By[i, j] - in_By[i, j]


cpdef void calculate_Bz(double[:, :] in_Bz, double[:, :] out_Bz,
                        double[:, :] jx, double[:, :] jy,
                        ThreadLocalStorage tls,
                        unsigned int n_dim, double h, unsigned int npq,
                        double x_max, double B_0, double[:] zz,
                        bint variant_A,
                        int num_threads,
                        ) nogil:
    cdef int i, j
    cdef double[:, :] djx_dy = tls.grad1
    cdef double[:, :] djy_dx = tls.grad2
    pader_y(jx, djx_dy, h, n_dim, num_threads)
    pader_x(jy, djy_dx, h, n_dim, num_threads)
    for i in prange(n_dim, num_threads=num_threads, nogil=True):
        for j in range(n_dim):
            tls.rhs[i, j] = -(djx_dy[i, j] - djy_dx[i, j])

    # TODO!
    out_Bz[...] = 0

    if variant_A:
        for i in range(n_dim):
            for j in range(n_dim):
                out_Bz[i, j] = 2 * out_Bz[i, j] - in_Bz[i, j]


cdef class DirichletSolver:
    def __init__(DirichletSolver self, int N, double h, int num_threads=1):
        self.h, self.N = h, N
        #self.mul = h**2 / (2 * (N - 1))  # total multiplier to compensate for the iDCT+DCT transforms
        self.mul = h**2 / 2 * (N - 1) # for use with unnormed Intel MKL, not scipy

        aa = 2 + 4 * np.sin(np.arange(1, N - 1) * np.pi / (2 * (N - 1)))**2
        alf = np.zeros((N - 2, N - 1))  # precalculated internal coefficients for tridiagonal solving
        for i in range(N - 2):
            alf[:, i + 1] = 1 / (aa - alf[:, i])
        self.alf = alf

        self.bet = np.zeros((N, N))
        self.num_threads = num_threads
        self.tt1 = TrigTransform(self.N - 2, tt_type='dst', num_threads=self.num_threads)
        self.tt2 = TrigTransform(self.N - 2, tt_type='dst', num_threads=self.num_threads)

    cpdef solve(DirichletSolver self, double[:, :] rhs, double[:, :] out):
        # Solve Laplace x = (-)? RHS for x with Dirichlet boundary conditions using DST-1
        # Only operates on the internal cells of rhs and out
        cdef int i, j
        assert rhs.shape[0] == rhs.shape[1] == self.N

        # 1. Apply DST-1 (Discrete Sine Transform Type 1) to the RHS
        #cdef double[:, :] tmp1 = scipy.fftpack.dst(x=rhs[1:-1, 1:-1], type=1).T
        #cdef double[:, :] tmp1 = self.tt.dst_2d(np.array(rhs[1:-1, 1:-1])).T
        self.tt1.array[...] = rhs[1:-1, 1:-1]
        self.tt1.dst_2d()
        cdef double[:, :] tmp1 = self.tt1.array.T  # a view, not a copy!
        cdef double[:, :] tmp2 = self.tt2.array.T  # a view, not a copy!

        # 2. Solve tridiagonal matrix equation for each spectral column with Thomas method:
        # A @ tmp_2[k, :] = tmp_1[k, :]
        # A has -1 on superdiagonal, -1 on subdiagonal and aa[i] at the main diagonal
        for i in prange(self.N - 2, num_threads=self.num_threads, nogil=True):
            self.bet[i, 0] = 0
            for j in range(self.N - 2):
                self.bet[i, j + 1] = (self.mul * tmp1[i, j] + self.bet[i, j]) * self.alf[i, j + 1]
            tmp2[i, self.N - 3] = 0 + self.bet[i, self.N - 2]  # 0 = tmp2[self.N - 2] (fake)
            for j in range(self.N - 4, 0 - 1, -1):
                tmp2[i, j] = self.alf[i, j + 1] * tmp2[i, j + 1] + self.bet[i, j + 1]

        # 3. Apply DST-1 (Discrete Sine Transform Type 1) to the transformed spectra
        #cdef double[:, :] tmp_out = scipy.fftpack.dst(self.tmp2.T, type=1, overwrite_x=True)
        #cdef double[:, :] tmp_out = self.tt.dst_2d(np.array(self.tmp2.T)).T
        self.tt2.dst_2d()  # .T of the argument implied, see above
        out[...] = 0
        out[1:-1, 1:-1] = self.tt2.array


cpdef calculate_Ez(double[:, :] in_Ez,
                        double[:, :] out_Ez,
                        double[:, :] jx,
                        double[:, :] jy,
                        ThreadLocalStorage tls,
                        DirichletSolver ds,
                        unsigned int n_dim,
                        double h,
                        unsigned int npq,
                        bint variant_A,
                        int num_threads,
                        ):
    cdef int i, j
    cdef double[:, :] djx_dx = tls.grad1
    cdef double[:, :] djy_dy = tls.grad2
    pader_x(jx, djx_dx, h, n_dim, num_threads)
    pader_y(jy, djy_dy, h, n_dim, num_threads)
    for i in prange(n_dim, num_threads=num_threads, nogil=True):
        for j in range(n_dim):
            tls.rhs[i, j] = -(djx_dx[i, j] + djy_dy[i, j])

    #reduction_Dirichlet1(tls.rhs, out_Ez, tls, n_dim, h, npq)
    ds.solve(tls.rhs, out_Ez)

    if variant_A:
        for i in range(n_dim):
            for j in range(n_dim):
                out_Ez[i, j] = 2 * out_Ez[i, j] - in_Ez[i, j]


cdef class FieldSolver:
    def __init__(self, n_dim, h, subtraction_trick=1, num_threads=1):
        self.n_dim, self.num_threads = n_dim, num_threads
        self.tls_0 = ThreadLocalStorage(n_dim, h)
        self.tls_1 = ThreadLocalStorage(n_dim, h)
        self.tls_2 = ThreadLocalStorage(n_dim, h)
        self.tls_3 = ThreadLocalStorage(n_dim, h)
        self.tls_4 = ThreadLocalStorage(n_dim, h)
        self.tls_5 = ThreadLocalStorage(n_dim, h)
        self.ds_Ez = DirichletSolver(n_dim, h, num_threads=num_threads)
        self.mxs_Ex = MixedSolver(n_dim, h,
                                  subtraction_trick=subtraction_trick,
                                  num_threads=num_threads)
        self.mxs_Ey = MixedSolver(n_dim, h,
                                  subtraction_trick=subtraction_trick,
                                  num_threads=num_threads)
        self.mxs_Bx = MixedSolver(n_dim, h,
                                  subtraction_trick=subtraction_trick,
                                  num_threads=num_threads)
        self.mxs_By = MixedSolver(n_dim, h,
                                  subtraction_trick=subtraction_trick,
                                  num_threads=num_threads)
        self.zz = np.zeros(n_dim)


    cpdef calculate_fields(FieldSolver self,
                           np.ndarray[RoJ_t, ndim=2] roj_cur,
                           np.ndarray[RoJ_t, ndim=2] roj_prev,
                           double[:, :] in_Ex,
                           double[:, :] in_Ey,
                           double[:, :] in_Ez,
                           double[:, :] in_Bx,
                           double[:, :] in_By,
                           double[:, :] in_Bz,
                           double[:, :] beam_ro,
                           double h,
                           unsigned int npq,
                           double x_max,
                           double h3,
                           double B_0,
                           double[:, :] out_Ex,
                           double[:, :] out_Ey,
                           double[:, :] out_Ez,
                           double[:, :] out_Bx,
                           double[:, :] out_By,
                           double[:, :] out_Bz,
                           bint variant_A):
        cdef int n_dim = self.n_dim
        cdef int i

        if variant_A:
            roj = np.zeros_like(roj_cur)
            for comp in 'ro', 'jx', 'jy', 'jz':
                roj[comp] = (roj_cur[comp] + roj_prev[comp]) / 2
        else:
            roj = roj_cur

        cdef double[:, :] ro = roj['ro'] + beam_ro
        cdef double[:, :] jx = roj['jx']
        cdef double[:, :] jy = roj['jy']
        cdef double[:, :] jz = roj['jz'] + beam_ro
        cdef double[:, :] jx_prev = roj_prev['jx']
        cdef double[:, :] jy_prev = roj_prev['jy']

        cdef double[:, :] out_Ey_T = out_Ey.T
        cdef double[:, :] out_Bx_T = out_Bx.T

        calculate_Ex(in_Ex, out_Ex, ro, jx, jx_prev, self.tls_0, self.mxs_Ex,
                     n_dim, h, h3, npq, self.zz, variant_A, self.num_threads)
        calculate_Ey(in_Ey, out_Ey_T, ro, jy, jy_prev, self.tls_1, self.mxs_Ey,
                     n_dim, h, h3, npq, self.zz, variant_A, self.num_threads)
        calculate_Bx(in_Bx, out_Bx_T, jz, jy, jy_prev, self.tls_2, self.mxs_Bx,
                     n_dim, h, h3, npq, self.zz, variant_A, self.num_threads)
        calculate_By(in_By, out_By, jz, jx, jx_prev, self.tls_3, self.mxs_By,
                     n_dim, h, h3, npq, self.zz, variant_A, self.num_threads)
        #calculate_Bz(in_Bz, out_Bz, jx, jy, self.tls_4,
        #             n_dim, h, npq, x_max, B_0, zz, variant_A)
        out_Bz[...] = 0
        calculate_Ez(in_Ez, out_Ez, jx, jy, self.tls_5, self.ds_Ez,
                     n_dim, h, npq, variant_A, self.num_threads)
        #calculate_Ez(in_Ez, out_Ez, jx, jy, self.tls_5, self.ds_Ez,
        #             n_dim, h, npq, False, self.num_threads)
