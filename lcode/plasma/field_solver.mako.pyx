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

import cython.parallel

import scipy.fftpack


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


cdef void Progonka(double aa,
                   double[:] ff,  # n_dim
                   double[:] vv,  # n_dim
                   ThreadLocalStorage tls
                   ) nogil:
    cdef double qkapa, qmu1, qmu2
    cdef unsigned int i
    tls.alf[0] = tls.bet[0] = 0

    qkapa = 2 / aa
    qmu1 = ff[0] / aa
    qmu2 = ff[-1] / aa
    tls.alf[1] = qkapa
    tls.bet[1] = qmu1
    for i in range(1, ff.shape[0] - 2 + 1):
        tls.alf[i + 1] = 1 / (aa - tls.alf[i])
        tls.bet[i + 1] = (ff[i] + tls.bet[i]) * tls.alf[i + 1]

    tls.bet[-1] = (qmu2 + qkapa * tls.bet[-2]) / (1 - qkapa * tls.alf[-2])
    vv[-1] = tls.bet[-1]
    for i in range(vv.shape[0] - 2, 0 - 1, -1):
        vv[i] = tls.alf[i + 1] * vv[i + 1] + tls.bet[i + 1]


cpdef void Progonka_Dirichlet(double aa,
                              double[:] ff,  # n_dim
                              double[:] vv,  # n_dim
                              ThreadLocalStorage tls,
                              ) nogil:
    cdef unsigned int i
    tls.alf[0] = tls.alf[1] = tls.bet[0] = tls.bet[1] = 0

    # assert(ff.shape[0] == tls.alf.shape[0])
    # assert(vv.shape[0] == tls.alf.shape[0])

    for i in range(1, ff.shape[0] - 2 + 1):
        tls.alf[i + 1] = 1 / (aa - tls.alf[i])
        tls.bet[i + 1] = (ff[i] + tls.bet[i]) * tls.alf[i + 1]
    vv[-1] = 0
    for i in range(vv.shape[0] - 2, 0 - 1, -1):
        vv[i] = tls.alf[i + 1] * vv[i + 1] + tls.bet[i + 1]


cpdef void Posson_reduct_12(double[:] r0,  # n_dim
                            double[:] r1,  # n_dim
                            double[:, :] Fi,  # n_dim, n_dim
                            double[:, :] P,  # n_dim, n_dim
                            ThreadLocalStorage tls,
                            unsigned int n_dim,
                            double h,
                            unsigned int npq,
                            ) nogil:
    cdef double a, a_, alfa
    cdef unsigned int i, j, k
    cdef unsigned long nj1, ii, jk, l, nk
    cdef int m1p

    # Filling P
    for i in range(n_dim):
        P[0, i] = h**2 * (Fi[0, i] + 2 / h * r0[i])
        P[-1, i] = h**2 * (Fi[-1, i] + 2 / h * r1[i])

    for j in range(n_dim):
        P[j, 0] = 0
        P[j, -1] = 0

    for j in range(1, n_dim - 2 + 1):
        for i in range(1, n_dim - 1):
            P[i, j] = h**2 * Fi[i, j]

    # Direct way
    for k in range(1, npq - 1 + 1):  # NOTE: 1
        jk = <unsigned long> (2 ** (k - 1))
        nj1 = <unsigned long> (2 ** (npq - k) - 1)
        for ii in range(1, nj1 + 1):  # NOTE: 1
            j = <unsigned long> (ii * 2 * jk)
            for i in range(n_dim):
                tls.RedFi[i] = P[i, j - jk] + P[i, j + jk]
                tls.Svl[i] = 0

            for l in range(1, jk + 1):  # NOTE: 1
                a_ = 1 + (1 - cos((2 * l - 1) * pi / (2 * jk)))
                a = (2 * a_ + h**2)
                m1p = (-1) ** (l + 1)  # I hate that -1**2 == -1...
                alfa = sin((2 * l - 1) * pi / (2 * jk)) * m1p / jk
                for i in range(n_dim):
                    tls.PrF[i] = alfa * tls.RedFi[i]

                Progonka(a, tls.PrF, tls.PrV, tls)
                for i in range(n_dim):
                    tls.Svl[i] += tls.PrV[i]

            for i in range(n_dim):
                P[i, j] = .5 * (P[i, j] + tls.Svl[i])

    # Back way
    for k in range(npq, 0, -1):
        jk = <unsigned long> (2 ** (k - 1))
        nk = <unsigned long> (2 ** (npq - k))
        for ii in range(1, nk + 1):  # NOTE: 1
            j = (2 * ii - 1) * jk
            for i in range(n_dim):
                tls.RedFi[i] = P[i, j - jk] + P[i, j + jk]
                tls.Psi[i] = P[i, j]
                tls.Svl[i] = 0

            for l in range(1, jk + 1):  # NOTE: 1
                a_ = 1 + (1 - cos((2 * l - 1) * pi / (2 * jk)))
                a = (2 * a_ + h**2)
                m1p = (-1) ** (l + 1)
                alfa = sin((2 * l - 1) * pi / (2 * jk)) * m1p / jk
                for i in range(n_dim):
                    tls.PrF[i] = tls.Psi[i] + alfa * tls.RedFi[i]

                Progonka(a, tls.PrF, tls.PrV, tls)
                for i in range(n_dim):
                    tls.Svl[i] += tls.PrV[i]

            for i in range(n_dim):
                P[i, j] = tls.Svl[i]


cpdef void reduction_Dirichlet1(double[:, :] Fi,  # n_dim, n_dim
                                double[:, :] P,  # n_dim, n_dim
                                ThreadLocalStorage tls,
                                unsigned int n_dim,
                                double h,
                                unsigned int npq,
                                ) nogil:
    cdef double a, alfa
    cdef unsigned int i, j, k
    cdef unsigned long nj1, ii, jk, nk, l
    cdef int m1p

    # Filling P
    for i in range(n_dim):
        P[0, i] = 0
        P[-1, i] = 0
        P[i, 0] = 0
        P[i, -1] = 0

    for j in range(1, n_dim - 1):
        for i in range(1, n_dim - 1):
            P[i, j] = h**2 * Fi[i, j]

    for k in range(1, npq):  # NOTE: 1
        jk = 2 ** (k - 1)
        nj1 = 2 ** (npq - k) - 1
        for ii in range(1, nj1 + 1):  # NOTE: 1
            j = ii * 2 * jk
            for i in range(1, tls.RedFi.shape[0] - 1):  # NOTE: 1
                tls.RedFi[i] = P[i, j - jk] + P[i, j + jk]
                tls.Svl[i] = 0

            for l in range(1, jk + 1):  # NOTE: 1
                m1p = (-1) ** (l + 1)
                a = 2 * (1 + (1 - cos((2 * l - 1) * pi / (2 * jk))))
                alfa = sin((2 * l - 1) * pi / (2 * jk)) * m1p / jk
                for i in range(1, tls.PrF.shape[0] - 1):  # NOTE: 1
                    tls.PrF[i] = alfa * tls.RedFi[i]

                Progonka_Dirichlet(a, tls.PrF, tls.PrV, tls)
                for i in range(1, tls.Svl.shape[0] - 1):  # NOTE: 1
                    tls.Svl[i] += tls.PrV[i]

            for i in range(1, P.shape[0] - 1):  # NOTE: 1
                P[i, j] = .5 * (P[i, j] + tls.Svl[i])
# c  back way********************************************************
    for k in range(npq, 0, -1):
        jk = 2 ** (k - 1)
        nk = 2 ** (npq - k)
        for ii in range(1, nk + 1):  # NOTE: 1
            j = (2 * ii - 1) * jk
            for i in range(1, n_dim - 1):  # NOTE: 1
                tls.RedFi[i] = P[i, j - jk] + P[i, j + jk]
                tls.Psi[i] = P[i, j]
                tls.Svl[i] = 0
            for l in range(1, jk + 1):  # NOTE: 1
                a = 2 * (1 + (1 - cos((2 * l - 1) * pi / (2 * jk))))
                m1p = (-1) ** (l + 1)
                alfa = sin((2 * l - 1) * pi / (2 * jk)) * m1p / jk
                for i in range(1, n_dim - 1):  # NOTE: 1
                    tls.PrF[i] = tls.Psi[i] + alfa * tls.RedFi[i]
                Progonka_Dirichlet(a, tls.PrF, tls.PrV, tls)
                for i in range(1, n_dim - 1):  # NOTE: 1
                    tls.Svl[i] += tls.PrV[i]
            for i in range(1, n_dim - 1):  # NOTE: 1
                P[i, j] = tls.Svl[i]


cpdef void Progonka_C(double[:, :] ff,  # n_dim, n_dim
                      double[:, :] vv,  # n_dim, n_dim
                      ThreadLocalStorage tls,
                      unsigned int n_dim,
                      ) nogil:
    cdef double aa, a_n_prka, b_0_prka
    cdef unsigned int i, j, M
    M = n_dim - 1
    a_n_prka = 2
    b_0_prka = 2
    aa = 4

    tls.alf[1] = b_0_prka / aa
    for j in range(0, M + 1, 2):
        tls.bet[1] = ff[0, j] / aa
        for i in range(1, M - 1 + 1):
            tls.alf[i + 1] = 1 / (aa - tls.alf[i])
            tls.bet[i + 1] = (ff[i, j] + tls.bet[i]) * tls.alf[i + 1]

        tls.bet[-1] = ((ff[M, j] + a_n_prka * tls.bet[M]) /
                       (aa - a_n_prka * tls.alf[M]))
        vv[M, j] = tls.bet[-1]
        for i in range(M - 1, 0 - 1, -1):
            vv[i, j] = tls.alf[i + 1] * vv[i + 1, j] + tls.bet[i + 1]


cpdef void Progonka_C_l_km1(double kk,
                            double[:, :] ff,  # n_dim, n_dim
                            double[:, :] vv,  # n_dim, n_dim
                            ThreadLocalStorage tls,
                            unsigned int n_dim,
                            unsigned int npq,
                            ) nogil:
    cdef double[:, :] p = tls.p_km1
    cdef double aa, a_n_prka, b_0_prka
    cdef unsigned int i, j, M, l, i1, ii, nj1s, jks
    a_n_prka = 2
    b_0_prka = 2
    M = n_dim - 1
    nj1s = <unsigned int> (2 ** (npq - kk))
    jks = <unsigned int> (2 ** (kk - 1))

    # Filling p
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            p[i, j] = 0

    for l in range(1, jks + 1):  # NOTE: 1
        aa = 2 * (1 + (1 - cos((2 * l - 1) * pi / (2 * jks))))
        for i1 in range(1, M - 1 + 1):  # NOTE: 1
            p[i1, i1] = aa
            p[i1, i1 + 1] = -1
            p[i1, i1 - 1] = -1

        p[0, 0] = aa
        p[0, 1] = -1
        p[M, M - 1] = -1
        p[M, M] = aa

        tls.alf[1] = b_0_prka / aa
        for ii in range(nj1s + 1):
            j = ii * 2 * jks
            tls.bet[1] = ff[0, j] / aa

            for i in range(1, M - 1 + 1):  # NOTE: 1
                tls.alf[i + 1] = 1 / (aa - tls.alf[i])
                tls.bet[i + 1] = (ff[i, j] + tls.bet[i]) * tls.alf[i + 1]

            tls.bet[-1] = ((ff[M, j] + a_n_prka * tls.bet[M]) /
                           (aa - a_n_prka * tls.alf[M]))
            vv[M, j] = tls.bet[-1]
            ff[M, j] = vv[M, j]
            for i in range(M - 1, 0 - 1, -1):
                vv[i, j] = tls.alf[i + 1] * vv[i + 1, j] + tls.bet[i + 1]
                ff[i, j] = vv[i, j]


cpdef void Progonka_C_l_km1_0N(double kk,
                               double[:, :] ff,  # n_dim, n_dim
                               double[:, :] vv,  # n_dim, n_dim
                               ThreadLocalStorage tls,
                               unsigned int n_dim,
                               ) nogil:
    cdef double aa, a_n_prka, b_0_prka
    cdef unsigned int i, j, M, l, i1, ii, nj1s, jks

    M = n_dim - 1
    a_n_prka = 2
    b_0_prka = 2
    jks = <unsigned int> 2 ** (kk - 1)

    for l in range(1, jks + 1):  # NOTE: 1
        aa = 2 * (1 + (1 - cos((2 * l - 1) * pi / (2 * jks))))
        tls.alf[1] = b_0_prka / aa
        j = 0
        while j < M + 1:
            tls.bet[1] = ff[0, j] / aa
            for i in range(1, M - 1 + 1):  # NOTE: 1
                tls.alf[i + 1] = 1 / (aa - tls.alf[i])
                tls.bet[i + 1] = (ff[i, j] + tls.bet[i]) * tls.alf[i + 1]

            tls.bet[-1] = ((ff[M, j] + a_n_prka * tls.bet[M]) /
                           (aa - a_n_prka * tls.alf[M]))
            vv[M, j] = tls.bet[-1]
            ff[M, j] = vv[M, j]
            for i in range(M - 1, 0 - 1, -1):
                vv[i, j] = tls.alf[i + 1] * vv[i + 1, j] + tls.bet[i + 1]
                ff[i, j] = vv[i, j]
            j += M  # Why such a strange iteration?..


cpdef void Progonka_C_l_km1_0N_Y(double kk,
                                 double p_0N,
                                 double[:, :] ff,  # n_dim, n_dim
                                 double[:, :] vv,  # n_dim, n_dim
                                 ThreadLocalStorage tls,
                                 unsigned int n_dim,
                                 ) nogil:
    cdef double aa, a_n_prka, b_0_prka, nj1s
    cdef unsigned int i, j, M, l, i1, ii, jks, jks2

    a_n_prka = 2
    b_0_prka = 2
    M = n_dim - 1
    jks = <unsigned int> (2 ** (kk - 1))
    jks2 = 2 * jks
    for l in range(1, jks2 - 1 + 1):  # NOTE: 1
        aa = 2 * (1 + (1 - cos(pi * l / jks)))
        tls.alf[1] = b_0_prka / aa
        j = 0
        tls.bet[1] = ff[0, j] / aa
        for i in range(1, M - 1 + 1):  # NOTE: 1
            tls.alf[i + 1] = 1 / (aa - tls.alf[i])
            tls.bet[i + 1] = (ff[i, j] + tls.bet[i]) * tls.alf[i + 1]

        tls.bet[-1] = ((ff[M, j] + a_n_prka * tls.bet[M]) /
                       (aa - a_n_prka * tls.alf[M]))
        vv[M, j] = tls.bet[-1]
        ff[M, j] = vv[M, j]
        for i in range(M - 1, 0 - 1, -1):
            vv[i, j] = tls.alf[i + 1] * vv[i + 1, j] + tls.bet[i + 1]
            ff[i, j] = vv[i, j]

    l = jks2
    aa = 2 * (1 + (1 - cos(pi * l / jks)))
    tls.alf[1] = b_0_prka / aa
    j = 0
    tls.bet[1] = ff[0, j] / aa
    for i in range(1, M - 1 + 1):  # NOTE: 1
        tls.alf[i + 1] = 1 / (aa - tls.alf[i])
        tls.bet[i + 1] = (ff[i, j] + tls.bet[i]) * tls.alf[i + 1]

    tls.bet[-1] = -p_0N
    vv[M, j] = tls.bet[-1]
    ff[M, j] = vv[M, j]
    for i in range(M - 1, 0 - 1, -1):
        vv[i, j] = tls.alf[i + 1] * vv[i + 1, j] + tls.bet[i + 1]
        ff[i, j] = vv[i, j]


cpdef void Progonka_C_Y(double[:, :] ff,  # n_dim, n_dim
                        double[:, :] vv,  # n_dim, n_dim
                        ThreadLocalStorage tls,
                        unsigned int n_dim,
                        ) nogil:
    cdef double aa, a_n_prka, b_0_prka
    cdef unsigned int i, j, M
    a_n_prka = 2
    b_0_prka = 2
    M = n_dim - 1
    aa = 4

    tls.alf[1] = b_0_prka / aa
    for j in range(1, M - 1 + 1, 2):
        tls.bet[1] = ff[0, j] / aa
        for i in range(1, M - 1 + 1):  # NOTE: 1
            tls.alf[i + 1] = 1 / (aa - tls.alf[i])
            tls.bet[i + 1] = (ff[i, j] + tls.bet[i]) * tls.alf[i + 1]

        tls.bet[-1] = ((ff[M, j] + a_n_prka * tls.bet[M]) /
                       (aa - a_n_prka * tls.alf[M]))
        vv[M, j] = tls.bet[-1]
        for i in range(M - 1, 0 - 1, -1):
            vv[i, j] = tls.alf[i + 1] * vv[i + 1, j] + tls.bet[i + 1]


cpdef void Progonka_C_l_km1_Y(unsigned int kk,
                              double[:, :] ff,  # n_dim, n_dim
                              double[:, :] vv,  # n_dim, n_dim
                              ThreadLocalStorage tls,
                              unsigned int n_dim,
                              unsigned int npq,
                              ) nogil:
    cdef double aa, a_n_prka, b_0_prka
    cdef unsigned int i, j, M, l, ii, nks, jks
    a_n_prka = 2
    b_0_prka = 2

    M = n_dim - 1
    nks = <unsigned int> 2 ** (npq - kk + 1)
    jks = <unsigned int> 2 ** (kk - 1)

    for l in range(1, jks + 1):  # NOTE: 1
        aa = 2 * (1 + (1 - cos((2 * l - 1) * pi / (2 * jks))))
        tls.alf[1] = b_0_prka / aa
        for ii in range(1, nks - 1 + 1, 2):
            j = ii * jks
            tls.bet[1] = ff[0, j] / aa

            for i in range(1, M - 1 + 1):  # NOTE: 1
                tls.alf[i + 1] = 1 / (aa - tls.alf[i])
                tls.bet[i + 1] = (ff[i, j] + tls.bet[i]) * tls.alf[i + 1]

            tls.bet[-1] = ((ff[M, j] + a_n_prka * tls.bet[M]) /
                           (aa - a_n_prka * tls.alf[M]))
            vv[M, j] = tls.bet[-1]
            ff[M, j] = vv[M, j]
            for i in range(M - 1, 0 - 1, -1):
                vv[i, j] = tls.alf[i + 1] * vv[i + 1, j] + tls.bet[i + 1]
                ff[i, j] = vv[i, j]


cpdef void Neuman_red(double B_00,
                      double[:] r0,  # n_dim
                      double[:] r1,  # n_dim
                      double[:] rb,  # n_dim
                      double[:] ru,  # n_dim
                      double[:, :] q,  # n_dim, n_dim
                      double[:, :] YE,  # n_dim, n_dim
                      ThreadLocalStorage tls,
                      unsigned int n_dim,
                      double h,
                      unsigned int npq,
                      double x_max,
                      ) nogil:
    ## TODO: move this allocation to colder code
    #cdef np.ndarray[double, ndim=2] p = np.zeros((n_dim, n_dim))
    #cdef np.ndarray[double, ndim=2] v = np.zeros((n_dim, n_dim))
    #cdef np.ndarray[double, ndim=2] v1 = np.zeros((n_dim, n_dim))
    cdef double[:, :] p = tls.p
    cdef double[:, :] v = tls.v
    cdef double[:, :] v1 = tls.v1
    cdef double p_M0, a, s_sol, x, y, s_numsol
    cdef unsigned int i, j, k, nj1, ii, jk, nk, M, N1
# c *************Direct way*************************************
    M = n_dim - 1
    for j in range(M + 1):
        for i in range(M + 1):
            YE[i, j] = 0
            p[i, j] = 0
            v[i, j] = 0
            v1[i, j] = 0

    for j in range(M + 1):
        q[0, j] = h**2 * (q[0, j] + 2 / h * r0[j])
        q[M, j] = h**2 * (q[M, j] + 2 / h * r1[j])

    for i in range(1, M):  # NOTE: 1
        q[i, 0] = h**2 * (q[i, 0] + 2 / h * rb[i])
        q[i, M] = h**2 * (q[i, M] + 2 / h * ru[i])

    q[0, 0] += h * 2 * rb[0]
    q[0, M] += h * 2 * ru[0]
    q[M, 0] += h * 2 * rb[M]
    q[M, M] += h * 2 * ru[M]

    for j in range(1, M):  # NOTE: 1
        for i in range(1, M):  # NOTE: 1
            q[i, j] = h**2 * q[i, j]

    # ******************************************
    k = 1
    jk = 1
    for j in range(0, M + 1, 2):
        for i in range(M + 1):
            v[i, j] = q[i, j]

    Progonka_C(v, v1, tls, n_dim)
    for j in range(2, M - 2 + 1, 2):
        for i in range(M + 1):
            p[i, j] = v1[i, j]
            q[i, j] = q[i, j - 1] + q[i, j + 1] + 2 * p[i, j]

    for i in range(M + 1):
        p[i, 0] = v1[i, 0]
        q[i, 0] = 2 * q[i, 1] + 2 * p[i, 0]
        p[i, M] = v1[i, M]
        q[i, M] = 2 * q[i, M - 1] + 2 * p[i, M]

    # *******************************************
    k = 2

    for k in range(2, npq - 1 + 1):  # NOTE: 1
        nj1 = <unsigned int> (2 ** (npq - k) - 1)
        jk = <unsigned int> (2 ** (k - 1))
        for ii in range(1, nj1 + 1):  # NOTE: 1
            j = ii * 2 * jk
            for i in range(M + 1):
                v[i, j] = p[i, j - jk] + p[i, j + jk] + q[i, j]

        for i in range(M + 1):
            v[i, 0] = 2 * p[i, jk] + q[i, 0]
            v[i, M] = 2 * p[i, M - jk] + q[i, M]

        Progonka_C_l_km1(k, v, v1, tls, n_dim, npq)
        for i in range(M + 1):
            for ii in range(1, nj1 + 1):  # NOTE: 1
                j = ii * 2 * jk  # pow(2, k)
                p[i, j] += v1[i, j]
                q[i, j] = q[i, j - jk] + q[i, j + jk] + 2 * p[i, j]

            p[i, 0] += v1[i, 0]
            q[i, 0] = 2 * q[i, jk] + 2 * p[i, 0]
            p[i, M] += v1[i, M]
            q[i, M] = 2 * q[i, M - jk] + 2 * p[i, M]

    # ************************************
    k = npq

    jk = <unsigned int> (2 ** (npq - 1))
    for i in range(M + 1):
        v[i, 0] = 2 * p[i, jk] + q[i, 0]
        v[i, M] = 2 * p[i, M - jk] + q[i, M]

    Progonka_C_l_km1_0N(k, v, v1, tls, n_dim)
    for i in range(M + 1):
        p[i, 0] += v1[i, 0]
        q[i, 0] = 2 * q[i, jk] + 2 * p[i, 0]
        p[i, M] += v1[i, M]
        q[i, M] = 2 * q[i, M - jk] + 2 * p[i, M]

    # c  back way********* YN  Y0 ***************************
    p_M0 = p[M, 0]

    for i in range(M + 1):
        v[i, 0] = 2 * p[i, 0] + q[i, M]

    Progonka_C_l_km1_0N_Y(npq, p_M0, v, v1, tls, n_dim)
    for i in range(M + 1):
        YE[i, 0] = p[i, 0] + v1[i, 0]
        YE[i, M] = p[i, M] + v1[i, 0]

    for k in range(npq, 2 - 1, -1):
        jk = <unsigned int> (2 ** (k - 1))
        nk = <unsigned int> (2 ** (npq - k + 1))
        for ii in range(1, nk - 1 + 1, 2):  # NOTE: 1
            j = ii * jk
            for i in range(M + 1):
                v[i, j] = YE[i, j - jk] + YE[i, j + jk] + q[i, j]
        Progonka_C_l_km1_Y(k, v, v1, tls, n_dim, npq)
        for ii in range(1, nk - 1 + 1, 2):  # NOTE: 1
            j = ii * jk
            for i in range(M + 1):
                YE[i, j] = p[i, j] + v1[i, j]
    k = 1
    for j in range(1, M - 1 + 1, 2):
        for i in range(M + 1):
            v[i, j] = YE[i, j - 1] + YE[i, j + 1] + q[i, j]
    Progonka_C_Y(v, v1, tls, n_dim)
    for j in range(1, M - 1 + 1, 2):
        for i in range(M + 1):
            YE[i, j] = v1[i, j]

    # Proverka*******************************************
    s_numsol = 0
    for i in range(M + 1):
        for j in range(M + 1):
            s_numsol += YE[i, j]

    s_numsol /= (M + 1) * (M + 1)
    # print("@@@@@@@@ s_numsol  %e" % s_numsol)
    for i in range(M + 1):
        for j in range(M + 1):
            YE[i, j] -= s_numsol + B_00 / (x_max * 2)**2



cpdef void pader_x(double[:, :] in_, double[:, :] out, double h, int n_dim) nogil:
    cdef int i, j
    cdef double h2 = h * 2
    for i in range(1, n_dim - 1):
        for j in range(n_dim):
            out[i, j] = (in_[i + 1, j] - in_[i - 1, j]) / h2
    for j in range(n_dim):
        out[0, j] = out[n_dim - 1, j] = 0
        # WRONG
        ##out[0, j] = 2 * (in_[1, j] - in_[0, j]) / h
        ##out[-1, j] = 2 * (in_[-1, j] - 2 * in_[-2, j]) / h


cpdef void pader_y(double[:, :] in_, double[:, :] out, double h, int n_dim) nogil:
    cdef int i, j
    cdef double h2 = h * 2
    for i in range(n_dim):
        for j in range(1, n_dim - 1):
            out[i, j] = (in_[i, j + 1] - in_[i, j - 1]) / h2
    for i in range(n_dim):
        out[i, 0] = out[i, n_dim - 1] = 0
        # WRONG
        ##out[i, 0] = 2 * (in_[i, 1] - in_[i, 0]) / h
        ##out[i, -1] = 2 * (in_[i, 1] - in_[i, -2]) / h


cpdef void pader_xi(double[:, :] in_prev, double[:, :] in_cur,
                    double[:, :] out, double h3, int n_dim) nogil:
    cdef int i, j
    for i in range(n_dim):
        for j in range(n_dim):
            out[i, j] = (in_prev[i, j] - in_cur[i, j]) / h3


cdef class MixedSolver:
    def __init__(MixedSolver self, int N, double h, double subtraction_trick=1):
        self.h, self.N = h, N
        self.subtraction_trick = subtraction_trick
        self.mul = h**2  / (2 * (N - 1))  # total multiplier to compensate for the iDCT+DCT transforms

        aa = 2 + 4 * np.sin(np.arange(0, N) * np.pi / (2 * (N - 1)))**2  # diagonal matrix elements
        if subtraction_trick:
            aa += self.h**2 * subtraction_trick
        alf = np.zeros((N, N + 1))  # precalculated internal coefficients for tridiagonal solving
        for i in range(1, N):
            alf[:, i + 1] = 1 / (aa - alf[:, i])
        self.alf = alf

        self.bet = np.zeros((N, N))
        # TODO: save space by using fewer arrays? or that would be negated with the FFTW interface?
        self.rhs_fixed = np.zeros((N, N))
        self.tmp1 = np.zeros((N, N))
        self.tmp2 = np.zeros((N, N))

    cpdef solve(MixedSolver self, double[:, :] rhs, double[:] bound_top, double[:] bound_bot, double[:, :] out):
        # Solve Laplace x = (-)? RHS for x with mixed boundary conditions using DCT-1
        cdef int i, j
        assert rhs.shape[0] == rhs.shape[1] == self.N

        # 1. Apply boundary conditions to the rhs
        self.rhs_fixed[...] = rhs.T
        for i in range(self.N):
            self.rhs_fixed[i, 0] += bound_top[i] * (2 / self.h)
            self.rhs_fixed[i, self.N - 1] += bound_bot[i] * (2 / self.h)
            # self.rhs_fixed[0, i] = self.rhs_fixed[self.N - 1, i] = 0  # changes nothing???

        # 2. Apply iDCT-1 (inverse Discrete Cosine Transform Type 1) to the RHS
        # TODO: accelerated version using fftw with extra codelets via ctypes?
        cdef double[:, :] tmp1 = scipy.fftpack.idct(x=self.rhs_fixed, type=1, overwrite_x=True).T
        self.tmp1[...] = tmp1

        # 3. Solve tridiagonal matrix equation for each spectral column with Thomas method:
        # A @ tmp_2[k, :] = tmp_1[k, :]
        # A has -1 on superdiagonal, -1 on subdiagonal and aa[k] at the main diagonal
        # The edge elements of each column are forced to 0!
        for i in range(self.N):  # TODO: change indices
            self.bet[i, 0] = 0
            for j in range(1, self.N - 1):
                self.bet[i, j + 1] = (self.mul * self.tmp1[i, j] + self.bet[i, j]) * self.alf[i, j + 1]
            self.tmp2[i, self.N - 1] = 0  # note the forced zero
            for j in range(self.N - 2, 0 - 1, -1):
                self.tmp2[i, j] = self.alf[i, j + 1] * self.tmp2[i, j + 1] + self.bet[i, j + 1]
            # self.tmp2[:, 0] == 0, it happens by itself

        # EXTRA: suppression
        #for i in range(self.N):
        #    for j in range(self.N):
        #        #self.tmp2[i, j] *= exp(-((i+1)/self.N + .1)**10)
        #        self.tmp2[i, j] *= 1 - (i/self.N)**2

        # 4. Apply DCT-1 (Discrete Cosine Transform Type 1) to the transformed spectra
        cdef double[:, :] tmp_out = scipy.fftpack.dct(self.tmp2.T, type=1, overwrite_x=True)
        out[...] = tmp_out.T


cpdef void calculate_Ex(double[:, :] in_Ex, double[:, :] out_Ex,
                        double[:, :] ro,
                        double[:, :] jx, double[:, :] jx_prev,
                        ThreadLocalStorage tls,
                        MixedSolver mxs,
                        unsigned int n_dim, double h, double h3,
                        unsigned int npq,
                        double[:] zz,
                        bint variant_A=False,
                        ):
    cdef int i, j
    cdef double[:, :] dro_dx = tls.grad1
    cdef double[:, :] djx_dxi = tls.grad2
    pader_x(ro, dro_dx, h, n_dim)
    pader_xi(jx_prev, jx, djx_dxi, h3, n_dim)
    for i in range(n_dim):
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
                        bint variant_A=False,
                        ):
    cdef int i, j
    cdef double[:, :] dro_dy = tls.grad1
    cdef double[:, :] djy_dxi = tls.grad2
    pader_y(ro, dro_dy, h, n_dim)
    pader_xi(jy_prev, jy, djy_dxi, h3, n_dim)
    for i in range(n_dim):
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
                        bint variant_A=False,
                        ):
    cdef int i, j
    cdef double[:, :] djz_dy = tls.grad1
    cdef double[:, :] djy_dxi = tls.grad2
    pader_y(jz, djz_dy, h, n_dim)
    pader_xi(jy_prev, jy, djy_dxi, h3, n_dim)
    for i in range(n_dim):
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
                        bint variant_A=False,
                        ):
    cdef int i, j
    cdef double[:, :] djz_dx = tls.grad1
    cdef double[:, :] djx_dxi = tls.grad2
    pader_x(jz, djz_dx, h, n_dim)
    pader_xi(jx_prev, jx, djx_dxi, h3, n_dim)
    for i in range(n_dim):
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
                        bint variant_A=False,
                        ) nogil:
    cdef int i, j
    cdef double[:, :] djx_dy = tls.grad1
    cdef double[:, :] djy_dx = tls.grad2
    pader_y(jx, djx_dy, h, n_dim)
    pader_x(jy, djy_dx, h, n_dim)
    for i in range(n_dim):
        for j in range(n_dim):
            tls.rhs[i, j] = -(djx_dy[i, j] - djy_dx[i, j])

    Neuman_red(B_0, zz, zz, zz, zz, tls.rhs, out_Bz, tls, n_dim, h, npq, x_max)

    if variant_A:
        for i in range(n_dim):
            for j in range(n_dim):
                out_Bz[i, j] = 2 * out_Bz[i, j] - in_Bz[i, j]


cdef class DirichletSolver:
    def __init__(DirichletSolver self, int N, double h):
        self.h, self.N = h, N
        self.mul = h**2 / (2 * (N - 1))  # total multiplier to compensate for the iDCT+DCT transforms

        aa = 2 + 4 * np.sin(np.arange(1, N - 1) * np.pi / (2 * (N - 1)))**2
        alf = np.zeros((N - 2, N - 1))  # precalculated internal coefficients for tridiagonal solving
        for i in range(N - 2):
            alf[:, i + 1] = 1 / (aa - alf[:, i])
        self.alf = alf

        self.bet = np.zeros((N, N))
        # TODO: save space by using fewer arrays? or that would be negated with the FFTW interface?
        self.tmp1 = np.zeros((N - 2, N - 2))
        self.tmp2 = np.zeros((N - 2, N - 2))

    cpdef solve(DirichletSolver self, double[:, :] rhs, double[:, :] out):
        # Solve Laplace x = (-)? RHS for x with Dirichlet boundary conditions using DST-1
        # Only operates on the internal cells of rhs and out
        cdef int i, j
        assert rhs.shape[0] == rhs.shape[1] == self.N

        # 1. Apply iDCT-1 (inverse Discrete Cosine Transform Type 1) to the RHS
        # TODO: accelerated version using fftw with extra codelets via ctypes?
        cdef double[:, :] tmp1 = scipy.fftpack.dst(x=rhs[1:-1, 1:-1], type=1).T
        self.tmp1[...] = tmp1

        # 2. Solve tridiagonal matrix equation for each spectral column with Thomas method:
        # A @ tmp_2[k, :] = tmp_1[k, :]
        # A has -1 on superdiagonal, -1 on subdiagonal and aa[i] at the main diagonal
        for i in range(self.N - 2):
            self.bet[i, 0] = 0
            for j in range(self.N - 2):
                self.bet[i, j + 1] = (self.mul * self.tmp1[i, j] + self.bet[i, j]) * self.alf[i, j + 1]
            self.tmp2[i, self.N - 3] = 0 + self.bet[i, self.N - 2]  # 0 = self.tmp2[self.N - 2] (fake)
            for j in range(self.N - 4, 0 - 1, -1):
                self.tmp2[i, j] = self.alf[i, j + 1] * self.tmp2[i, j + 1] + self.bet[i, j + 1]

        # 3. Apply DCT-1 (Discrete Cosine Transform Type 1) to the transformed spectra
        cdef double[:, :] tmp_out = scipy.fftpack.dst(self.tmp2.T, type=1, overwrite_x=True)
        out[...] = 0
        out[1:-1, 1:-1] = tmp_out


cpdef calculate_Ez(double[:, :] in_Ez,
                        double[:, :] out_Ez,
                        double[:, :] jx,
                        double[:, :] jy,
                        ThreadLocalStorage tls,
                        DirichletSolver ds,
                        unsigned int n_dim,
                        double h,
                        unsigned int npq,
                        bint variant_A=False,
                        ):
    cdef int i, j
    cdef double[:, :] djx_dx = tls.grad1
    cdef double[:, :] djy_dy = tls.grad2
    pader_x(jx, djx_dx, h, n_dim)
    pader_y(jy, djy_dy, h, n_dim)
    for i in range(n_dim):
        for j in range(n_dim):
            tls.rhs[i, j] = -(djx_dx[i, j] + djy_dy[i, j])

    #reduction_Dirichlet1(tls.rhs, out_Ez, tls, n_dim, h, npq)
    ds.solve(tls.rhs, out_Ez)

    if variant_A:
        for i in range(n_dim):
            for j in range(n_dim):
                out_Ez[i, j] = 2 * out_Ez[i, j] - in_Ez[i, j]


cdef class FieldSolver:
    def __init__(self, n_dim, h, subtraction_trick=1, iterations=1, threads=1):
        self.n_dim, self.threads, self.iterations = n_dim, threads, iterations
        self.tls_0 = ThreadLocalStorage(n_dim, h)
        self.tls_1 = ThreadLocalStorage(n_dim, h)
        self.tls_2 = ThreadLocalStorage(n_dim, h)
        self.tls_3 = ThreadLocalStorage(n_dim, h)
        self.tls_4 = ThreadLocalStorage(n_dim, h)
        self.tls_5 = ThreadLocalStorage(n_dim, h)
        self.ds_Ez = DirichletSolver(n_dim, h)
        self.mxs_Ex = MixedSolver(n_dim, h, subtraction_trick=subtraction_trick)
        self.mxs_Ey = MixedSolver(n_dim, h, subtraction_trick=subtraction_trick)
        self.mxs_Bx = MixedSolver(n_dim, h, subtraction_trick=subtraction_trick)
        self.mxs_By = MixedSolver(n_dim, h, subtraction_trick=subtraction_trick)

    cpdef calculate_fields(self,
                           np.ndarray[RoJ_t, ndim=2] roj_cur,
                           np.ndarray[RoJ_t, ndim=2] roj_prev,
                           double[:, :] in_Ex_,
                           double[:, :] in_Ey_,
                           double[:, :] in_Ez_,
                           double[:, :] in_Bx_,
                           double[:, :] in_By_,
                           double[:, :] in_Bz_,
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
        cdef int n_dim = self.n_dim, threads = self.threads
        cdef int i

        if variant_A:
            roj = np.zeros_like(roj_cur)
            for comp in 'ro', 'jx', 'jy', 'jz':
                roj[comp] = (roj_cur[comp] + roj_prev[comp]) / 2
        else:
            roj = roj_cur

        cdef double[:, :] in_Ex = in_Ex_.copy()
        cdef double[:, :] in_Ey = in_Ey_.copy()
        cdef double[:, :] in_Ez = in_Ez_.copy()
        cdef double[:, :] in_Bx = in_Bx_.copy()
        cdef double[:, :] in_By = in_By_.copy()
        cdef double[:, :] in_Bz = in_Bz_.copy()
        cdef double[:, :] ro = roj['ro'] + beam_ro
        cdef double[:, :] jx = roj['jx']
        cdef double[:, :] jy = roj['jy']
        cdef double[:, :] jz = roj['jz'] + beam_ro
        cdef double[:, :] jx_prev = roj_prev['jx']
        cdef double[:, :] jy_prev = roj_prev['jy']

        cdef double[:, :] out_Ey_T = out_Ey.T
        cdef double[:, :] out_Bx_T = out_Bx.T

        cdef double[:] zz = np.zeros(n_dim)

        for i in range(self.iterations):
            calculate_Ex(in_Ex, out_Ex, ro, jx, jx_prev, self.tls_0, self.mxs_Ex,
                         n_dim, h, h3, npq, zz, variant_A)
            calculate_Ey(in_Ey, out_Ey_T, ro, jy, jy_prev, self.tls_1, self.mxs_Ey,
                         n_dim, h, h3, npq, zz, variant_A)
            calculate_Bx(in_Bx, out_Bx_T, jz, jy, jy_prev, self.tls_2, self.mxs_Bx,
                         n_dim, h, h3, npq, zz, variant_A)
            calculate_By(in_By, out_By, jz, jx, jx_prev, self.tls_3, self.mxs_By,
                         n_dim, h, h3, npq, zz, variant_A)
            in_Ex[...] = out_Ex
            in_Ey[...] = out_Ey
            in_Bx[...] = out_Bx
            in_By[...] = out_By
        #calculate_Bz(in_Bz, out_Bz, jx, jy, self.tls_4,
        #             n_dim, h, npq, x_max, B_0, zz, variant_A)
        out_Bz[...] = 0
        calculate_Ez(in_Ez, out_Ez, jx, jy, self.tls_5, self.ds_Ez,
                     n_dim, h, npq, variant_A)
