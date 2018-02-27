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


from libc.math cimport sin, cos
from libc.math cimport M_PI as pi  # 3.141592653589793 on my machine

import numpy as np
cimport numpy as np


cdef class ProgonkaTmp:
    # two very popular temporary arrays
    cdef double[:] alf  # n_dim
    cdef double[:] bet  # n_dim + 1
    # temporary arrays for Posson_reduct_12 and reduction_Dirichlet1
    cdef double[:] RedFi  # n_dim
    cdef double[:] Svl  # n_dim
    cdef double[:] PrF  # n_dim
    cdef double[:] PrV  # n_dim
    cdef double[:] Psi  # n_dim

    def __init__(self, n_dim):
        self.alf = np.zeros(n_dim)
        self.bet = np.zeros(n_dim + 1)
        self.RedFi = np.zeros(n_dim)
        self.Svl = np.zeros(n_dim)
        self.PrF = np.zeros(n_dim)
        self.PrV = np.zeros(n_dim)
        self.Psi = np.zeros(n_dim)


cdef void Progonka(double aa,
                   double[:] ff,  # n_dim
                   double[:] vv,  # n_dim
                   ProgonkaTmp tmp):
    cdef double qkapa, qmu1, qmu2
    cdef unsigned int i
    tmp.alf[0] = tmp.bet[0] = 0

    qkapa = 2 / aa
    qmu1 = ff[0] / aa
    qmu2 = ff[-1] / aa
    tmp.alf[1] = qkapa
    tmp.bet[1] = qmu1
    for i in range(1, ff.shape[0] - 2 + 1):
        tmp.alf[i + 1] = 1 / (aa - tmp.alf[i])
        tmp.bet[i + 1] = (ff[i] + tmp.bet[i]) * tmp.alf[i + 1]

    tmp.bet[-1] = (qmu2 + qkapa * tmp.bet[-2]) / (1 - qkapa * tmp.alf[-2])
    vv[-1] = tmp.bet[-1]
    for i in range(vv.shape[0] - 2, 0 - 1, -1):
        vv[i] = tmp.alf[i + 1] * vv[i + 1] + tmp.bet[i + 1]


cpdef void Progonka_Dirichlet(double aa,
                              double[:] ff,  # n_dim
                              double[:] vv,  # n_dim
                              ProgonkaTmp tmp):
    cdef unsigned int i
    tmp.alf[0] = tmp.alf[1] = tmp.bet[0] = tmp.bet[1] = 0

    assert(ff.shape[0] == tmp.alf.shape[0])
    assert(vv.shape[0] == tmp.alf.shape[0])

    for i in range(1, ff.shape[0] - 2 + 1):
        tmp.alf[i + 1] = 1 / (aa - tmp.alf[i])
        tmp.bet[i + 1] = (ff[i] + tmp.bet[i]) * tmp.alf[i + 1]
    vv[-1] = 0
    for i in range(vv.shape[0] - 2, 0 - 1, -1):
        vv[i] = tmp.alf[i + 1] * vv[i + 1] + tmp.bet[i + 1]


cpdef void Posson_reduct_12(double[:] r0,  # n_dim
                            double[:] r1,  # n_dim
                            double[:, :] Fi,  # n_dim, n_dim
                            double[:, :] P,  # n_dim, n_dim
                            ProgonkaTmp tmp,
                            unsigned int n_dim,
                            double h,
                            unsigned int npq,
                            ):
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
                tmp.RedFi[i] = P[i, j - jk] + P[i, j + jk]
                tmp.Svl[i] = 0

            for l in range(1, jk + 1):  # NOTE: 1
                a_ = 1 + (1 - cos((2 * l - 1) * pi / (2 * jk)))
                a = (2 * a_ + h**2)
                m1p = (-1) ** (l + 1)  # I hate that -1**2 == -1...
                alfa = sin((2 * l - 1) * pi / (2 * jk)) * m1p / jk
                for i in range(n_dim):
                    tmp.PrF[i] = alfa * tmp.RedFi[i]

                Progonka(a, tmp.PrF, tmp.PrV, tmp)
                for i in range(n_dim):
                    tmp.Svl[i] += tmp.PrV[i]

            for i in range(n_dim):
                P[i, j] = .5 * (P[i, j] + tmp.Svl[i])

    # Back way
    for k in range(npq, 0, -1):
        jk = <unsigned long> (2 ** (k - 1))
        nk = <unsigned long> (2 ** (npq - k))
        for ii in range(1, nk + 1):  # NOTE: 1
            j = (2 * ii - 1) * jk
            for i in range(n_dim):
                tmp.RedFi[i] = P[i, j - jk] + P[i, j + jk]
                tmp.Psi[i] = P[i, j]
                tmp.Svl[i] = 0

            for l in range(1, jk + 1):  # NOTE: 1
                a_ = 1 + (1 - cos((2 * l - 1) * pi / (2 * jk)))
                a = (2 * a_ + h**2)
                m1p = (-1) ** (l + 1)
                alfa = sin((2 * l - 1) * pi / (2 * jk)) * m1p / jk
                for i in range(n_dim):
                    tmp.PrF[i] = tmp.Psi[i] + alfa * tmp.RedFi[i]

                Progonka(a, tmp.PrF, tmp.PrV, tmp)
                for i in range(n_dim):
                    tmp.Svl[i] += tmp.PrV[i]

            for i in range(n_dim):
                P[i, j] = tmp.Svl[i]


cpdef void reduction_Dirichlet1(double[:, :] Fi,  # n_dim, n_dim
                                double[:, :] P,  # n_dim, n_dim
                                ProgonkaTmp tmp,
                                unsigned int n_dim,
                                double h,
                                unsigned int npq,
                                ):
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
            for i in range(1, tmp.RedFi.shape[0] - 1):  # NOTE: 1
                tmp.RedFi[i] = P[i, j - jk] + P[i, j + jk]
                tmp.Svl[i] = 0

            for l in range(1, jk + 1):  # NOTE: 1
                m1p = (-1) ** (l + 1)
                a = 2 * (1 + (1 - cos((2 * l - 1) * pi / (2 * jk))))
                alfa = sin((2 * l - 1) * pi / (2 * jk)) * m1p / jk
                for i in range(1, tmp.PrF.shape[0] - 1):  # NOTE: 1
                    tmp.PrF[i] = alfa * tmp.RedFi[i]

                Progonka_Dirichlet(a, tmp.PrF, tmp.PrV, tmp)
                for i in range(1, tmp.Svl.shape[0] - 1):  # NOTE: 1
                    tmp.Svl[i] += tmp.PrV[i]

            for i in range(1, P.shape[0] - 1):  # NOTE: 1
                P[i, j] = .5 * (P[i, j] + tmp.Svl[i])
# c  back way********************************************************
    for k in range(npq, 0, -1):
        jk = 2 ** (k - 1)
        nk = 2 ** (npq - k)
        for ii in range(1, nk + 1):  # NOTE: 1
            j = (2 * ii - 1) * jk
            for i in range(1, n_dim - 1):  # NOTE: 1
                tmp.RedFi[i] = P[i, j - jk] + P[i, j + jk]
                tmp.Psi[i] = P[i, j]
                tmp.Svl[i] = 0
            for l in range(1, jk + 1):  # NOTE: 1
                a = 2 * (1 + (1 - cos((2 * l - 1) * pi / (2 * jk))))
                m1p = (-1) ** (l + 1)
                alfa = sin((2 * l - 1) * pi / (2 * jk)) * m1p / jk
                for i in range(1, n_dim - 1):  # NOTE: 1
                    tmp.PrF[i] = tmp.Psi[i] + alfa * tmp.RedFi[i]
                Progonka_Dirichlet(a, tmp.PrF, tmp.PrV, tmp)
                for i in range(1, n_dim - 1):  # NOTE: 1
                    tmp.Svl[i] += tmp.PrV[i]
            for i in range(1, n_dim - 1):  # NOTE: 1
                P[i, j] = tmp.Svl[i]


cpdef Progonka_C(double[:, :] ff,  # n_dim, n_dim
                 double[:, :] vv,  # n_dim, n_dim
                 ProgonkaTmp tmp,
                 unsigned int n_dim,
                 ):
    cdef double aa, a_n_prka, b_0_prka
    cdef unsigned int i, j, M
    M = n_dim - 1
    a_n_prka = 2
    b_0_prka = 2
    aa = 4

    tmp.alf[1] = b_0_prka / aa
    for j in range(0, M + 1, 2):
        tmp.bet[1] = ff[0, j] / aa
        for i in range(1, M - 1 + 1):
            tmp.alf[i + 1] = 1 / (aa - tmp.alf[i])
            tmp.bet[i + 1] = (ff[i, j] + tmp.bet[i]) * tmp.alf[i + 1]

        tmp.bet[-1] = ((ff[M, j] + a_n_prka * tmp.bet[M]) /
                       (aa - a_n_prka * tmp.alf[M]))
        vv[M, j] = tmp.bet[-1]
        for i in range(M - 1, 0 - 1, -1):
            vv[i, j] = tmp.alf[i + 1] * vv[i + 1, j] + tmp.bet[i + 1]


cpdef Progonka_C_l_km1(double kk,
                       double[:, :] ff,  # n_dim, n_dim
                       double[:, :] vv,  # n_dim, n_dim
                       ProgonkaTmp tmp,
                       unsigned int n_dim,
                       unsigned int npq,
                       ):
    # TODO: move this allocation to colder code
    cdef np.ndarray[double, ndim=2] p = np.zeros((n_dim, n_dim))
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

        tmp.alf[1] = b_0_prka / aa
        for ii in range(nj1s + 1):
            j = ii * 2 * jks
            tmp.bet[1] = ff[0, j] / aa

            for i in range(1, M - 1 + 1):  # NOTE: 1
                tmp.alf[i + 1] = 1 / (aa - tmp.alf[i])
                tmp.bet[i + 1] = (ff[i, j] + tmp.bet[i]) * tmp.alf[i + 1]

            tmp.bet[-1] = ((ff[M, j] + a_n_prka * tmp.bet[M]) /
                           (aa - a_n_prka * tmp.alf[M]))
            vv[M, j] = tmp.bet[-1]
            ff[M, j] = vv[M, j]
            for i in range(M - 1, 0 - 1, -1):
                vv[i, j] = tmp.alf[i + 1] * vv[i + 1, j] + tmp.bet[i + 1]
                ff[i, j] = vv[i, j]


cpdef Progonka_C_l_km1_0N(double kk,
                          double[:, :] ff,  # n_dim, n_dim
                          double[:, :] vv,  # n_dim, n_dim
                          ProgonkaTmp tmp,
                          unsigned int n_dim,
                          ):
    cdef double aa, a_n_prka, b_0_prka
    cdef unsigned int i, j, M, l, i1, ii, nj1s, jks

    M = n_dim - 1
    a_n_prka = 2
    b_0_prka = 2
    jks = <unsigned int> 2 ** (kk - 1)

    for l in range(1, jks + 1):  # NOTE: 1
        aa = 2 * (1 + (1 - cos((2 * l - 1) * pi / (2 * jks))))
        tmp.alf[1] = b_0_prka / aa
        for j in range(0, M + 1, M):  # TODO: make it Cython-optimizable
            tmp.bet[1] = ff[0, j] / aa
            for i in range(1, M - 1 + 1):  # NOTE: 1
                tmp.alf[i + 1] = 1 / (aa - tmp.alf[i])
                tmp.bet[i + 1] = (ff[i, j] + tmp.bet[i]) * tmp.alf[i + 1]

            tmp.bet[-1] = ((ff[M, j] + a_n_prka * tmp.bet[M]) /
                           (aa - a_n_prka * tmp.alf[M]))
            vv[M, j] = tmp.bet[-1]
            ff[M, j] = vv[M, j]
            for i in range(M - 1, 0 - 1, -1):
                vv[i, j] = tmp.alf[i + 1] * vv[i + 1, j] + tmp.bet[i + 1]
                ff[i, j] = vv[i, j]


cpdef Progonka_C_l_km1_0N_Y(double kk,
                            double p_0N,
                            double[:, :] ff,  # n_dim, n_dim
                            double[:, :] vv,  # n_dim, n_dim
                            ProgonkaTmp tmp,
                            unsigned int n_dim,
                            ):
    cdef double aa, a_n_prka, b_0_prka, nj1s
    cdef unsigned int i, j, M, l, i1, ii, jks, jks2

    a_n_prka = 2
    b_0_prka = 2
    M = n_dim - 1
    jks = <unsigned int> (2 ** (kk - 1))
    jks2 = 2 * jks
    for l in range(1, jks2 - 1 + 1):  # NOTE: 1
        aa = 2 * (1 + (1 - cos(pi * l / jks)))
        tmp.alf[1] = b_0_prka / aa
        j = 0
        tmp.bet[1] = ff[0, j] / aa
        for i in range(1, M - 1 + 1):  # NOTE: 1
            tmp.alf[i + 1] = 1 / (aa - tmp.alf[i])
            tmp.bet[i + 1] = (ff[i, j] + tmp.bet[i]) * tmp.alf[i + 1]

        tmp.bet[-1] = ((ff[M, j] + a_n_prka * tmp.bet[M]) /
                       (aa - a_n_prka * tmp.alf[M]))
        vv[M, j] = tmp.bet[-1]
        ff[M, j] = vv[M, j]
        for i in range(M - 1, 0 - 1, -1):
            vv[i, j] = tmp.alf[i + 1] * vv[i + 1, j] + tmp.bet[i + 1]
            ff[i, j] = vv[i, j]

    l = jks2
    aa = 2 * (1 + (1 - cos(pi * l / jks)))
    tmp.alf[1] = b_0_prka / aa
    j = 0
    tmp.bet[1] = ff[0, j] / aa
    for i in range(1, M - 1 + 1):  # NOTE: 1
        tmp.alf[i + 1] = 1 / (aa - tmp.alf[i])
        tmp.bet[i + 1] = (ff[i, j] + tmp.bet[i]) * tmp.alf[i + 1]

    tmp.bet[-1] = -p_0N
    vv[M, j] = tmp.bet[-1]
    ff[M, j] = vv[M, j]
    for i in range(M - 1, 0 - 1, -1):
        vv[i, j] = tmp.alf[i + 1] * vv[i + 1, j] + tmp.bet[i + 1]
        ff[i, j] = vv[i, j]


cpdef Progonka_C_Y(double[:, :] ff,  # n_dim, n_dim
                   double[:, :] vv,  # n_dim, n_dim
                   ProgonkaTmp tmp,
                   unsigned int n_dim,
                   ):
    cdef double aa, a_n_prka, b_0_prka
    cdef unsigned int i, j, M
    a_n_prka = 2
    b_0_prka = 2
    M = n_dim - 1
    aa = 4

    tmp.alf[1] = b_0_prka / aa
    for j in range(1, M - 1 + 1, 2):
        tmp.bet[1] = ff[0, j] / aa
        for i in range(1, M - 1 + 1):  # NOTE: 1
            tmp.alf[i + 1] = 1 / (aa - tmp.alf[i])
            tmp.bet[i + 1] = (ff[i, j] + tmp.bet[i]) * tmp.alf[i + 1]

        tmp.bet[-1] = ((ff[M, j] + a_n_prka * tmp.bet[M]) /
                       (aa - a_n_prka * tmp.alf[M]))
        vv[M, j] = tmp.bet[-1]
        for i in range(M - 1, 0 - 1, -1):
            vv[i, j] = tmp.alf[i + 1] * vv[i + 1, j] + tmp.bet[i + 1]


cpdef Progonka_C_l_km1_Y(unsigned int kk,
                         double[:, :] ff,  # n_dim, n_dim
                         double[:, :] vv,  # n_dim, n_dim
                         ProgonkaTmp tmp,
                         unsigned int n_dim,
                         unsigned int npq,
                         ):
    cdef double aa, a_n_prka, b_0_prka
    cdef unsigned int i, j, M, l, ii, nks, jks
    a_n_prka = 2
    b_0_prka = 2

    M = n_dim - 1
    nks = <unsigned int> 2 ** (npq - kk + 1)
    jks = <unsigned int> 2 ** (kk - 1)

    for l in range(1, jks + 1):  # NOTE: 1
        aa = 2 * (1 + (1 - cos((2 * l - 1) * pi / (2 * jks))))
        tmp.alf[1] = b_0_prka / aa
        for ii in range(1, nks - 1 + 1, 2):
            j = ii * jks
            tmp.bet[1] = ff[0, j] / aa

            for i in range(1, M - 1 + 1):  # NOTE: 1
                tmp.alf[i + 1] = 1 / (aa - tmp.alf[i])
                tmp.bet[i + 1] = (ff[i, j] + tmp.bet[i]) * tmp.alf[i + 1]

            tmp.bet[-1] = ((ff[M, j] + a_n_prka * tmp.bet[M]) /
                           (aa - a_n_prka * tmp.alf[M]))
            vv[M, j] = tmp.bet[-1]
            ff[M, j] = vv[M, j]
            for i in range(M - 1, 0 - 1, -1):
                vv[i, j] = tmp.alf[i + 1] * vv[i + 1, j] + tmp.bet[i + 1]
                ff[i, j] = vv[i, j]


cpdef Neuman_red(double B_00,
                 double[:] r0,  # n_dim
                 double[:] r1,  # n_dim
                 double[:] rb,  # n_dim
                 double[:] ru,  # n_dim
                 double[:, :] q,  # n_dim, n_dim
                 double[:, :] YE,  # n_dim, n_dim
                 ProgonkaTmp tmp,
                 unsigned int n_dim,
                 double h,
                 unsigned int npq,
                 double x_max,
                 ):
    # TODO: move this allocation to colder code
    cdef np.ndarray[double, ndim=2] p = np.zeros((n_dim, n_dim))
    cdef np.ndarray[double, ndim=2] v = np.zeros((n_dim, n_dim))
    cdef np.ndarray[double, ndim=2] v1 = np.zeros((n_dim, n_dim))
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

    Progonka_C(v, v1, tmp, n_dim)
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

        Progonka_C_l_km1(k, v, v1, tmp, n_dim, npq)
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

    Progonka_C_l_km1_0N(k, v, v1, tmp, n_dim)
    for i in range(M + 1):
        p[i, 0] += v1[i, 0]
        q[i, 0] = 2 * q[i, jk] + 2 * p[i, 0]
        p[i, M] += v1[i, M]
        q[i, M] = 2 * q[i, M - jk] + 2 * p[i, M]

    # c  back way********* YN  Y0 ***************************
    p_M0 = p[M, 0]

    for i in range(M + 1):
        v[i, 0] = 2 * p[i, 0] + q[i, M]

    Progonka_C_l_km1_0N_Y(npq, p_M0, v, v1, tmp, n_dim)
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
        Progonka_C_l_km1_Y(k, v, v1, tmp, n_dim, npq)
        for ii in range(1, nk - 1 + 1, 2):  # NOTE: 1
            j = ii * jk
            for i in range(M + 1):
                YE[i, j] = p[i, j] + v1[i, j]
    k = 1
    for j in range(1, M - 1 + 1, 2):
        for i in range(M + 1):
            v[i, j] = YE[i, j - 1] + YE[i, j + 1] + q[i, j]
    Progonka_C_Y(v, v1, tmp, n_dim)
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
