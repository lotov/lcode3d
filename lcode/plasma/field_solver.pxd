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


import numpy as np
cimport numpy as np


# RoJ for both scalar charge density ro and vector current j, TODO: get rid of
cdef packed struct RoJ_t:
    double ro
    double jz
    double jx
    double jy


cdef class ThreadLocalStorage:
    # two very popular temporary arrays
    cdef double[:] alf  # n_dim
    cdef double[:] bet  # n_dim + 1
    # temporary arrays for Posson_reduct_12 and reduction_Dirichlet1
    cdef double[:] RedFi  # n_dim
    cdef double[:] Svl  # n_dim
    cdef double[:] PrF  # n_dim
    cdef double[:] PrV  # n_dim
    cdef double[:] Psi  # n_dim
    cdef double[:, :] p  # n_dim, n_dim
    cdef double[:, :] v  # n_dim, n_dim
    cdef double[:, :] v1  # n_dim, n_dim
    cdef double[:, :] p_km1  # n_dim, n_dim
    cdef double[:, :] rhs  # n_dim, n_dim
    cdef double[:, :] grad1  # n_dim, n_dim
    cdef double[:, :] grad2  # n_dim, n_dim

    cdef double[:] Ez_a  # n_dim - 1
    cdef double[:, :] Ez_bet  # n_dim - 1, n_dim
    cdef double[:, :] Ez_alf  # n_dim - 1, n_dim
    cdef double[:, :] Ez_PrFi  # n_dim - 1, n_dim - 1
    cdef double[:, :] Ez_P  # n_dim - 1, n_dim - 1


cdef class MixedSolver:
    cdef public int N
    cdef public bint subtraction_trick
    cdef double h, mul
    # TODO: tune C/F layout, specify with ::1?
    cdef double[:, :] alf
    cdef double[:, :] bet
    cdef double[:, :] rhs_fixed
    cdef double[:, :] tmp1
    cdef double[:, :] tmp2
    cpdef solve(MixedSolver self, double[:, :] rhs, double[:] bound_top, double[:] bound_bot, double[:, :] out)


cdef class DirichletSolver:
    cdef public int N
    cdef double h, mul
    # TODO: tune C/F layout, specify with ::1?
    cdef double[:, :] alf
    cdef double[:, :] bet
    cdef double[:, :] tmp1
    cdef double[:, :] tmp2
    cpdef solve(DirichletSolver self, double[:, :] rhs, double[:, :] out)


cdef class FieldSolver:
    cdef int n_dim, threads
    cdef ThreadLocalStorage tls_0, tls_1, tls_2, tls_3, tls_4, tls_5
    cdef DirichletSolver ds_Ez
    cdef MixedSolver mxs_Ex, mxs_Ey, mxs_Bx, mxs_By

    cpdef calculate_fields(self,
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
                           bint variant_A)
