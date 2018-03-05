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



cdef class FieldSolver:
    cdef int n_dim, threads
    cdef object tlss

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
