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


import numpy as np

import lcode.configuration
import lcode.plasma.field_solver as field_solver


def make_diagonally_symmetric(a, T):
    upper = np.triu(a)
    lower = np.triu(a).copy().T
    np.fill_diagonal(lower, 0)
    if T == -1:
        np.fill_diagonal(upper, 0)
    return upper + lower * T


def gen_random_symmetric(N, x=None, y=None, T=None):
    a = np.random.uniform(0, 1 / 8, (N, N))
    N2 = N // 2
    if T:  # do not use w/o x and y
        d = make_diagonally_symmetric(a, T)
        a[...] = 0
        a[:N2+1, :N2+1] = d[:N2+1, :N2+1]
        check_symmetry(a, T=T, equal=np.array_equal)
    if x:
        a[:-N2-1:-1, :] = a[:N2, :] * x   # x is either 1 or -1
        if N % 2 and x == -1:
            a[N2, :] = 0
        check_symmetry(a, x=x, equal=np.array_equal)
    if y:
        a[:, :-N2-1:-1] = a[:, :N2] * y
        if N % 2 and y == -1:
            a[:, N2] = 0
        check_symmetry(a, y=y, equal=np.array_equal)
    check_symmetry(a, x=x, y=y, T=T, equal=np.array_equal)
    return a


def tuned_allclose(a, b):
    return np.allclose(a, b, rtol=0, atol=1e-14)


def check_symmetry(a, x=None, y=None, T=None, equal=tuned_allclose):
    if x:
        assert equal(a, a[::-1, :] * x)
    if y:
        assert equal(a, a[:, ::-1] * y)
    if T:
        assert equal(a, a.T * T)


def test_symmetry():
    npq = 4  # the error is sensitive to this
    N = 2**npq + 1

    roj_curr = np.empty((N, N), dtype=field_solver.RoJ_dtype)
    roj_prev = np.empty((N, N), dtype=field_solver.RoJ_dtype)
    beam_ro = gen_random_symmetric(N, x=1, y=1)
    roj_curr['ro'] = gen_random_symmetric(N, x=1, y=1)
    roj_prev['ro'] = gen_random_symmetric(N, x=1, y=1)
    roj_curr['jx'] = gen_random_symmetric(N, x=-1, y=1)
    roj_prev['jx'] = gen_random_symmetric(N, x=-1, y=1)
    roj_curr['jy'] = gen_random_symmetric(N, x=1, y=-1)
    roj_prev['jy'] = gen_random_symmetric(N, x=1, y=-1)
    roj_curr['jz'] = gen_random_symmetric(N, x=1, y=1)
    roj_prev['jz'] = gen_random_symmetric(N, x=1, y=1)
    roj_curr['jz'] = roj_curr['ro']
    roj_prev['jz'] = roj_prev['ro']
    Ex = gen_random_symmetric(N, x=-1, y=1)
    Ey = gen_random_symmetric(N, x=1, y=-1)
    Ez = gen_random_symmetric(N, x=1, y=1)
    Bx = gen_random_symmetric(N, x=1, y=-1)
    By = gen_random_symmetric(N, x=-1, y=1)
    Bz = np.zeros_like(Ex)
    out_Ex, out_Ey, out_Ez, out_Bx, out_By, out_Bz = [np.empty_like(Ex)
                                                      for _ in range(6)]

    # test pader_*
    dro_dx, dro_dy, dro_dxi = [np.empty_like(Ex) for _ in range(3)]
    field_solver.pader_x(roj_curr['ro'], dro_dx, 1, N)
    field_solver.pader_y(roj_curr['ro'], dro_dy, 1, N)
    field_solver.pader_xi(roj_prev['ro'], roj_curr['ro'], dro_dxi, 1, N)
    check_symmetry(dro_dx, x=-1, y=1, equal=np.array_equal)
    check_symmetry(dro_dy, x=1, y=-1, equal=np.array_equal)
    check_symmetry(dro_dxi, x=1, y=1, equal=np.array_equal)

    # test calculate_fields
    fs = field_solver.FieldSolver(N, 1)
    fs.calculate_fields(roj_curr, roj_prev, Ex, Ey, Ez, Bx, By, Bz, beam_ro,
                        1, npq, 1, 1, 0,
                        out_Ex, out_Ey, out_Ez, out_Bx, out_By, out_Bz,
                        False)

    check_symmetry(out_Ex, x=-1, y=1)
    check_symmetry(out_Ey, x=1, y=-1)
    check_symmetry(out_Ez, x=1, y=1)
    check_symmetry(out_Bx, x=1, y=-1)
    check_symmetry(out_By, x=-1, y=1)
    assert np.allclose(Bz, 0)
