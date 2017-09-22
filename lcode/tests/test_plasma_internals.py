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
import lcode.plasma_solver


def test_Progonka_Dirichlet():
    config = lcode.plasma_solver.PlasmaSolverConfig(
        lcode.configuration.get('grid_steps = 2**2 + 1')
    )
    tmp = lcode.plasma_solver.ProgonkaTmp(config)

    ff = np.array([-1, 3, 2, 3, -1], np.double)
    vv_correct = np.array([0, 1, 1, 1, 0], np.double)

    vv = np.empty_like(vv_correct)
    lcode.plasma_solver.Progonka_Dirichlet(4, ff, vv, tmp)

    assert np.array_equal(vv_correct, vv)


def test_reduction_Dirichlet1():
    config = lcode.plasma_solver.PlasmaSolverConfig(
        lcode.configuration.get('grid_steps = 2**10 + 1; window_width = 1')
    )
    tmp = lcode.plasma_solver.ProgonkaTmp(config)
    n_dim, Lx = config.n_dim, config.x_max * 2
    config.h = Lx / (n_dim - 1)  # values are not grid-centered in this test

    x = np.linspace(0, Lx, n_dim)[:, None]
    y = np.linspace(0, Lx, n_dim)[None, :]
    Fi = -2 * y * (y - Lx) - 2 * x * (x - Lx)
    P_correct = x * (x - Lx) * y * (y - Lx)

    P = np.empty_like(P_correct)
    lcode.plasma_solver.reduction_Dirichlet1(config, Fi, P, tmp)

    assert np.allclose(P_correct, P, rtol=1e-20, atol=1e-12)


def test_reduction_Neuman_red():
    n_dim = 2**9 + 1
    Lx = 2
    B_0 = 0
    c = lcode.configuration.get({
        'grid_steps': n_dim,
        'window_width': Lx,
        'plasma_solver_B_0': B_0,
    })
    config = lcode.plasma_solver.PlasmaSolverConfig(c)
    tmp = lcode.plasma_solver.ProgonkaTmp(config)
    config.h = Lx / (n_dim - 1)  # values on edges in the test check
    P = np.zeros((n_dim, n_dim))

    x = np.linspace(-Lx / 2, Lx / 2, n_dim)  # 1D x
    y = np.linspace(-Lx / 2, Lx / 2, n_dim)  # 1D y

    r0 = 3 - 2 - 2 * y**2
    r1 = 3 - 2 + 2 * y**2
    rb = -2 - 2 * x**2
    ru = +2 + 2 * x**2

    x = x[:, None]  # 2D x
    y = y[None, :]  # 2D y
    Fi = 6 * x + 2 + 2 * y**2 + 2 * x**2
    P_correct = x**3 + y**2 - 2*x + x**2 * y**2

    lcode.plasma_solver.Neuman_red(config, B_0,
                                   -r0, r1, -rb, ru,
                                   -Fi, P,
                                   tmp)

    P_correct -= np.average(P_correct)

    print(P)
    print(P_correct)

    print(np.max(np.absolute(P_correct - P)))
    assert np.all(np.absolute(P_correct - P) < 1e-4)
