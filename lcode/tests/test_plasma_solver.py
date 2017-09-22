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
import lcode.plasma_particle


CONFIG = """
grid_steps = 2**3 + 1
plasma_solver_corrector_passes = 0
# TODO: reenable
"""


def test_single_particle_response():
    config = lcode.configuration.get(CONFIG)
    plasma_solver_config = lcode.plasma_solver.PlasmaSolverConfig(config)

    Ex, Ey, Ez, Bx, By, Bz = [np.zeros((config.grid_steps, config.grid_steps))
                              for _ in range(6)]
    beam_ro = np.zeros((config.grid_steps, config.grid_steps))
    roj = np.zeros((config.grid_steps, config.grid_steps),
                   dtype=lcode.plasma_solver.RoJ_dtype)
    roj_prev, roj_pprv = np.zeros_like(roj), np.zeros_like(roj)

    the_only_electron = lcode.plasma_particle.Electron(x=0, y=0)
    plasma = lcode.plasma_particle.PlasmaParticleArray([the_only_electron])
    plasma_cor = np.zeros_like(plasma)

    lcode.plasma_solver.response(plasma_solver_config, 0,
                                 plasma, plasma_cor, beam_ro,
                                 roj_pprv, roj_prev,
                                 mut_Ex=Ex, mut_Ey=Ey, mut_Ez=Ez,
                                 mut_Bx=Bx, mut_By=By, mut_Bz=Bz,
                                 out_plasma=plasma,
                                 out_plasma_cor=plasma_cor,
                                 out_roj=roj)

    def symmetrical_x(arr): return np.allclose(arr, arr[::-1, :])

    def symmetrical_y(arr): return np.allclose(arr, arr[:, ::-1])

    def symmetrical_xy(arr): return symmetrical_x(arr) and symmetrical_y(arr)

    def symmetrical_all(arr): return (symmetrical_xy(arr) and
                                      np.allclose(arr, arr.T))

    def center(array):
        return array[array.shape[0] // 2, array.shape[1] // 2]

    # correct
    initial = lcode.plasma_particle.PlasmaParticleArray([the_only_electron])
    assert plasma == initial

    assert symmetrical_all(Ez)
    assert center(Ez) == 0
    assert np.array_equal(Ex, Ey.T)

    assert symmetrical_all(roj['ro'])
    assert np.allclose(roj['jx'], 0)
    assert np.allclose(roj['jy'], 0)
    assert np.allclose(roj['jz'], 0)
    assert symmetrical_xy(abs(Ex))
    assert symmetrical_xy(abs(Ey))
    assert np.allclose(Ex[config.grid_steps // 2, config.grid_steps // 2], 0)
    assert np.allclose(Ey[config.grid_steps // 2, config.grid_steps // 2], 0)

    # currently false
    # assert symmetrical_all(abs(Ex))
    # assert symmetrical_all(abs(Ey))
