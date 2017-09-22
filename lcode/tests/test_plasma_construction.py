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

import lcode.plasma_construction
import lcode.plasma_particle


def pythonized_list_variant():
    return [
        lcode.plasma_particle.Ion(q=2, v_x=1),
        lcode.plasma_particle.Ion(q=2, v_x=2),
        lcode.plasma_particle.Electron(q=-2, v_x=1),
        lcode.plasma_particle.Electron(q=-2, v_x=2),
    ]


def numpy_array_variant():
    return lcode.plasma_particle.PlasmaParticleArray(pythonized_list_variant())


def generator_function():
    yield lcode.plasma_particle.Ion(q=2, v_x=1)
    yield lcode.plasma_particle.Ion(q=2, v_x=2)
    yield lcode.plasma_particle.Electron(q=-2, v_x=1)
    yield lcode.plasma_particle.Electron(q=-2, v_x=2)


def generator_variant():
    return generator_function()


def generator_function_variant():
    return generator_function


def test_numpy_array_passthrough():
    var_np = numpy_array_variant()
    constructed = lcode.plasma_construction.construct(var_np)
    assert np.array_equal(constructed, var_np)


def test_pythonized_list_variant():
    var_np = lcode.plasma_construction.construct(numpy_array_variant())
    var_lp = lcode.plasma_construction.construct(pythonized_list_variant())
    assert np.array_equal(var_np, var_lp)


def test_generator_variant():
    var_np = lcode.plasma_construction.construct(numpy_array_variant())
    var_ge = lcode.plasma_construction.construct(generator_variant())
    assert np.array_equal(var_np, var_ge)


def test_generator_function_variant():
    var_np = lcode.plasma_construction.construct(numpy_array_variant())
    var_gf = lcode.plasma_construction.construct(generator_function_variant())
    assert np.array_equal(var_np, var_gf)
