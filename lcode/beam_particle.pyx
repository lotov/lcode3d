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


# cython: language_level=3


"""
An object that represents a beam particle.
Lacks behaviour, stores data in a C struct
that sometimes poses as a convenient numpy dtype.
"""


import numpy as np
cimport numpy as np


dtype = np.dtype([
    ('r', np.double, (3,)),
    ('p', np.double, (3,)),
    ('N', np.long),
    ('m', np.double),
    ('q', np.double),
    ('W', np.double),
    ('t', np.double),
], align=False)


cpdef inline BeamParticle_t BeamParticle(
        double m=0, double q=0, double W=1, long N=0, double t=0,
        double xi=0, double x=0, double y=0,
        double p_xi=0, double p_x=0, double p_y=0):
    cdef BeamParticle_t b
    b.m, b.q, b.W, b.N, b.t = m, q, W, N, t
    b.r[0], b.r[1], b.r[2] = xi, x, y
    b.p[0], b.p[1], b.p[2] = p_xi, p_x, p_y
    return b


def BeamParticleArray(pythonized_beam_particles):
    """
    Create an numpy array if bodies from an iterable of dict-like particles.
    When BeamParticle surfaces from Cython-land to Python-land it gets
    pythonized-autoconverted into a dict-like onject.
    This function should pack such poor objects into a numpy array.
    """
    pythonized_beam_particles = list(pythonized_beam_particles)
    beam_particles_array = np.empty(len(pythonized_beam_particles),
                                    dtype=dtype)
    for i, b in enumerate(pythonized_beam_particles):
        beam_particles_array[i]['m'] = b['m']
        beam_particles_array[i]['q'] = b['q']
        beam_particles_array[i]['W'] = b['W']
        beam_particles_array[i]['N'] = b['N']
        beam_particles_array[i]['t'] = b['t']
        beam_particles_array[i]['r'] = b['r']
        beam_particles_array[i]['p'] = b['p']
    return beam_particles_array
