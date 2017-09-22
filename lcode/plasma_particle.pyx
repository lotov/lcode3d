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


import inspect

import numpy as np
cimport numpy as np


USUAL_ELECTRON_CHARGE = -1
USUAL_ELECTRON_MASS = 1
USUAL_ION_CHARGE = 1
USUAL_ION_MASS = 1836.152674 * 85.4678


# TODO: macrosity


dtype = np.dtype([
    ('v', np.double, (3,)),
    ('p', np.double, (3,)),  # TODO: internal to move_particles, do not store
    ('N', np.long),
    ('x', np.double),
    ('y', np.double),
    ('q', np.double),
    ('m', np.double),
], align=False)


cpdef inline PlasmaParticle_t PlasmaParticle(
        double q=1, double m=1,
        double x=0, double y=0,
        double v_x=0, double v_y=0, double v_z=0,
        double p_x=0, double p_y=0, double p_z=0,
        long N=0):
    cdef PlasmaParticle_t p
    p.m, p.q = m, q
    p.x, p.y = x, y
    p.v[0], p.v[1], p.v[2] = v_z, v_x, v_y
    p.p[0], p.p[1], p.p[2] = p_z, p_x, p_y
    p.N = N
    return p


def PlasmaParticleArray(pythonized_plasma_particles):
    if isinstance(pythonized_plasma_particles, np.ndarray):
        if pythonized_plasma_particles.dtype == dtype:
            return pythonized_plasma_particles.copy()
    if inspect.isgeneratorfunction(pythonized_plasma_particles):
        pythonized_plasma_particles = pythonized_plasma_particles()

    pythonized_plasma_particles = list(pythonized_plasma_particles)
    cdef int l = len(pythonized_plasma_particles)
    cdef np.ndarray[t] plasma_particles
    plasma_particles = np.empty(len(pythonized_plasma_particles), dtype=dtype)
    for i in range(l):
        plasma_particles['q'][i] = pythonized_plasma_particles[i]['q']
        plasma_particles['m'][i] = pythonized_plasma_particles[i]['m']
        plasma_particles['x'][i] = pythonized_plasma_particles[i]['x']
        plasma_particles['y'][i] = pythonized_plasma_particles[i]['y']
        plasma_particles['v'][i] = pythonized_plasma_particles[i]['v']
        plasma_particles['p'][i] = pythonized_plasma_particles[i]['p']
        plasma_particles['N'][i] = pythonized_plasma_particles[i]['N']
    return plasma_particles


cpdef inline PlasmaParticle_t Electron(
        double q=USUAL_ELECTRON_CHARGE, double m=USUAL_ELECTRON_MASS,
        double x=0, double y=0,
        double v_x=0, double v_y=0, double v_z=0,
        double p_x=0, double p_y=0, double p_z=0,
        long N=0):
    return PlasmaParticle(q, m, x, y, v_x, v_y, v_z, p_x, p_y, p_z, N)


cpdef inline PlasmaParticle_t Ion(
        double q=USUAL_ION_CHARGE, double m=USUAL_ION_MASS,
        double x=0, double y=0,
        double v_x=0, double v_y=0, double v_z=0,
        double p_x=0, double p_y=0, double p_z=0,
        long N=0):
    return PlasmaParticle(q, m, x, y, v_x, v_y, v_z, p_x, p_y, p_z, N)
