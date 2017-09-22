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


cdef packed struct PlasmaParticle_t:
    double v[3]
    double p[3]  # TODO: internal to move_particles, do not store
    long N  # [1]
    double x
    double y
    double q
    double m

ctypedef PlasmaParticle_t t

# [1] Cython breaks when a struct has consecutive fields
#     of same type and different dimension.
#     see https://github.com/cython/cython/issues/1407
