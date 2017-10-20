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


import hacks


def mln(d):
    return sum((len(arr) for arr in d.values()))


@hacks.into('print_extra')
@hacks.stealing
def PrintBeamInfo(*a, beam_layer=hacks.steal, total_substeps=hacks.steal,
                  moved=hacks.steal, fell=hacks.steal, lost=hacks.steal):
    if not mln(beam_layer):
        return
    return ' '.join((
        'N=%d=%d+%d+%d' % (mln(beam_layer), mln(moved), mln(fell), mln(lost)),
        's=%.3f' % (total_substeps / mln(beam_layer)),
    ))
