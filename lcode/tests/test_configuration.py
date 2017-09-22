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

import lcode.configuration


def test_from_dict():
    c = lcode.configuration.get({'a': 4, 'b': 3})
    assert c.a == 4
    assert c.b == 3


def test_from_string():
    c = lcode.configuration.get('a=4;b=a-1')
    assert c.a == 4
    assert c.b == 3
