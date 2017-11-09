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

import lcode.configuration


def test_from_dict():
    c = lcode.configuration.get({'a': 4, 'b': 3})
    assert c.a == 4
    assert c.b == 3


def test_from_string():
    c = lcode.configuration.get('a=4;b=a-1')
    assert c.a == 4
    assert c.b == 3


CONFIG_TIME_DEPENDENCE = '''
time_step_size = 3
def variable_t_i(t_i):
    return t_i * 4
def variable_t(t):
    return t**2
'''


def test_time_dependence():
    @hacks.friendly('simulation_time_step')
    def fake_simulation_time_step(config, t_i):
        c = lcode.configuration.get(config, t_i)
        assert c.variable_t_i == t_i * 4
        assert c.variable_t == (t_i * 3)**2

    with hacks.use(lcode.configuration.TimeDependence):
        for t_i in range(2, 4):
            fake_simulation_time_step(CONFIG_TIME_DEPENDENCE, t_i)


def test_templating():
    c = lcode.configuration.get('variable = ${t_i * 4}', t_i=2)
    assert c.variable == 8
    assert c.__source__ == 'variable = ${t_i * 4}'
    assert c.__source_templated__ == 'variable = 8'


CONFIG_COMBINED = '''
time_step_size = 3
def variable_t_i(t_i):
    return t_i * 4
def variable_t(t):
    return t**2
<% t = t_i * 3 %>
combined = ${t}, ${t_i}
'''


def test_combined():
    @hacks.friendly('simulation_time_step')
    def fake_simulation_time_step(config, t_i):
        c = lcode.configuration.get(config, t_i)
        assert c.variable_t_i == t_i * 4
        assert c.variable_t == (t_i * 3)**2
        assert c.combined == (t_i * 3, t_i)

    with hacks.use(lcode.configuration.TimeDependence):
        for t_i in range(2, 4):
            fake_simulation_time_step(CONFIG_COMBINED, t_i)
