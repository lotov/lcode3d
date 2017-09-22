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


import functools
import os
import tempfile


def in_temp_dir(*a, **kwa):
    def in_temp_dir_decorator(func):
        @functools.wraps(func)
        def in_temp_dir_decorated_func(*func_a, **func_kwa):
            prevdir = os.getcwd()
            with tempfile.TemporaryDirectory(*a, **kwa) as tmpdir:
                os.chdir(tmpdir)
                try:
                    func(*func_a, **func_kwa)
                finally:
                    os.chdir(prevdir)
        return in_temp_dir_decorated_func
    return in_temp_dir_decorator


def fmt_time(t_i):
    return '%05d' % t_i


def h5_filename(t_i, template='{time_i}.h5'):
    return template.format(time_i=fmt_time(t_i))
