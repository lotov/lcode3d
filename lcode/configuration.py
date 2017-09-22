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


import imp
import importlib


USE_LCODE_DEFAULT = object()  # sentinel


# pylint: disable=no-else-return
def get(something=None, default=USE_LCODE_DEFAULT):
    if default == USE_LCODE_DEFAULT:
        default = load_default_lcode_config()
    if something is None:
        return default
    elif isinstance(something, str):
        if '\n' in something or '=' in something or something == '':
            # Probably configuration data, execute it
            return from_string(something, default=default)
        else:
            # Probably a configuration file path, read and execute it
            return from_filename(something, default=default)
    elif isinstance(something, dict):
        config = default  # No copying as default is unique every time
        config.__dict__.update(something)
        return config
    else:
        return something  # Maybe it's already good enough, let's try


def from_filename(filename, default=None):
    with open(filename) as config_file:
        contents = config_file.read()
    return from_string(contents, filename=filename, default=default)


def from_string(config_string, filename='<string>', default=None):
    code = compile(config_string, filename, 'exec')
    config = imp.new_module('config')
    # config.__dict__.update(default)  # That would allow using defvals
    exec(code, config.__dict__)
    # Let's inject default values after the config execution
    if default:
        for k in default.__dict__:
            if k not in config.__dict__:
                config.__dict__[k] = default.__dict__[k]
    return config


def load_default_lcode_config():
    # TODO: obtain filename without imnporting
    from . import default_config
    try:
        # The should be the safest method:
        # execute the default config file contents
        # in a separate imp.new_module,
        # bypassing usual import machinery and caching:
        return from_filename(default_config.__file__)
    except IOError:
        # Unfortunately, default_config.__file__ can be undefined
        # and unreacheable. Let's try to use importlib.reload() in this case:
        print('WARNING: Reloading default config instead of reexecuting it')
        importlib.reload(default_config)
        return default_config
