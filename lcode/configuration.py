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
import inspect

import mako.template
import mako.lookup

import hacks


USE_LCODE_DEFAULT = object()  # sentinel


def get(something=None, default=USE_LCODE_DEFAULT, t_i=0):
    if something is None:
        if default == USE_LCODE_DEFAULT:
            default = load_default_lcode_config()
        return default
    if isinstance(something, str):
        if default == USE_LCODE_DEFAULT:
            default = load_default_lcode_config()
        if '\n' in something or '=' in something or something == '':
            # Probably configuration data, execute it
            return from_string(something, default=default, t_i=t_i)
        # Probably a configuration file path, read and execute it
        return from_filename(something, default=default, t_i=t_i)
    if isinstance(something, dict):
        if default == USE_LCODE_DEFAULT:
            default = load_default_lcode_config()
        config = default  # No copying as default is unique every time
        config.__dict__.update(something)
        return config
    return something  # Maybe it's already good enough, let's try


def from_filename(filename, default=None, t_i=0):
    with open(filename) as config_file:
        contents = config_file.read()
    return from_string(contents, filename=filename, default=default, t_i=t_i)


def from_string(config_string, filename='<string>', default=None, t_i=0):
    lookup = mako.lookup.TemplateLookup(directories=['.'])
    template = mako.template.Template(config_string, lookup=lookup)
    config_string_templated = template.render(t_i=t_i)

    code = compile(config_string_templated, filename, 'exec')
    config = imp.new_module('config')
    # config.__dict__.update(default)  # That would allow using defvals
    exec(code, config.__dict__)

    config.__source__ = config_string
    config.__source_templated__ = config_string_templated

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
        importlib.reload(default_config)
        return default_config


class TimeDependence:  # pylint: disable=too-few-public-methods
    @staticmethod
    @hacks.around('simulation_time_step')
    def postprocess_config(simulation_time_step):
        def time_parametrized_simulation_time_step(config=None, t_i=0):
            '''Call simulation_time_step expanding functions of t and t_i.'''
            config = get(config)
            t = config.time_start + config.time_step_size * t_i
            # a really shallow copy of the config
            expanded_config = imp.new_module(config.__name__)
            expanded_config.__dict__.update(config.__dict__)
            for attrname in dir(config):
                if inspect.isroutine(getattr(config, attrname)):
                    func = getattr(config, attrname)
                    params = list(inspect.signature(func).parameters)
                    if params == ['t']:
                        setattr(expanded_config, attrname, func(t))
                    elif params == ['t_i']:
                        setattr(expanded_config, attrname, func(t_i))
            return simulation_time_step(expanded_config, t_i)
        return time_parametrized_simulation_time_step
