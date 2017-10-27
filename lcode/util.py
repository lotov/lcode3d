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


import datetime
import functools
import logging
import os
import sys
import tempfile

import hacks


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


class DebugLogging:  # pylint: disable=too-few-public-methods
    @hacks.after('lcode.main.configure_logging')
    def reconfigure_logging(self, _, *a, **kwa):
        logger = logging.getLogger('lcode')
        self.old_level = logger.level
        logger.setLevel(logging.DEBUG)

    def __on_exit__(self, _):
        logging.getLogger('lcode').setLevel(self.old_level)


class FileLogging:  # pylint: disable=too-few-public-methods
    @hacks.before('lcode.main.configure_logging')
    def reconfigure_logging(self, _, *a, **kwa):
        logger = logging.getLogger('lcode')
        self.file_handler = logging.FileHandler('lcode.log')
        self.file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(name)s %(levelname)s: %(message)s'))
        self.file_handler.setLevel(logging.DEBUG)
        logger.addHandler(self.file_handler)

    def __on_exit__(self, _):
        logging.getLogger('lcode').removeHandler(self.file_handler)


class FancyLogging:  # pylint: disable=too-few-public-methods
    @hacks.before('lcode.main.configure_logging')
    def reconfigure_logging(self, _, *a, **kwa):
        logger = logging.getLogger('lcode')
        try:
            import colorlog
            self.handler = colorlog.StreamHandler()
            self.handler.setFormatter(colorlog.ColoredFormatter(
                '%(log_color)s%(message)s'))
            logger.addHandler(self.handler)
            logger.setLevel(logging.INFO)
            return hacks.FakeResult(None)
        except ImportError:
            logger.warn('Install colorlog for fancy logging!')

    def __on_exit__(self, _):
        logger = logging.getLogger('lcode')
        try:
            logger.removeHandler(self.handler)
        except AttributeError:
            pass


class LcodeInfo:
    @staticmethod
    @hacks.after('lcode.main.configure_logging')
    def welcome(_, *a, **kwa):
        logger = logging.getLogger(__name__)
        logger.info('LCODE %s', LcodeInfo.get_lcode_version())
        for key, value in LcodeInfo.describe_environment():
            logger.debug('| %s: %s', key, value)

    @staticmethod
    def __on_exit__(_):
        logger = logging.getLogger(__name__)
        logger.debug('| Finished: %s', datetime.datetime.utcnow().isoformat())

    @staticmethod
    def get_lcode_version():
        try:
            logging.disable(logging.DEBUG)
            from pbr.version import VersionInfo
            lcode_version = VersionInfo('lcode').release_string()
            logging.disable(logging.NOTSET)
            return lcode_version
        except ImportError:
            return '<unknown>'

    @staticmethod
    def describe_environment():
        import platform
        import getpass
        try:
            import lcode
            lcode_path = os.path.abspath(lcode.__path__[0])
        except ImportError:
            lcode_path = '<unknown>'
        return (
            ('Working directory', os.getcwd()),
            ('Arguments', str(sys.argv)),
            ('LCODE path', lcode_path),
            ('Python', (platform.python_implementation() + ' ' +
                        platform.python_version())),
            ('Operating system', platform.platform()),
            ('User', getpass.getuser()),
            ('Node', platform.node()),
            ('Started', datetime.datetime.utcnow().isoformat())
        )
