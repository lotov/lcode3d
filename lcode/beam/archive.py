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

import lcode.beam_construction
import lcode.util


class BeamDuplicatingSink:  # pylint: disable=too-few-public-methods
    def __init__(self, sink1, sink2):
        self.sink1, self.sink2 = sink1, sink2

    def __enter__(self):
        self.sink1.__enter__()
        self.sink2.__enter__()
        return self

    def put(self, layer):
        self.sink1.put(layer)
        self.sink2.put(layer)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sink2.__exit__(exc_type, exc_val, exc_tb)
        self.sink1.__exit__(exc_type, exc_val, exc_tb)

    def __repr__(self):
        return '<%s: %s+%s>' % (self.__class__.__name__,
                                repr(self.sink1),
                                repr(self.sink2))


class BeamArchive:
    '''
    Archives the beam to the separate file when config.archive(t, t_i) is True
    or config.archive is a literal True or a template string.
    Archive filename will be taken from config.archive(t, t_i) or autoguessed.
    It is implemented by hijacking choose_beam_sink and setting it to
    a BeamDuplicatingSink instance that puts to both the original sink
    and an extra BeamFileSink.
    '''

    # TODO: feed parameters to config.archive based on inspected signature,
    #       presumably in a highly unified manner...

    @staticmethod
    @hacks.after('lcode.main.choose_beam_sink')
    def hijack_choose_beam_source(retval, config, t_i=0):
        original_sink = retval
        t = config.time_start + config.time_step_size * t_i
        if BeamArchive.should_archive(config, t, t_i):
            filename = BeamArchive.guess_filename(config, t, t_i)
            archive_sink = lcode.beam_construction.BeamFileSink(config,
                                                                filename)
            duplicating_sink = BeamDuplicatingSink(original_sink, archive_sink)
            return hacks.FakeResult(duplicating_sink)

    @staticmethod
    def should_archive(config, t, t_i):
        if 'archive' not in dir(config):
            return False
        return bool(BeamArchive.eval_archive(config, t, t_i))

    @staticmethod
    def eval_archive(config, t, t_i):
        assert 'archive' in dir(config)
        if config.archive is True or isinstance(config.archive, str):
            return config.archive
        # Assume it's a function of t and t_i for now
        return config.archive(t, t_i)

    @staticmethod
    def guess_filename(config, t, t_i):
        assert 'archive' in dir(config)
        a = BeamArchive.eval_archive(config, t, t_i)
        template = a if isinstance(a, str) else 'archive_{time_i}.h5'
        return lcode.util.h5_filename(t_i + 1, template)
