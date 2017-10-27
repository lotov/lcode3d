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


import logging

from mpi4py import MPI

import hacks


class BeamMPISource:  # pylint: disable=too-few-public-methods
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = MPI.COMM_WORLD.Get_size()
        assert self.rank != 0
        self.src = self.rank - 1

    def __enter__(self):
        return self

    def __next__(self):
        return self.comm.recv(source=self.src)  # may be optimized

    def __iter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __repr__(self):
        return ('<' + self.__class__.__name__ + ': ' +
                '{rank}/{size}'.format(rank=self.rank, size=self.size) + '>')


class BeamMPISink:  # pylint: disable=too-few-public-methods
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = MPI.COMM_WORLD.Get_size()
        assert self.rank != self.size - 1
        self.dst = self.rank + 1

    def __enter__(self):
        return self

    def put(self, layer):
        self.comm.send(layer, dest=self.dst)  # may be optimized

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __repr__(self):
        return ('<' + self.__class__.__name__ + ': ' +
                '{rank}/{size}'.format(rank=self.rank, size=self.size) + '>')


class MPIPowers:
    @hacks.before('simulation_time_step')
    def init(self, simulation_time_step, config, t_i=0):
        logger = logging.getLogger(__name__)
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.size = MPI.COMM_WORLD.Get_size()
        last_round_working_odd = config.time_steps % self.size
        if t_i % self.size != self.rank:
            if t_i == config.time_steps - 1 and last_round_working_odd:
                if self.rank >= last_round_working_odd:
                    logger.debug('MPI Barrier (idle %d)', self.rank)
                    MPI.COMM_WORLD.Barrier()  # wait for last round
            return hacks.FakeResult('Not my time step!')

    @hacks.before('lcode.main.choose_beam_source')
    def hijack_choose_beam_source(self, choose_beam_source, config, t_i=0):
        if self.rank > 0 and t_i > 0:
            return hacks.FakeResult(BeamMPISource())

    @hacks.before('lcode.main.choose_beam_sink')
    def hijack_choose_beam_sink(self, choose_beam_sink, config, t_i=0):
        if self.rank < self.size - 1 and t_i < config.time_steps - 1:
            return hacks.FakeResult(BeamMPISink())

    # TODO: here or where?
    @hacks.after('simulation_time_step')
    def barrier_each_t(self, retval, *a, **kwa):
        logger = logging.getLogger(__name__)
        MPI.COMM_WORLD.Barrier()
        if self.rank == self.size - 1:
            logger.debug('MPI Barrier (working %d)', self.rank)

    @hacks.into('each_xi')
    def barrier_each_xi(self, simulation_time_step, *a, xi=None, **kwa):
        # TODO: progress bar for a race
        pass

    @hacks.into('print_extra')
    def print_extra_rank(self, *a):
        return 'MPI:{rank}/{size}'.format(rank=self.rank, size=self.size)


# TODO: possibly optimize with user-defined MPI datatypes.
# https://groups.google.com/forum/#!topic/mpi4py/pHA_s7fS0q0
