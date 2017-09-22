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


from mpi4py import MPI
import numpy as np

import hacks

import lcode.beam_construction
import lcode.beam_particle


class BeamMPISource:  # pylint: disable=too-few-public-methods
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = MPI.COMM_WORLD.Get_size()
        assert self.rank != 0
        self.src = self.rank - 1

    def __enter__(self):
        return self

    def __next__(self):  # pylint: disable=no-self-use
        l = self.comm.recv(source=self.src)
        layer = np.zeros(l, dtype=lcode.beam_particle.dtype)
        self.recv_array(layer['r'])
        self.recv_array(layer['p'])
        self.recv_array(layer['N'])
        self.recv_array(layer['m'])
        self.recv_array(layer['q'])
        self.recv_array(layer['W'])
        self.recv_array(layer['t'])
        return layer

    def recv_array(self, to):
        tmp = to.copy()
        self.comm.Recv(tmp, source=self.src)
        to[...] = tmp

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
        self.comm.send(layer.shape[0], dest=self.dst)
        self.send_array(layer['r'])
        self.send_array(layer['p'])
        self.send_array(layer['N'])
        self.send_array(layer['m'])
        self.send_array(layer['q'])
        self.send_array(layer['W'])
        self.send_array(layer['t'])

    def send_array(self, arr):
        self.comm.Send(arr.copy(), dest=self.dst)

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __repr__(self):
        return ('<' + self.__class__.__name__ + ': ' +
                '{rank}/{size}'.format(rank=self.rank, size=self.size) + '>')


class MPIPowers:
    @hacks.before('simulation_time_step')
    def init(self, simulation_time_step, config, t_i=0):
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.size = MPI.COMM_WORLD.Get_size()
        if t_i % self.size != self.rank:
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
        MPI.COMM_WORLD.Barrier()
        if self.rank == self.size - 1:
            print('#' * 80)

    @hacks.into('each_xi')
    def barrier_each_xi(self, simulation_time_step, *a, xi=None, **kwa):
        # TODO: progress bar for a race
        pass

    @hacks.into('print_extra')
    def print_extra_rank(self, *a):
        return 'MPI:{rank}/{size}'.format(rank=self.rank, size=self.size)
