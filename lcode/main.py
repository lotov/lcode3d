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


import inspect
import sys

import numpy as np

import hacks

from . import beam_construction
from . import beam_depositor
from . import beam_mover
from . import beam_particle
from . import configuration
from . import plasma_construction
from . import plasma_solver
from . import util


def main():
    config = sys.argv[1] if len(sys.argv) > 1 else None
    run(config)


def run(config=None):
    config = configuration.get(config)
    with hacks.use(*config.hacks):
        for t_i in range(config.time_steps):
            simulation_time_step(config, t_i)


# pylint: disable=too-many-statements
@hacks.friendly
def simulation_time_step(config=None, t_i=0):
    config = configuration.get(config)
    t = config.time_start + config.time_step_size * t_i
    shape = config.plasma_density_shape(t + config.time_step_size / 2)

    plasma = plasma_construction.construct(config.plasma, shape)
    plasma_cor = plasma.copy()
    plasma_solver_config = plasma_solver.PlasmaSolverConfig(config)

    beam_ro = np.zeros((config.grid_steps, config.grid_steps))
    beam_ro_curr = np.zeros_like(beam_ro)
    beam_ro_next = np.zeros_like(beam_ro)
    beam_ro_from_prev = np.zeros_like(beam_ro)
    roj = np.zeros((config.grid_steps, config.grid_steps),
                   dtype=plasma_solver.RoJ_dtype)
    roj_prev, roj_pprv = np.zeros_like(roj), np.zeros_like(roj)

    Ex = np.zeros((config.grid_steps, config.grid_steps))
    Phi_Ez = np.zeros_like(Ex)
    Ey, Ez = np.zeros_like(Ex), np.zeros_like(Ex)
    Bx, By, Bz = np.zeros_like(Ex), np.zeros_like(Ex), np.zeros_like(Ex)

    fell_from_prev_layer = np.zeros((0,), beam_particle.dtype)

    with choose_beam_source(config, t_i) as beam_source, \
         choose_beam_sink(config, t_i) as beam_sink:  # noqa: E127

        print('TIME', t, t_i, config.time_step_size)
        print('Beam data flows from', beam_source, 'to', beam_sink)

        for xi_i in range(config.xi_steps):
            xi = -config.xi_step_size * xi_i

            # Rotate and relabel arrays without copying data.
            roj_pprv, roj_prev, roj = roj_prev, roj, roj_pprv

            beam_layer = next(beam_source)
            beam_layer = np.append(beam_layer, fell_from_prev_layer)
            # TODO: come up with a better criterion
            active = beam_layer[
                beam_layer['t'] <= t + config.time_step_size / 2
            ]
            # TODO: deposit to TWO arrays, beam_ro_curr and beam_ro_prev
            # Maybe introduce a 'weights' parameter?..
            beam_xis = active['r'][:, 0]
            beam_ro_curr[...] = beam_depositor.deposit_beam(
                config, active, weights=(1 - (beam_xis - xi))
            )
            beam_ro_next[...] = beam_depositor.deposit_beam(
                config, active, weights=(beam_xis - xi)
            )
            beam_ro[...] = beam_ro_curr + beam_ro_from_prev

            # Only used when there is no input beam, only a beam ro function
            hacks.call.beam_ro_from_function_kludge(config, beam_source, xi_i,
                                                    beam_ro)

            if xi_i == 0:
                roj_prev['jz'] = beam_ro  # ???

            plasma_solver.response(plasma_solver_config, xi_i,
                                   plasma, plasma_cor, beam_ro,
                                   roj_pprv, roj_prev,
                                   mut_Ex=Ex, mut_Ey=Ey, mut_Ez=Ez,
                                   mut_Bx=Bx, mut_By=By, mut_Bz=Bz,
                                   out_plasma=plasma,
                                   out_plasma_cor=plasma_cor,
                                   out_roj=roj)

            import scipy.ndimage
            Phi_Ez += Ez * -config.xi_step_size
            xs, ys = plasma['x'].copy(), plasma['y'].copy()
            xs += config.window_width / 2
            xs *= config.grid_steps / config.window_width
            xs -= .5
            ys += config.window_width / 2
            ys *= config.grid_steps / config.window_width
            ys -= .5
            Phi_Ezs = scipy.ndimage.map_coordinates(Phi_Ez,  # noqa: F841
                                                    (xs, ys),
                                                    order=1)

            edges = np.linspace(-config.window_width / 2,
                                +config.window_width / 2,
                                config.grid_steps + 1)
            gamma = np.sqrt(1 + plasma['p'][:, 0]**2 +
                            plasma['p'][:, 1]**2 + plasma['p'][:, 2]**2)
            Sz = (  # noqa: F841
                Ex * By - Ey * Bx -
                (Ex**2 + Ey**2 + Ez**2 + Bx**2 + By**2 + Bz**2) / 2 +
                np.histogram2d(
                    plasma['x'], plasma['y'], edges,
                    weights=(gamma - 1) * (plasma['v'][:, 0] - 1)
                )[0]
            )

            moved, fell, lost, total_substeps = beam_mover.move(
                config, beam_layer, t, xi, Ex, Ey, Ez, Bx, By, Bz
            )
            beam_sink.put(moved)
            fell_from_prev_layer = fell

            beam_ro_from_prev[...] = beam_ro_next

            hacks.call.each_xi(config, t_i, xi_i, roj, plasma,
                               Ex, Ey, Ez, Bx, By, Bz)

            if config.print_every_xi_steps:
                if xi_i % config.print_every_xi_steps == 0:
                    print('xi=%.4f' % xi,
                          *[ex for ex in hacks.call.print_extra() if ex])


@hacks.friendly('lcode.main.choose_beam_source')
def choose_beam_source(config, t_i=0):
    # def beam(xi, x, y) case is hacked in from lcode.beam.ro_function
    # MPI data exchange case is hacked in from lcode.beam.mpi
    if t_i and config.beam_save:
        # Just take what was saved during the previous step
        if isinstance(config.beam_save, str):
            prev_filename = util.h5_filename(t_i, config.beam_save)
        else:
            prev_filename = util.h5_filename(t_i)
        return beam_construction.BeamFileSource(config, prev_filename)
    if not t_i and inspect.isgeneratorfunction(config.beam):
        return beam_construction.BeamConstructionSource(config, config.beam)
    elif isinstance(config.beam, str):
        filename = util.h5_filename(t_i, config.beam)
        return beam_construction.BeamFileSource(config, filename)
    else:
        raise RuntimeError('Unsupported beam definition')


@hacks.friendly('lcode.main.choose_beam_sink')
def choose_beam_sink(config, t_i=0):
    # def beam(xi, x, y) case is hacked in from lcode.beam.ro_function
    # MPI data exchange case is hacked in from lcode.beam.mpi
    # beam data archival case is hacked in from lcode.beam.archive
    if config.beam_save or config.time_steps:
        if isinstance(config.beam_save, str):
            filename = util.h5_filename(t_i + 1, config.beam_save)
        else:
            filename = util.h5_filename(t_i + 1)
        return beam_construction.BeamFileSink(config, filename)
    else:
        return beam_construction.BeamFakeSink()


if __name__ == '__main__':
    main()
