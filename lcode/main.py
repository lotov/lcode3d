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
import logging
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
    run(config, configure_logging=True)


def run(config=None, configure_logging=False):
    config_pre = configuration.get(config)
    with hacks.use(*config_pre.hacks):
        if configure_logging:
            _configure_logging()
        for t_i in range(config_pre.time_steps):
            simulation_time_step(config, t_i)


@hacks.friendly('lcode.main.configure_logging')
def _configure_logging():
    root_lcode_logger = logging.getLogger('lcode')
    root_lcode_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    root_lcode_logger.addHandler(handler)


# pylint: disable=too-many-statements
@hacks.friendly
def simulation_time_step(config=None, t_i=0):
    config = configuration.get(config, t_i=t_i)
    logger = logging.getLogger(__name__)
    t = config.time_start + config.time_step_size * t_i

    plasma = plasma_construction.construct(config.plasma,
                                           config.plasma_density_shape)
    plasma_cor = plasma.copy()
    if config.plasma_solver == 'v2':
        from .plasma import solver_v2 as plasma_solver
    elif config.plasma_solver == 'v2_monolithic':
        from .plasma import solver_v2_monolithic as plasma_solver
        plasma_solver = plasma_solver.PlasmaSolver(config)
    elif config.plasma_solver == 'v1':
        from . import plasma_solver
    else:
        plasma_solver = config.plasma_solver
    plasma_solver_config = plasma_solver.PlasmaSolverConfig(config)

    beam_ro = np.zeros((config.grid_steps, config.grid_steps))
    beam_ro_next = np.zeros_like(beam_ro)
    beam_ro_from_prev = np.zeros_like(beam_ro)
    roj = np.zeros((config.grid_steps, config.grid_steps),
                   dtype=plasma_solver.RoJ_dtype)
    roj_prev, roj_pprv = np.zeros_like(roj), np.zeros_like(roj)

    Ex = np.zeros((config.grid_steps, config.grid_steps))
    Phi_Ez = np.zeros_like(Ex)
    Ey, Ez = np.zeros_like(Ex), np.zeros_like(Ex)
    Bx, By, Bz = np.zeros_like(Ex), np.zeros_like(Ex), np.zeros_like(Ex)
    Ez[...], Ex[...], Ey[...] = config.Ez, config.Ex, config.Ey
    Bz[...], Bx[...], By[...] = config.Bz, config.Bx, config.By

    fell_from_prev_layer = np.zeros((0,), beam_particle.dtype)

    with choose_beam_source(config, t_i) as beam_source, \
         choose_beam_sink(config, t_i) as beam_sink:  # noqa: E127

        logger.info('Step %d, t = %f', t_i, t)
        logger.debug('Beam data flows from %s', beam_source)
        logger.debug('Beam data flows to %s', beam_sink)

        for xi_i in range(config.xi_steps):
            xi = -config.xi_step_size * xi_i

            # Rotate and relabel arrays without copying data.
            roj_pprv, roj_prev, roj = roj_prev, roj, roj_pprv

            beam_layer = next(beam_source)
            beam_layer = beam_append(beam_layer, fell_from_prev_layer)

            deposit_beam(config, t, xi, beam_layer,
                         beam_ro_from_prev,
                         out_beam_ro_curr=beam_ro,
                         out_beam_ro_next=beam_ro_next)

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

            moved, fell, lost, total_substeps = beam_move(
                config, beam_layer, t, xi, Ex, Ey, Ez, Bx, By, Bz
            )
            beam_sink.put(moved)
            fell_from_prev_layer = fell

            # TODO: come up with better array names, skip this copying
            beam_ro_from_prev[...] = beam_ro_next

            hacks.call.each_xi(config, t_i, xi_i, roj, plasma,
                               Ex, Ey, Ez, Bx, By, Bz)

            sys.stdout.flush()
            if config.print_every_xi_steps:
                if xi_i % config.print_every_xi_steps == 0:
                    logger.info('xi=%.4f ' % xi + ' '.join(
                        [ex for ex in hacks.call.print_extra() if ex]
                    ))


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
    if not t_i and isinstance(config.beam, dict):
        # FIXME: for now it's a catchall
        # The whole beam data architecture should be reworked for flexibility
        return beam_construction.BeamConstructionSource(config, config.beam)
    elif isinstance(config.beam, str):
        filename = util.h5_filename(t_i, config.beam)
        return beam_construction.BeamFileSource(config, filename)
    else:
        raise RuntimeError('Unsupported beam definition')


# pylint: disable=too-many-arguments
@hacks.friendly('lcode.main.deposit_beam')
def deposit_beam(config, t, xi, beam_layer,
                 beam_ro_from_prev, out_beam_ro_curr, out_beam_ro_next):
    out_beam_ro_curr[...] = beam_ro_from_prev
    out_beam_ro_next[...] = 0

    if isinstance(beam_layer, dict):
        species = beam_layer
    else:
        species = {'particles': beam_layer}

    for name in sorted(species):
        particles_layer = species[name]
        assert isinstance(particles_layer, np.ndarray)
        assert particles_layer.dtype == beam_particle.dtype
        # TODO: come up with a better criterion

        if config.time_step_size:
            active = particles_layer[
                particles_layer['t'] <= t + config.time_step_size / 2
            ]
        else:
            active = particles_layer

        beam_xis = active['r'][:, 0]
        weights_next = (-beam_xis - -xi) / config.xi_step_size
        weights_curr = 1 - weights_next
        out_beam_ro_curr += beam_depositor.deposit_beam(config, active,
                                                        weights_curr)
        out_beam_ro_next += beam_depositor.deposit_beam(config, active,
                                                        weights_next)


def beam_append(beam_layer, fell_from_prev_layer):
    res = {sp_name: l.copy() for sp_name, l in beam_layer.items()}
    for sp_name in fell_from_prev_layer:
        assert sp_name in beam_layer, 'fallthrough particle of missing species'
        res[sp_name] = np.append(res[sp_name], fell_from_prev_layer[sp_name])
    return res


# pylint: disable=too-many-arguments
@hacks.friendly('lcode.main.beam_move')
def beam_move(config, beam_layer, t, xi, Ex, Ey, Ez, Bx, By, Bz):
    if isinstance(beam_layer, dict):
        species = beam_layer
    else:
        species = {'particles': beam_layer}

    moved, fell, lost, total_substeps = {}, {}, {}, 0
    for sp_name in species:
        moved_, fell_, lost_, total_substeps_ = beam_mover.move(
            config, beam_layer[sp_name], t, xi, Ex, Ey, Ez, Bx, By, Bz
        )
        moved[sp_name] = moved_
        fell[sp_name] = fell_
        lost[sp_name] = lost_
        total_substeps += total_substeps_
    return moved, fell, lost, total_substeps


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
