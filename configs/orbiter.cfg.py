import numpy as np
from numpy import sqrt, exp, cos, pi
import lcode.beam_particle
import lcode.plasma_particle

import hacks as hks


@hks.into('print_extra')
@hks.stealing
def print_plasma(*a, plasma=hks.steal):
    return (' '.join(['x=' + str(plasma['x'][0]),
                      'y=' + str(plasma['y'][0]),
                      'p=' + str(plasma['p'][0, 1]**2 +
                                 plasma['p'][0, 2]**2)
                     ]))


hacks = [
    'lcode.beam.ro_function:BeamRoFunction',
    #'lcode.diagnostics.quick_density:QuickDensity',
    'lcode.diagnostics.orbiter:Orbiter',
    #'lcode.diagnostics.transverse_peek:TransversePeek',
    print_plasma,
]


dt = t_max = 1

def beam(xi, x, y):
    return 0

window_width = 6
grid_steps = 2**4 + 1
plasma_solver_eps = 0.0000001
plasma_solver_B_0 = 0
plasma_solver_corrector_passes = 1
plasma_solver_corrector_transverse_passes = 3
plasma_solver_particle_mover_corrector = 2
xi_step_size = .05 * 4
xi_steps = int(180 / xi_step_size)
print_every_xi_steps = 1
openmp_limit_threads = 0
plasma_solver_fields_interpolation_order = -2
#plasma_solver_fields_smoothing_level = 0.001

electron = lcode.plasma_particle.Electron(x=1.5, y=0, p_y=.12)
ion = lcode.plasma_particle.Ion(x=0, y=0)
plasma = lcode.plasma_particle.PlasmaParticleArray([electron, ion])

def force_cut_enabled(xi, xi_i):
    return True
    return xi_i % 2 == 0

def transverse_peek_enabled(xi, xi_i):
    return xi_i % 100 == 0

def track_plasma_particles(plasma):
    return plasma
