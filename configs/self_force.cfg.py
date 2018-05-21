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
                      'p=' + str(plasma['p'][0])]))


hacks = [
    'lcode.beam.ro_function:BeamRoFunction',
    'lcode.diagnostics.force_cut:ForceCut',
    #'lcode.diagnostics.transverse_peek:TransversePeek',
    print_plasma,
    #'lcode.diagnostics.main:Diagnostics',
    #'lcode.diagnostics.live:LiveDiagnostics',
]


dt = t_max = 1

def beam(xi, x, y):
    return 0

plasma_solver = 'v2_monolithic'
#plasma_padding = 6
#field_solver_subtraction_trick = 0
#field_solver_iterations = 2
print_every_xi_steps = 5

window_width = 12.85
grid_steps = 2**5 + 1
xi_step_size = .05
xi_steps = int(360 / xi_step_size)
print_every_xi_steps = 1
openmp_limit_threads = 0

the_only_electron = lcode.plasma_particle.Electron(x=-3, y=0, p_x=.02)
useless_ion = lcode.plasma_particle.Ion(x=0, y=0, q=1e-300, m=1e300)
plasma = lcode.plasma_particle.PlasmaParticleArray([the_only_electron, useless_ion])

openmp_limit_threads = 4

def force_cut_enabled(xi, xi_i):
    #return True
    return xi_i % 2 == 0

def transverse_peek_enabled(xi, xi_i):
    return xi_i % 100 == 0

def track_plasma_particles(plasma):
    return plasma[:0+1]

#from lcode.diagnostics.main import EachXi, EachXi2D
#diagnostics_enabled = True
#diagnostics = [
#    EachXi2D('ro', lambda roj: roj['ro'], vmin=-0.2, vmax=0.2),
#    EachXi2D('roT', lambda roj: roj['ro'].T, vmin=-0.2, vmax=0.2),
#    EachXi2D('Ex', lambda Ex: Ex, vmin=-0.001, vmax=0.001),
#    EachXi2D('jz', lambda roj: roj['jz'], vmin=-0.001, vmax=0.001),
#    EachXi('Ez_00', lambda Ez: Ez[grid_steps // 2, grid_steps // 2]),
#]
