from numpy import sqrt, exp, cos, pi
import numpy as np

import lcode.plasma.virtual.interleaved
import lcode.plasma_particle

hacks = [
    'lcode.beam.ro_function:BeamRoFunction',
    'lcode.diagnostics.main:Diagnostics',
    'lcode.diagnostics.print_Ez_max:PrintEzMax',
    'lcode.diagnostics.quick_density:QuickDensity',
#    'lcode.diagnostics.transverse_peek:TransversePeek',
    'lcode.diagnostics.live:LiveDiagnostics',
]

def transverse_peek_enabled(xi, xi_i):
    return True
    #return xi_i % 2 == 0

def track_plasma_particles(plasma):
    return []


def beam(xi, x, y):
    if xi < -2 * sqrt(2 * pi):
        return 0
    r = sqrt(x**2 + y**2)
    return .05 * exp(-.5 * r**2) * (1 - cos(xi * sqrt(pi / 2)))

window_width = 12.85 * 2 # window size
grid_steps = 2**8 + 1
plasma_solver_eps = 0.0000001
plasma_solver_B_0 = 0
plasma_solver_corrector_passes = 2
plasma_solver_corrector_transverse_passes = 1
plasma_solver_particle_mover_corrector = 2
xi_step_size = .05
xi_steps = 1400 * 10
print_every_xi_steps = 4
openmp_limit_threads = 0
#plasma_solver_fields_interpolation_order = -1
plasma_solver_use_average_speed = False
#plasma_solver_zero_edges = True


plasma, virtualize = lcode.plasma.virtual.interleaved.make(
    window_width, grid_steps, coarseness=2, fineness=2
)


from lcode.diagnostics.main import EachXi
diagnostics_enabled = True
diagnostics = [
    EachXi('Ez_00', lambda Ez: Ez[grid_steps // 2, grid_steps // 2]),
    EachXi('Bz_00', lambda Bz: Bz[grid_steps // 2, grid_steps // 2]),
]
