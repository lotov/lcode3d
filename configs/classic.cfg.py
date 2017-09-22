from numpy import sqrt, exp, cos, pi
import numpy as np

import lcode.plasma_construction
import lcode.plasma_particle

hacks = [
    'lcode.beam.ro_function:BeamRoFunction',
    'lcode.diagnostics.main:Diagnostics',
    'lcode.diagnostics.print_beam_info:PrintBeamInfo',
#    'lcode.diagnostics.quick_density:QuickDensity',
#    'lcode.diagnostics.transverse_peek:TransversePeek',
    'lcode.diagnostics.live:LiveDiagnostics',
]

def transverse_peek_enabled(xi, xi_i):
    return xi_i % 2 == 0

def track_plasma_particles(plasma):
    return []


def beam(xi, x, y):
    if xi < -2 * sqrt(2 * pi):
        return 0
    r = sqrt(x**2 + y**2)
    return .05 * exp(-.5 * r**2) * (1 - cos(xi * sqrt(pi / 2)))

window_width = 12.85  # window size
grid_steps = 2**6 + 1
plasma_solver_eps = 0.0000001
plasma_solver_B_0 = 0
plasma_solver_corrector_passes = 1
plasma_solver_corrector_transverse_passes = 4
plasma_solver_particle_mover_corrector = 2
xi_step_size = .05 * 4
xi_steps = 1400 * 4 // 4
print_every_xi_steps = 140 // 4
openmp_limit_threads = 0
#plasma_solver_fields_interpolation_order = -2
#plasma_solver_zero_edges = True

plasma = lcode.plasma_construction.UniformPlasma(window_width,
                                                 grid_steps,
                                                 substep=2)

from lcode.diagnostics.main import EachXi, EachXi2D
diagnostics_enabled = True
diagnostics = [
    EachXi2D('ro', lambda roj: roj['ro'], vmin=-0.2, vmax=0.2),
    EachXi('Phi', lambda Sz: np.sum(Sz)),
    EachXi2D('Sz', lambda Sz: Sz, vmin=-0.01, vmax=0.01),
    EachXi('W_kin', lambda plasma:
        np.sum(plasma['m'] * (plasma['v'][:, 0]**2 +
                              plasma['v'][:, 1]**2 +
                              plasma['v'][:, 2]**2)) / 2
    ),
    EachXi('W_pot', lambda plasma, Phi_Ezs: np.sum(plasma['q'] * Phi_Ezs)),
    EachXi('W_tot', lambda current: current['W_pot'] + current['W_kin']),
    EachXi('Ez_00', lambda Ez: Ez[grid_steps // 2, grid_steps // 2]),
    EachXi('Bz_00', lambda Bz: Bz[grid_steps // 2, grid_steps // 2]),
]

