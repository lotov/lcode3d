from numpy import sqrt, exp, cos, pi
import numpy as np

import lcode.plasma.virtual.interleaved
import lcode.plasma_construction
import lcode.plasma_particle

hacks = [
    'lcode.beam.ro_function:BeamRoFunction',
    'lcode.diagnostics.main:Diagnostics',
    'lcode.diagnostics.print_Ez_max:PrintEzMax',
    'lcode.diagnostics.live:LiveDiagnostics',
]

def transverse_peek_enabled(xi, xi_i):
    return True

def beam(xi, x, y):
    COMPRESS, BOOST, S, SHIFT = 1, 1, 1, 0  # 2, 1, 1, 0
    if xi < -2 * sqrt(2 * pi) / COMPRESS:
        return 0
    r = sqrt(x**2 + (y - SHIFT)**2)
    A = .05 * BOOST
    return A * exp(-.5 * (r/S)**2) * (1 - cos(xi * COMPRESS * sqrt(pi / 2)))

window_width = 12.85  # window size
grid_steps = 2**8 + 1
plasma_solver_corrector_passes = 2
plasma_solver_corrector_transverse_passes = 1
plasma_solver_particle_mover_corrector = 2
#plasma_solver_fields_interpolation_order = -2
#plasma_solver_boundary_suppression = 1
xi_step_size = .05
xi_steps = int(300 // xi_step_size)
#plasma_solver_reuse_EB = True
plasma_solver_use_average_speed = True

grid_step = window_width / grid_steps
#plasma, virtualize = lcode.plasma.virtual.interleaved.make(
#    window_width - 6 * grid_step, grid_steps - 6, coarseness=2, fineness=2
#)
#plasma = lcode.plasma_construction.UniformPlasma(
#    window_width - 6 * grid_step, grid_steps - 6, substep=1
#)
plasma, _ = lcode.plasma.virtual.interleaved.make(
    window_width - 6 * grid_step, grid_steps - 6, coarseness=1, fineness=1
)
#plasma = lcode.plasma_particle.PlasmaParticleArray(plasma)
#plasma_r = np.sqrt(plasma['x']**2 + plasma['y']**2)
#plasma = plasma[plasma_r < window_width / 2 - 6 * grid_steps / 2]
#plasma = plasma[plasma_r < 4]

from lcode.diagnostics.main import EachXi, EachXi2D
diagnostics_enabled = True
diagnostics = [
EachXi(
    'Ez_00', lambda Ez: Ez[grid_steps // 2, grid_steps // 2]),
    #EachXi2D('ro', lambda roj: roj['ro'], vmin=-0.2, vmax=0.2),
    #EachXi('Phi', lambda Sz: np.sum(Sz)),
    #EachXi2D('Sz', lambda Sz: Sz, vmin=-0.01, vmax=0.01),
    #EachXi('W_kin', lambda plasma:
    #    np.sum(plasma['m'] * (plasma['v'][:, 0]**2 +
    #                          plasma['v'][:, 1]**2 +
    #                          plasma['v'][:, 2]**2)) / 2
    #),
    #EachXi('W_pot', lambda plasma, Phi_Ezs: np.sum(plasma['q'] * Phi_Ezs)),
    #EachXi('W_tot', lambda current: current['W_pot'] + current['W_kin']),
]
