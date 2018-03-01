import numpy as np
from numpy import sqrt, exp, cos, pi

import lcode.plasma.virtual.interleaved
import lcode.plasma.solver_v2_monolithic

hacks = [
    'lcode.beam.ro_function:BeamRoFunction',
    'lcode.diagnostics.main:Diagnostics',
    'lcode.diagnostics.Ez00_peaks:Ez00Peaks',
]

def beam(xi, x, y):
    COMPRESS, BOOST, SIGMA, SHIFT = 1, 1, 1, 0
    if xi < -2 * np.sqrt(2 * np.pi) / COMPRESS:
        return 0
    r = sqrt(x**2 + (y - SHIFT)**2)
    return (.05 * BOOST * exp(-.5 * (r / SIGMA)**2) *
            (1 - np.cos(xi * COMPRESS * sqrt(np.pi / 2))))

window_width = 12.85  # window size
grid_steps = 2**8 + 1
xi_step_size = .05
xi_steps = int(100 // xi_step_size)

noise_reductor_enable = True
plasma_solver = lcode.plasma.solver_v2_monolithic

grid_step = window_width / grid_steps
plasma = lcode.plasma.solver_v2_monolithic.make_plasma(
    window_width - 6 * grid_step, grid_steps - 6, per_xi_step=1
)


def transverse_peek_enabled(xi, xi_i):
    return True

from lcode.diagnostics.main import EachXi, EachXi2D
diagnostics_enabled = True
diagnostics = [
EachXi(
    'Ez_00', lambda Ez: Ez[grid_steps // 2, grid_steps // 2]),
    #EachXi2D('ro', lambda roj: roj['ro'], vmin=-0.2, vmax=0.2),
]
