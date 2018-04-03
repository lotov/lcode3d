import numpy as np
from numpy import sqrt, exp, cos, pi

import lcode.plasma.virtual.interleaved
import lcode.plasma.solver_v2_monolithic

import hacks as hs
import scipy.ndimage
max_zn = 0
@hs.into('print_extra')
@hs.stealing
def print_zn(config:hs.steal, roj:hs.steal):
    global max_zn
    sigma = 0.25 * config.grid_steps / config.window_width
    blurred = scipy.ndimage.gaussian_filter(roj['ro'], sigma=sigma)
    hf = roj['ro'] - blurred
    #zn = np.abs(hf).mean() / 4.20229925e-4
    #zn = np.abs(hf).mean() / 6.30581890e-3  # COMPRESSED
    zn = np.abs(hf).mean() / 4.23045376e-04
    max_zn = max(max_zn, zn)
    #return 'zn=%.8e' % max_zn
    return 'zn=%.3f/%.3f' % (zn, max_zn)

hacks = [
    'lcode.beam.ro_function:BeamRoFunction',
    'lcode.diagnostics.main:Diagnostics',
    'lcode.diagnostics.Ez00_peaks:Ez00Peaks',
    'lcode.diagnostics.quick_density:QuickDensity',
    print_zn,
    #'lcode.diagnostics.live:LiveDiagnostics',
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
xi_steps = int(3000 // xi_step_size)

A, B = 0.08, 0.32

noise_reductor_enable = False
noise_reductor_final_only = True
#noise_reductor_equalization = 1
#noise_reductor_friction = 0.001
noise_reductor_equalization = A
noise_reductor_friction = A * B
#noise_reductor_reach = 2.5
plasma_solver = 'v2_monolithic'
#variant_A_predictor = variant_A_corrector = True
print_every_xi_steps = 20

grid_step = window_width / grid_steps
plasma = lcode.plasma.solver_v2_monolithic.make_plasma(
    window_width - 6 * grid_step, grid_steps - 6, 1
)
openmp_limit_threads = 8


def transverse_peek_enabled(xi, xi_i):
    return xi_i % 5 == 0
    #return True

from lcode.diagnostics.main import EachXi, EachXi2D
diagnostics_enabled = True
diagnostics = [
    EachXi('Ez_00', lambda Ez: Ez[grid_steps // 2, grid_steps // 2]),
    EachXi('zn', lambda roj: abs(np.gradient(roj['ro'], axis=0).sum())),
    #EachXi2D('ro', lambda roj: roj['ro'], vmin=-0.2, vmax=0.2),
]
