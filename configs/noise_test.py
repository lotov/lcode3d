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
    zn = np.abs(hf).mean() / 4.23045376e-04
    max_zn = max(max_zn, zn)
    return 'zn=%.3f/%.3f' % (zn, max_zn)

hacks = [
    'lcode.beam.ro_function:BeamRoFunction',
    'lcode.diagnostics.main:Diagnostics',
    'lcode.diagnostics.Ez00_peaks:Ez00Peaks',
    'lcode.diagnostics.quick_density:QuickDensity',
    print_zn,
    'lcode.diagnostics.live:LiveDiagnostics',
]

def beam(xi, x, y):
    A, L = 0.1, 24
    b = np.zeros_like(x * y)
    N = b.shape[0]
    b[N//2, N//2] = A * (xi / L if xi < L else 1)
    return b
    #COMPRESS, BOOST, SIGMA, SHIFT = 1, 2, 1, 0  #1.5
    #if xi < -2 * np.sqrt(2 * np.pi) / COMPRESS:
    #    return 0
    #r = sqrt(x**2 + (y - SHIFT)**2)
    #return (.05 * BOOST * exp(-.5 * (r / SIGMA)**2) *
    #        (1 - np.cos(xi * COMPRESS * sqrt(np.pi / 2))))

window_width = 3.3  # window size
grid_steps = 2**6 + 1
xi_step_size = .2
xi_steps = int(3000 // xi_step_size)

plasma_solver = 'v2_monolithic'
print_every_xi_steps = 5

grid_step = window_width / grid_steps
plasma = lcode.plasma.solver_v2_monolithic.make_plasma(
    window_width - 6 * grid_step, grid_steps - 6, 1
)
openmp_limit_threads = 4

noise_reductor_enable = True
noise_reductor_equalization = 0.05
noise_reductor_friction = 0.1
#noise_reductor_final_only = True
#close_range_compensation = -1


def transverse_peek_enabled(xi, xi_i):
    return xi_i % 5 == 0
    #return True

init_plasma = plasma.copy()
h = window_width / grid_steps
from lcode.diagnostics.main import EachXi, EachXi2D
diagnostics_enabled = True
diagnostics = [
    #EachXi2D('pz', lambda plasma: plasma['p'][:, 0], vmin=-1e-2 * h, vmax=1e-2 * h),
    #EachXi2D('x_offt', lambda plasma: plasma['x'] - init_plasma['x'], vmin=-1e-8 * h, vmax=1e-8 * h),
    EachXi2D('ro', lambda roj: roj['ro'], vmin=-0.2/4, vmax=0.2/4),
    #EachXi2D('roT', lambda roj: roj['ro'].T, vmin=-0.2, vmax=0.2),
    #EachXi2D('Ex', lambda Ex: Ex, vmin=-0.001, vmax=0.001),
    #EachXi2D('jz', lambda roj: roj['jz'], vmin=-0.001, vmax=0.001),
    #EachXi('Ez_00', lambda Ez: Ez[grid_steps // 2, grid_steps // 2]),
]
