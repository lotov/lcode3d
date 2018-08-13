import numpy as np
from numpy import sqrt, exp, cos, pi

import lcode.plasma.virtual.fast
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

#window_width = 12.85 # window size
window_width = 12.85 * (401 / 257)  # window size
#grid_steps = 2**9 + 1
#grid_steps = 401
grid_steps = 801
#window_width = 20  # window size
#grid_steps = 513
xi_step_size = .01
xi_steps = int(3000 // xi_step_size)

plasma_solver = 'v2_monolithic'
field_solver_subtraction_trick = 1
field_solver_iterations = 1
#plasma_padding = 3  # !!!
plasma_padding = 3
#variant_A_predictor = variant_A_corrector = True
print_every_xi_steps = int(1 / xi_step_size)
grid_step = window_width / grid_steps
#plasma = lcode.plasma.solver_v2_monolithic.make_plasma(
#    window_width - plasma_padding * 2 * grid_step, grid_steps - plasma_padding * 2, 1
#)
plasma, virtualize = lcode.plasma.virtual.fast.make(
    window_width - plasma_padding * 2 * grid_step, grid_steps - plasma_padding * 2,
    coarseness=3, fineness=2
)
openmp_limit_threads = 4

#density_noise_reductor = -.03  # 1dnr optimum
#density_noise_reductor = .06  # 4dnr optimum?
#density_noise_reductor = .2
#noise_reductor_enable = True
#noise_reductor_equalization = .1
#noise_reductor_friction = .0003
#noise_reductor_friction_pz = .0003
#noise_reductor_reach = 1
#noise_reductor_final_only = False
#close_range_compensation = -.04
#close_range_compensation = -.02  # !!! disabled for virtplasmas for now !!!

# 0.02/0.02 nr
# 3ppc, comp 0: 313 to zn>.17, 575 to zn=.5, 833 to zn=1, 947 to zn=2, 1379 to zn=3
# 1ppc, comp 0: 374 to zn>.17, 732 to zn=.5, 1191 to zn=1, >2573 to zn=1.5, -10%@634, -50%@1370
# 1ppc, comp -.25: 255 to zn>.17, 299 to zn=.5, 324 to zn=1, 372 to zn=2, -10%@>537
# 1ppc, comp -1: 115 to zn>.17, 141 to zn=.5, 148 to zn=1, 168 to zn=2, -2.5%@>176
# -- ordered noise reduction: equalization pass first, then friction pass on top of it --
# 1ppc, comp 0: 357 to zn>.17, 723 to zn=.5, 1191 to zn=1, ... -5%@357, -10%@634
# -- ordered noise reduction: friction pass first, then equalization pass on top of it --
# 1ppc, comp 0: 332 to zn>.17, 562 to zn=.5, ??? to zn=1, ... -1.47%697 center is noisy

# eq-fric, normal-sized window 0.02/0.02
# 1ppc, comp 0: 392>1, 518>1.5, 625>2, 864>3; -5%@364, -10%@628
# eq-fric, normal-sized window 0.00/0.02
# 1ppc, comp 0: 386>1, 499>1.5, 606>2, 839>3; -5%@364, -10%@647
# +dnr=0.6 (SAME!)
# no dnr, final_only 0/0.02
# 1ppc, comp 0: 368>1, 474>1.5, 562>2, 751>3; -5%@452, -10%@735
# +dnr=0.2
# 1ppc, comp 0: 367>1, 474>1.5, 562>2, 751>3; -5%@452, -10%@735
# +dnr=0.6
# now with nr 0/0.01/-final-only
# 1ppc, comp +1:     19>1,  23>1.5,  25>2,  26>3
# 1ppc, comp +0.2:   66>1,  78>1.5,  82>2,  85>3
# 1ppc, comp +0.02: 348>1, 417>1.5, 468>2, 517>3; -5%@452, -10%@741
# 1ppc, comp +0.005:367>1, 474>1.5, 556>2, 732>3; -5%@452, -10%@741
# 1ppc, comp  0:    368>1, 474>1.5, 562>2, 751>3; -5%@452, -10%@735
# 1ppc, comp -0.01: 367>1, 474>1,5, 562>2, 751>3; -5%@452, -10%@735
# 1ppc, comp -0.05: 363>1, 474>1.5, 556>2, 644>3; -5%@445, -10%@728
# 1ppc, comp -0.2:  287>1, 305>1.5, 317>2, 330>3; -5%@426, -10%@698
# 1ppc, comp -1:    126>1, 133>1.5, 139>2, 144>3;
# 1ppc, comp -2:     89>1,  95>1.5,  98>2, 101>3


def transverse_peek_enabled(xi, xi_i):
    return xi_i % int(.5 / xi_step_size) == 0
    #return True

from lcode.diagnostics.main import EachXi, EachXi2D
diagnostics_enabled = True
diagnostics = [
    EachXi('Ez_00', lambda Ez: Ez[grid_steps // 2, grid_steps // 2]),
    #EachXi2D('ro', lambda roj: roj['ro'], vmin=-0.2, vmax=0.2),
]

# 401x401, dxi .01 up to 800, 4 cores, 7x7 compensation
#real    2193m17.783s
#user    7860m34.780s
#sys     77m28.662s

# 401x401, dxi .02 up to 783, 4 cores, 5x5 compensation
#real    736m34.111s
#user    2519m50.175s
#sys     27m54.174s

