import numpy as np
from numpy import sqrt, exp, cos, pi

import lcode.plasma.virtual.fast
import lcode.plasma.solver_v2_monolithic

import hacks as hks
import scipy.ndimage
max_zn = 0
@hks.into('print_extra')
@hks.stealing
def print_zn(config:hks.steal, roj:hks.steal):
    global max_zn
    sigma = 0.25 * config.grid_steps / config.window_width
    blurred = scipy.ndimage.gaussian_filter(roj['ro'], sigma=sigma)
    hf = roj['ro'] - blurred
    zn = np.abs(hf).mean() / 4.23045376e-04
    max_zn = max(max_zn, zn)
    return 'zn=%.3f/%.3f' % (zn, max_zn)

idx = 0
@hks.after('simulation_time_step')
def make_video(self, retval, *a, **kwa):
    global idx
    cmd = "ffmpeg -r 30 -pattern_type glob -y -i 'transverse/*.png' -b:v 4m v_aw_%05d.mp4"
    import os
    os.system(cmd % idx)
    idx += 1

h5_idx = 1
@hks.after('simulation_time_step')
def beam_portrait(self, retval, *a, **kwa):
    global h5_idx
    import h5py
    from matplotlib import pyplot as plt
    with h5py.File('%05d.h5' % h5_idx, 'r') as f:
        b = np.array(f['beam']['particles'])
        plt.figure(figsize=(19.2 * 2, 10.8 * 2), dpi=100)
        plt.ylim(-6.4, 6.4)
        xis = np.linspace(np.ceil(min(b['r'][:, 0])), 0, 601)
        rs = np.linspace(-12.85, 12.85, 101)
        H, _, _ = np.histogram2d(b['r'][:, 0], b['r'][:, 1],
                                 weights=b['q'] * b['W'],
                                 bins=(xis, rs))
        #plt.scatter(b['r'][:, 0], b['r'][:, 1], s=.5, marker='.', color='black')
        plt.pcolormesh(xis, rs, H.T, cmap='jet', vmin=0, vmax=1e7)
        plt.colorbar()
        plt.savefig('beam_portrait%05d.png' % h5_idx)
        plt.close()
        h5_idx += 1


hacks = [
    'lcode.diagnostics.main:Diagnostics',
    'lcode.diagnostics.Ez00_peaks:Ez00Peaks',
    'lcode.diagnostics.print_beam_info:PrintBeamInfo',
    'lcode.diagnostics.print_Ez_max:PrintEzMax',
    'lcode.diagnostics.quick_density:QuickDensity',
    #'lcode.diagnostics.transverse_peek:TransversePeek',
    print_zn,
    #'lcode.diagnostics.live:LiveDiagnostics',
    make_video,
    #beam_portrait,
    #'lcode.beam.mpi:MPIPowers',
    'lcode.beam.archive:BeamArchive',
]

def transverse_peek_enabled(xi, xi_i): return xi_i % 4 == 0

time_start = 0
time_steps = 5000
#time_step_size = 50
time_step_size = 200

#window_width = 12.85 # window size
#grid_steps = 2**8 + 1
window_width = 12.85 * (401 / 257)  # window size
grid_steps = 401
#grid_steps = 801
xi_step_size = .01
xi_steps = int(1500 // xi_step_size)

plasma_solver = 'v2_monolithic'
field_solver_subtraction_trick = 1
field_solver_iterations = 1
plasma_padding = 3
print_every_xi_steps = int(1 / xi_step_size)
grid_step = window_width / grid_steps
#plasma = lcode.plasma.solver_v2_monolithic.make_plasma(
#    window_width - plasma_padding * 2 * grid_step, grid_steps - plasma_padding * 2, 1
#)
plasma, virtualize = lcode.plasma.virtual.fast.make(
    window_width - plasma_padding * 2 * grid_step, grid_steps - plasma_padding * 2,
    coarseness=2, fineness=2
)
openmp_limit_threads = 4

#density_noise_reductor = .2
#noise_reductor_enable = True
#noise_reductor_equalization = .1
#noise_reductor_friction = .0003
#noise_reductor_friction_pz = .0003
#noise_reductor_reach = 1
#noise_reductor_final_only = False
# close_range_compensation = -.04     # !!! disabled for now for virtplasmas !!!


def transverse_peek_enabled(xi, xi_i):
    #return True
    return xi_i % int(.5 / xi_step_size) == 0

from lcode.diagnostics.main import EachXi, EachXi2D
diagnostics_enabled = True
diagnostics = [
    EachXi('Ez_00', lambda Ez: Ez[grid_steps // 2, grid_steps // 2]),
    #EachXi2D('ro', lambda roj: roj['ro'], vmin=-0.2, vmax=0.2),
    EachXi('Ez_max', lambda Ez: Ez.max()),
]

beam_length_in_xi_steps = int(1500 / 2 / xi_step_size)
beam_deposit_nearest = False

def beam():
    PARTICLES_PER_LAYER = 2000  # increase?
    WEIGHT = 3e11 / 2 / beam_length_in_xi_steps / PARTICLES_PER_LAYER
    # / 2 is because only half a beam travels in ionized plasma
    np.random.seed(0)
    N = 0
    for xi_i in range(xi_steps):
        xi_microstep = xi_step_size / PARTICLES_PER_LAYER
        xi = -xi_i * xi_step_size - xi_microstep / 2
        phase = np.pi * xi_i / beam_length_in_xi_steps
        weight = WEIGHT * (1 + cos(phase)) / 2 * 2
        # * 2  =>  integral = 1  =>  total charge = 3e11 / 2
        for _ in range(PARTICLES_PER_LAYER):
            yield lcode.beam_particle.BeamParticle(
                x=np.random.normal(),
                y=np.random.normal(),
                xi=xi,
                p_xi=np.random.normal(7.83e5, 2740),
                W=weight,
                m=1836.152674,
                q=1,
                N=N
            )
            xi -= xi_microstep
            N += 1


def archive(t, t_i):
    return (t_i + 1) % 8 == 0


from lcode.diagnostics.main import EachXi, EachXi2D
diagnostics_enabled = True
diagnostics = [
    #EachXi2D('ro', lambda roj: roj['ro'], vmin=-0.005, vmax=0.005),
    #EachXi2D('beam_ro', lambda beam_ro: beam_ro, vmin=-0.005, vmax=0.005),
    EachXi('Ez_00', lambda Ez: Ez[grid_steps // 2, grid_steps // 2]),
    EachXi('Bz_00', lambda Bz: Bz[grid_steps // 2, grid_steps // 2]),
]
