from numpy import sqrt, exp, cos, pi
import numpy as np

import matplotlib
matplotlib.use('Agg')

import hacks as hks

import lcode.beam_particle
import lcode.plasma.virtual.interleaved
import lcode.plasma_particle

idx = 0
@hks.after('simulation_time_step')
def make_video(self, retval, *a, **kwa):
    global idx
    cmd = "ffmpeg -r 30 -pattern_type glob -y -i 'transverse/*.png' v_aw_%05d.mp4"
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
        #plt.figure(figsize=(8 * 19.2, 10.8), dpi=100)
        b = np.array(f['beam'])
        plt.figure(figsize=(19.2, 10.8), dpi=100)
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
    'lcode.diagnostics.print_beam_info:PrintBeamInfo',
    'lcode.diagnostics.print_Ez_max:PrintEzMax',
    'lcode.diagnostics.quick_density:QuickDensity',
    #'lcode.diagnostics.transverse_peek:TransversePeek',
    #'lcode.diagnostics.live:LiveDiagnostics',
    #make_video,
    #beam_portrait,
    'lcode.beam.mpi:MPIPowers',
    'lcode.beam.archive:BeamArchive',
]

def transverse_peek_enabled(xi, xi_i): return xi_i % 4 == 0

def track_plasma_particles(plasma): return []


time_start = 0
# time_steps = 100
time_steps = 5000
time_step_size = 50
#time_step_size = 1200


from lcode.diagnostics.main import EachXi
diagnostics_enabled = True
diagnostics = [
    EachXi('Ez_00', lambda Ez: Ez[grid_steps // 2, grid_steps // 2]),
    EachXi('Ez_max', lambda Ez: Ez.max()),
]

window_width = 12.85 # window size
grid_steps = 2**7 + 1
plasma_solver_eps = 0.0000001
plasma_solver_B_0 = 0
plasma_solver_corrector_passes = 2
plasma_solver_corrector_transverse_passes = 1
plasma_solver_particle_mover_corrector = 2
xi_step_size = .05 / 2
beam_length_in_xi_steps = int(1500 / xi_step_size)
xi_steps = beam_length_in_xi_steps // 15
print_every_xi_steps = 20
openmp_limit_threads = 1
plasma_solver_fields_interpolation_order = -1
plasma_solver_use_average_speed = False
plasma_solver_zero_edges = True
beam_deposit_nearest = False


plasma, virtualize = lcode.plasma.virtual.interleaved.make(
    window_width, grid_steps, coarseness=2, fineness=2
)


def beam():
    PARTICLES_PER_LAYER = 1000  # increase?
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
    #EachXi2D('ro', lambda roj: roj['ro'], vmin=-0.05, vmax=0.05),
    EachXi('Ez_00', lambda Ez: Ez[grid_steps // 2, grid_steps // 2]),
    EachXi('Bz_00', lambda Bz: Bz[grid_steps // 2, grid_steps // 2]),
]
