from numpy import sqrt, exp, cos, pi
import numpy as np

import matplotlib
matplotlib.use('Agg')

import hacks as hks

import lcode.beam_particle
import lcode.plasma_construction


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
        b = np.array(f['beam'])[::7]
        #plt.figure(figsize=(8 * 19.2, 10.8), dpi=100)
        plt.figure(figsize=(19.2, 10.8), dpi=100)
        plt.ylim(-6.4, 6.4)
        plt.scatter(b['r'][:, 0], b['r'][:, 1], s=.5, marker='.', color='black')
        plt.savefig('beam_portrait%05d.png' % h5_idx)
        plt.close()
        h5_idx += 1


hacks = [
    'lcode.diagnostics.main:Diagnostics',
    'lcode.diagnostics.print_beam_info:PrintBeamInfo',
    'lcode.diagnostics.print_Ez_max:PrintEzMax',
    'lcode.diagnostics.quick_density:QuickDensity',
#    'lcode.diagnostics.transverse_peek:TransversePeek',
    make_video, beam_portrait
]

#def transverse_peek_enabled(xi, xi_i): return abs(xi % .5) < 1e-3
#def transverse_peek_enabled(xi, xi_i): return True
def transverse_peek_enabled(xi, xi_i): return xi_i % 4 == 0

def track_plasma_particles(plasma): return []


time_start = 0
# time_steps = 100
time_steps = 5000
#time_step_size = 5 * 3000
time_step_size = 1200


from lcode.diagnostics.main import EachXi
diagnostics_enabled = True
diagnostics = [
    EachXi('Ez_00', lambda Ez: Ez[grid_steps // 2, grid_steps // 2]),
    EachXi('Ez_max', lambda Ez: Ez.max()),
]


window_width = 12.8
#grid_steps = 2**8 + 1  # 257; 15 / 257 ~= 0.0583
grid_steps = 2**6 + 1
# plasma_solver_eps = 0.0000001
# plasma_solver_B_0 = 0
# plasma_solver_corrector_passes = 1
# plasma_solver_corrector_transverse_passes = 3
# plasma_solver_particle_mover_corrector = 3
# plasma_solver_reuse_EB = True
# plasma_solver_use_average_speed = False
# xi_step_size = .05 
xi_step_size = .05
beam_length_in_xi_steps = int(1500 / xi_step_size)
# xi_steps = beam_length_in_xi_steps
# xi_steps = beam_length_in_xi_steps // 3
xi_steps = beam_length_in_xi_steps // 5
# print_every_xi_steps = 1
# openmp_limit_threads = 0
# plasma_solver_fields_interpolation_order = -1
plasma_solver_fields_interpolation_order = -2
# beam_deposit_nearest = True
beam_deposit_nearest = False
# beam_save = True  # better do this by default


def beam():
    #PARTICLES_PER_LAYER = 160  # increase
    PARTICLES_PER_LAYER = 1000  # increase?
    #PARTICLES_PER_LAYER = 4000  # increase?
    # CURRENT = 0.00281
    # weight = 12.56637061 * CURRENT / PARTICLES_PER_LAYER  # ?????
    WEIGHT = 3e11 / 2 / beam_length_in_xi_steps / PARTICLES_PER_LAYER
    #print('weight', WEIGHT)
    #WEIGHT *= xi_step_size * (window_width / grid_steps) ** 2
    # n0 / (wp/c)**3  or whatever else the charge unit really is  ~=  5e9
    # source: Tuev
    WEIGHT /= 5e9 * xi_step_size * (window_width / grid_steps) ** 2
    #print('weight', WEIGHT)

    np.random.seed(0)
    N = 0
    for xi_i in range(xi_steps):
        xi_microstep = xi_step_size / PARTICLES_PER_LAYER
        xi = -xi_i * xi_step_size - xi_microstep / 2
        # Doesn't match LCODE's check
        weight = WEIGHT * cos(.5 * np.pi * xi_i / beam_length_in_xi_steps)
        #print('weight', weight, WEIGHT, weight / WEIGHT)
        for _ in range(PARTICLES_PER_LAYER):
            yield lcode.beam_particle.BeamParticle(
                x=np.random.normal(),
                y=np.random.normal(),
                xi=xi,
                p_xi=np.random.normal(7.83e5, 2740),  # *2?,
                m=weight * 3000,
                q=weight,
                N=N
            )
            xi -= xi_microstep
            N += 1


plasma = lcode.plasma_construction.UniformPlasma(window_width,
                                                 grid_steps,
                                                 substep=2)
plasma = lcode.plasma_particle.PlasmaParticleArray(plasma)
