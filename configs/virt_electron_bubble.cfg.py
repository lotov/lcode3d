from numpy import sqrt, exp, cos, pi
import numpy as np

import matplotlib
matplotlib.use('Agg')

import hacks as hks

import lcode.beam_particle
import lcode.plasma.virtual.interleaved
import lcode.plasma_particle


hacks = [
    'lcode.diagnostics.main:Diagnostics',
    'lcode.diagnostics.print_beam_info:PrintBeamInfo',
    'lcode.diagnostics.print_Ez_max:PrintEzMax',
    'lcode.beam.mpi:MPIPowers',
    'lcode.beam.archive:BeamArchive',
]


time_start = 0
time_steps = 1000
time_step_size = 50


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
xi_step_size = .05
beam_length_in_xi_steps = int(5 / xi_step_size)
window_length_in_xi_steps = int((5 + 5) / xi_step_size)
xi_steps = window_length_in_xi_steps
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
    TOTAL_CHARGE_ABS = 71600190255.280746  # Approximately matched to 2D
    # Also, sum(q * W / r) = -91843517281.042526
    PARTICLES_PER_LAYER = 10000  # increase?
    WEIGHT = TOTAL_CHARGE_ABS / beam_length_in_xi_steps / PARTICLES_PER_LAYER
    np.random.seed(0)
    N = 0
    for xi_i in range(beam_length_in_xi_steps):
        xi_microstep = xi_step_size / PARTICLES_PER_LAYER
        xi = -xi_i * xi_step_size - xi_microstep / 2
        phase = np.pi * xi_i / beam_length_in_xi_steps * 2
        weight = WEIGHT * (1 - cos(phase)) / 2 * 2
        # * 2  =>  integral = 1  =>  total charge matches
        r = 1
        for _ in range(PARTICLES_PER_LAYER):
            yield lcode.beam_particle.BeamParticle(
                x=np.random.normal(0, r),
                y=np.random.normal(0, r),
                xi=xi,
                # TODO: correct angspread!!!
                p_x=np.random.uniform(0, 1) * 1,
                p_y=np.random.uniform(0, 1) * 1,
                #p_xi=np.random.normal(1e5, 2740),
                p_xi=1e5,
                W=weight,
                m=1,
                q=-1,
                N=N
            )
            xi -= xi_microstep
            N += 1

    SKIPPINESS, LIGHTNESS = 100, 1
    for xi_i in range(xi_i, window_length_in_xi_steps):
        xi_microstep = xi_step_size / PARTICLES_PER_LAYER * SKIPPINESS
        xi = -xi_i * xi_step_size - xi_microstep / 2
        weight = WEIGHT / LIGHTNESS
        r = 0.33
        for _ in range(PARTICLES_PER_LAYER // SKIPPINESS):
            yield lcode.beam_particle.BeamParticle(
                x=np.random.normal(0, r),
                y=np.random.normal(0, r),
                xi=xi,
                # TODO: correct angspread!!!
                p_x=np.random.uniform(0, 1) * 1,
                p_y=np.random.uniform(0, 1) * 1,
                #p_xi=np.random.normal(1e5, 2740),
                p_xi=1e5,
                W=weight,
                m=1,
                q=-1,
                N=N
            )
            xi -= xi_microstep
            N += 1

q, s = 0, 0
for p in beam():
    r = sqrt(p['r'][1]**2 + p['r'][2]**2)
    q += p['q'] * p['W']
    s += p['q'] * p['W'] / r
print('q', q, 's', s)
#import sys; sys.exit()


def archive(t, t_i):
    return (t_i + 1) % 1 == 0
