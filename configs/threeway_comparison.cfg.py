from numpy import sqrt, exp, cos, pi
import numpy as np

import hacks as hks

import lcode.beam_particle
import lcode.plasma.virtual.fast
import lcode.plasma_particle


hacks = [
    'lcode.diagnostics.main:Diagnostics',
    'lcode.diagnostics.print_beam_info:PrintBeamInfo',
    'lcode.diagnostics.print_Ez_max:PrintEzMax',
    'lcode.diagnostics.quick_density:QuickDensity',
    'lcode.beam.mpi:MPIPowers',
    'lcode.beam.archive:BeamArchive',
]


time_start = 0
time_steps = 200
time_step_size = 500


window_width = 16 # window size
grid_steps = 2**8 + 1
plasma_solver_eps = 0.0000001
plasma_solver_B_0 = 0
plasma_solver_corrector_passes = 1
plasma_solver_corrector_transverse_passes = 5
plasma_solver_particle_mover_corrector = 2
xi_step_size = .3 / 5
beam_length_in_xi_steps = int(750 / xi_step_size)
xi_steps = beam_length_in_xi_steps
print_every_xi_steps = 1
openmp_limit_threads = 1
plasma_solver_fields_interpolation_order = -1
plasma_solver_use_average_speed = False
plasma_solver_zero_edges = True
beam_deposit_nearest = False


def plasma_density_shape(t):
    if t <= 5000:
        return 1
    elif 5000 < t <= 6000:
        return 1 + 0.03 * (t - 5000) / (6000 - 5000)
    else:
        return 1.03


plasma, virtualize = lcode.plasma.virtual.fast.make(
    window_width/2, grid_steps/2, coarseness=2, fineness=2
)


def beam():
    PARTICLES_PER_LAYER = 4000  # increase?
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


from lcode.diagnostics.main import EachXi
diagnostics_enabled = True
diagnostics = [
    EachXi('Ez_00', lambda Ez: Ez[grid_steps // 2, grid_steps // 2]),
    EachXi('Bz_00', lambda Bz: Bz[grid_steps // 2, grid_steps // 2]),
    EachXi('Ez_max', lambda Ez: np.max(Ez)),
]
