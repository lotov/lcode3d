import numpy as np

import lcode.beam_particle
import lcode.plasma.virtual.fast
import lcode.plasma_particle

hacks = [
    'lcode.util:FancyLogging',
    'lcode.util:DebugLogging',
    'lcode.util:FileLogging',
    'lcode.util:LcodeInfo',
    'lcode.diagnostics.print_beam_info:PrintBeamInfo',
]

time_steps = 1
time_step_size = 200

window_width = 12.85 # window size
grid_steps = 2**8 + 1
plasma_solver_eps = 0.0000001
plasma_solver_B_0 = 0
plasma_solver_corrector_passes = 2
plasma_solver_corrector_transverse_passes = 1
plasma_solver_particle_mover_corrector = 2
xi_step_size = .05
beam_length_in_xi_steps = int(1500 / xi_step_size)
xi_steps = beam_length_in_xi_steps // 15
print_every_xi_steps = 20
openmp_limit_threads = 1
plasma_solver_fields_interpolation_order = -1
plasma_solver_use_average_speed = False
plasma_solver_zero_edges = True
beam_deposit_nearest = False

plasma, virtualize = lcode.plasma.virtual.fast.make(
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
        weight = WEIGHT * (1 + np.cos(phase)) / 2 * 2
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
