from numpy import sqrt, exp, cos, pi

import lcode.beam_particle
import lcode.plasma_construction


hacks = [
    'lcode.beam.ro_function:BeamRoFunction',
    'lcode.diagnostics.main:Diagnostics',
    'lcode.diagnostics.print_beam_info:PrintBeamInfo',
]

time_start = 0
time_steps = 1
time_step_size = 0


def transverse_peek_enabled(xi, xi_i):
    return xi_i % 2 == 0


def beam(xi, x, y):
    COMPRESS, BOOST, S, SHIFT = 1, 1, 1, 0  # 2, 1, 1, 0
    if xi < -2 * sqrt(2 * pi) / COMPRESS:
        return 0
    r = sqrt(x**2 + (y - SHIFT)**2)
    A = .05 * BOOST
    return A * exp(-.5 * (r / S)**2) * (1 - cos(xi * COMPRESS * sqrt(pi / 2)))


window_width = 12.85
grid_steps = 2**6 + 1
plasma_solver_eps = 0.0000001
plasma_solver_B_0 = 0
plasma_solver_corrector_passes = 1
plasma_solver_corrector_transverse_passes = 3
plasma_solver_particle_mover_corrector = 3
plasma_solver_reuse_EB = True
plasma_solver_use_average_speed = False
xi_step_size = .05
xi_steps = 140
print_every_xi_steps = 1
openmp_limit_threads = 0
plasma_solver_fields_interpolation_order = -2
beam_deposit_nearest = True
beam_save = True  # better do this by default
plasma_solver_zero_edges = False
beam_mover_p_iterations = 3
beam_mover_substepping_trigger = .2
beam_mover_max_substepping = 1000
base_plasma_density = 7e14
plasma_solver_boundary_suppression = 1
Ez = Ex = Ey = Bz = Bx = By = 0

plasma_solver = 'v2'
plasma_padding = 3
variant_A_predictor = variant_A_corrector = False


def virtualize(a):
    return a


plasma_density_shape = 1


plasma = lcode.plasma_construction.UniformPlasma(window_width,
                                                 grid_steps,
                                                 substep=2)
plasma = lcode.plasma_particle.PlasmaParticleArray(plasma)
