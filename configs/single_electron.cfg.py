import numpy as np
from numpy import sqrt, exp, cos, pi
import lcode.beam_particle
import lcode.plasma_construction


hacks = [
    'lcode.beam.ro_function:BeamRoFunction',
    'lcode.diagnostics.quick_density:QuickDensity',
#    'lcode.diagnostics.print_Ez_max:PrintEzMax',
]


dt = t_max = 1


def beam(xi, x, y):
    return 0


window_width = 12.85
grid_steps = 2**7 + 1
plasma_solver_eps = 0.0000001
plasma_solver_B_0 = 0
plasma_solver_corrector_passes = 1
plasma_solver_corrector_transverse_passes = 5
plasma_solver_particle_mover_corrector = 3
xi_step_size = .05
xi_steps = 1400 * 100
print_every_xi_steps = 1
openmp_limit_threads = 0
plasma_solver_fields_interpolation_order = -1
plasma_solver_fields_smoothing_level = 0.001

plasma = lcode.plasma_construction.UniformPlasma(window_width,
                                                 grid_steps,
                                                 substep=2)
plasma = lcode.plasma_particle.PlasmaParticleArray(plasma)

electrons = plasma[plasma['q'] < 0]


def transverse_peek_enabled(xi, xi_i):
    #return True
    return xi_i % 4 == 0
    #return abs(xi % 8) < 0.01 or (abs(xi % 2) < 0.01 and xi < -40)


def closest_electron(x, y):
    dist2 = (electrons['x'] - x)**2 + (electrons['y'] - y)**2
    return electrons[np.argmin(dist2)]['N']


probe_numbers = [closest_electron(2, -i) for i in range(7)]

r = sqrt(plasma['x']**2 + plasma['y']**2)
plasma = plasma[np.logical_or(plasma['q'] > 0, r > window_width / grid_steps)]


def track_plasma_particles(plasma):
    return plasma[np.in1d(plasma['N'], probe_numbers)]
