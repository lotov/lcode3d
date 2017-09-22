from numpy import sqrt, exp, cos, pi

import lcode.default_config
import lcode.beam_construction

hacks = lcode.default_config.hacks + [
    'lcode.beam.mpi:MPIPowers',
]

def transverse_peek_enabled(xi, xi_i):
    return True

beam = lcode.beam_construction.some_particle_beam

grid_steps = 2**7 + 1
xi_step_size = .05
xi_steps = 4
time_steps = 9
