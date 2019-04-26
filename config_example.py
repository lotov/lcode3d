grid_steps = 641  #: Transverse grid size in cells
grid_step_size = .025  #: Transverse grid step size in plasma units

xi_step_size = .005
xi_steps = int(3000 // xi_step_size)

diagnostics_each_N_steps = int(1 / xi_step_size)

field_solver_subtraction_trick = 1
field_solver_variant_A = True  #: Use Variant A or Variant B for Ex, Ey, Bx, By

reflect_padding_steps = 5
plasma_padding_steps = 10

plasma_coarseness = 3  #: Square root of the amount of cells per coarse particle
plasma_fineness = 2  #: Square root of the amount of fine particles per cell


from numpy import cos, exp, pi, sqrt

def beam(xi_i, x, y):
    xi = -xi_i * xi_step_size
    COMPRESS, BOOST, SIGMA, SHIFT = 1, 1, 1, 0
    if xi < -2 * sqrt(2 * pi) / COMPRESS:
        return 0
    r = sqrt(x**2 + (y - SHIFT)**2)
    return (.05 * BOOST * exp(-.5 * (r / SIGMA)**2) *
            (1 - cos(xi * COMPRESS * sqrt(pi / 2))))

gpu_index = 0  #: Index of the GPU that should perform the calculations

