import numpy as np
import sympy as sp

"""Eventually, this module should be able to set up a 3D transient fluid flow problem for a rectangular
duct for the solver inputs. Currently, only a 2D advection-diffusion problem is to be set up to get
things rolling. The initial conditions, boundary conditions, and fluid properties will be established here"""


def initial_conditions(velocity, grid):
    """Returns the velocity distribution as a numpy array at t = 0
    For 2D, the velocity input could be an integer or float if the velocity is the same along the
    entire channel. It could also be a function of x and y in the channel"""

    em1 = 'Initial velocity should be positive'
    em2 = 'Velocity must be scalar or function of x and y coordinates and grid must be numpy array'

    try:
        v = velocity(grid)
        if np.amin(v) < 0:
            return em1
        else:
            return v
    except TypeError:
        try:
            v = velocity*np.ones(np.shape(grid))
            if np.amin(v) < 0:
                return em1
            else:
                return v
        except TypeError:
            return em2


def momentum_finite_diff_equ():
    """Returns the symbolic momentum equations for single phase 2D unsteady Newtonian fluid flow.
    This is intended to eventually return the integrated and discretized form. The built in vector
    module in sympy will also be used in the future."""
    x, y, t, dx, dy, dz, dt, u, v, mu, rho, p, su = sp.symbols('x y t dx dy dz dt u v mu rho p su')

    u = sp.Function('u')(x, y, t)
    v = sp.Function('v')(x, y, t)
    p = sp.Function('p')(x, y, t)
    rho = sp.Function('rho')(x, y, t)
    mu = sp.Function('mu')(x, y, t)

    term_time = sp.Derivative(u, t)
    term_advection_out = sp.Derivative(rho*u**2, x) + sp.Derivative(rho*u*v, y)
    term_pressure_grad = sp.Derivative(p, x)
    term_diffusion = -sp.Derivative(mu*sp.Derivative(u, x), x) - sp.Derivative(mu*sp.Derivative(u, y), y)
    term_source = -su
    return term_time + term_advection_out + term_pressure_grad + term_diffusion + term_source

