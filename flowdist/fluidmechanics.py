import numpy as np

"""Eventually, this module should be able to set up a 3D transient fluid flow problem for a rectangular
duct for the solver inputs. Currently, only a 1D advection-diffusion problem is to be set up to get
things rolling. The initial conditions, boundary conditions, and fluid properties will be established here"""


def get_velocity_distribution_at_t0(velocity, grid_position):
    """Returns the velocity distribution as a numpy array at t = 0
    For 1D, the velocity input could be an integer or float if the velocity is the same along the
    entire channel. It could also be a function of the position along the channel"""

    try:
        return velocity(grid_position)
    except TypeError:
        return velocity*np.ones(len(grid_position))


def get_boundary_conditions(inlet, outlet, wall, ):
    """"""



def velexample(x):
    return 2*x + 3


if __name__ == '__main__':
    print(get_velocity_distribution_at_t0(velexample, np.array([1, 2, 3, 4])))
