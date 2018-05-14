import numpy as np


"""Eventually, this module should be able to set up a 3D transient fluid flow problem for a rectangular
duct for the solver inputs. Currently, only a 2D advection-diffusion problem is to be set up to get
things rolling. The initial conditions, boundary conditions, and fluid properties will be established here"""


class FiniteVolume(object):
    """A finite volume is a defined space in which energy and mass can enter and leave via several processes including
    diffusion and advection. In this class, the finite volume is a quadrahedron.

    Attributes
    __________

    size : length, width, and depth of the quadrahedron in [m]
    time_step : in [s]
    position : x, y, and z location in [m]
    reference : i, j, and k as negative or positive number
    velocity_estimated : u, v, and w velocities in [m/s]
    velocity : u, v, and w velocities in [m/s]
    pressure_estimated : p pressure in [Pa]
    pressure : p pressure in [Pa]
    density : fluid density in [kg/m^3]
    viscosity : in [kg/(m*s)]
    conductivity : fluid thermal conductivity [W/(m*K)]
    """

    def __init__(self, size, time_step, position, reference, velocity_estimated, velocity, pressure_estimated, pressure, density,
                 viscosity, conductivity):

        self.size = size
        self.time_step = time_step
        self.position = position
        self.reference = reference
        self.velocity_estimated = velocity_estimated
        self.velocity = velocity
        self.pressure_estimated = pressure_estimated
        self.pressure = pressure
        self.density = density
        self.viscosity = viscosity
        self.conductivity = conductivity

    def get_momentum_coefficients(self):
        """Calculates the time coefficient for all the momentum equations at the current time step and the a
        coefficients in the x, y, and z directions. The coefficients are dependent on the reference, being the relative
        location of an adjacent node. Currently, the generalized convection-diffusion scheme is used from Chapter 8 of
        "Computational Methods for Heat and Mass Transfer" by Pradip Majumdar with the option for adding 5 different
        approximation schemes. The Hybrid scheme is selected for now."""

        a_m = self.density*self.size['dx']*self.size['dy']*self.size['dz']/self.time_step

        pe_i = self.density*self.velocity_estimated['u']*self.size['dx']/self.viscosity
        a_pe_i = np.max(np.array([0, 1 - (np.abs(pe_i)/2)]))
        if self.reference['i'] > 0:
            a_i = (self.viscosity / self.size['dx'])*a_pe_i + np.max(
                np.array([0, -self.density*self.velocity_estimated['u']]))
        elif self.reference['i'] < 0:
            a_i = (self.viscosity / self.size['dx'])*a_pe_i + np.max(
                np.array([0, self.density*self.velocity_estimated['u']]))
        else:
            print('FlowElement.reference must be a positive or negative number')

        pe_j = self.density*self.velocity_estimated['v']*self.size['dy']/self.viscosity
        a_pe_j = np.max(np.array([0, 1 - (np.abs(pe_j)/2)]))
        if self.reference['j'] > 0:
            a_j = (self.viscosity / self.size['dy'])*a_pe_j + np.max(
                np.array([0, -self.density*self.velocity_estimated['v']]))
        elif self.reference['j'] < 0:
            a_j = (self.viscosity / self.size['dy'])*a_pe_j + np.max(
                np.array([0, self.density*self.velocity_estimated['v']]))
        else:
            print('FlowElement.reference must be a positive or negative number')

        pe_k = self.density*self.velocity_estimated['w']*self.size['dz']/self.viscosity
        a_pe_k = np.max(np.array([0, 1 - (np.abs(pe_k)/2)]))
        if self.reference['k'] > 0:
            a_k = (self.viscosity / self.size['dz'])*a_pe_k + np.max(
                np.array([0, -self.density*self.velocity_estimated['w']]))
        elif self.reference['k'] < 0:
            a_k = (self.viscosity / self.size['dz'])*a_pe_k + np.max(
                np.array([0, self.density*self.velocity_estimated['w']]))
        else:
            print('FlowElement.reference must be a positive or negative number')

        return [a_m, a_i, a_j, a_k]

    


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

