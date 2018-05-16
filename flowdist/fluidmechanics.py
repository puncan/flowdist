import numpy as np


"""Eventually, this module should be able to set up a 3D transient fluid flow problem for a rectangular
duct for the solver inputs. Currently, only a 2D advection-diffusion problem is to be set up to get
things rolling. The initial conditions, boundary conditions, and fluid properties will be established here"""


class FiniteVolume(object):
    """A finite volume is a defined space in which energy and mass can enter and leave via several processes including
    diffusion and advection. In this class, the finite volume is a quadrahedron.

    Attributes
    __________

    size : dx, dy, and dz as length, height, and depth of the quadrahedron in [m]
    time_step : dt in [s]
    position : x, y, and z location in [m]
    boundary : 'top', 'bottom', 'left', 'right', 'front', 'back' of domain where up to 3 may be chosen
    velocity_estimated : u, v, and w velocities in [m/s]
    velocity : u, v, and w velocities in [m/s]
    pressure_estimated : p pressure in [Pa]
    pressure : p pressure in [Pa]
    temperature : temp
    density : fluid density in [kg/m^3]
    viscosity : in [kg/(m*s)]
    conductivity : fluid thermal conductivity [W/(m*K)]
    """

    def __init__(self, size, time_step, position, boundary, velocity_estimated, velocity, pressure_estimated, pressure,
                 temperature, density, viscosity, conductivity):

        self.size = size
        self.time_step = time_step
        self.position = position
        self.boundary = boundary
        self.velocity_estimated = velocity_estimated
        self.velocity = velocity
        self.pressure_estimated = pressure_estimated
        self.pressure = pressure
        self.temperature = temperature
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

        pe_i = self.density*self.velocity['u']*self.size['dx']/self.viscosity
        a_pe_i = np.max(np.array([0, 1 - (np.abs(pe_i)/2)]))
        a_i_base = (self.viscosity / self.size['dx'])*a_pe_i
        a_i_sup = self.density*self.velocity['u']

        pe_j = self.density*self.velocity['v']*self.size['dy']/self.viscosity
        a_pe_j = np.max(np.array([0, 1 - (np.abs(pe_j)/2)]))
        a_j_base = (self.viscosity / self.size['dy'])*a_pe_j
        a_j_sup = self.density*self.velocity['v']

        pe_k = self.density*self.velocity['w']*self.size['dz']/self.viscosity
        a_pe_k = np.max(np.array([0, 1 - (np.abs(pe_k)/2)]))
        a_k_base = (self.viscosity / self.size['dz'])*a_pe_k
        a_k_sup = self.density*self.velocity['w']

        return a_m, [a_i_base, a_j_base, a_k_base], [a_i_sup, a_j_sup, a_k_sup]

    def set_boundary_values(self, p_inlet, p_outlet):
        """Depending on the boundary of the finite volume, the boundary conditions for fluid flow will either be a known
        pressure or a velocity of zero in all directions"""

        if 'top' or 'bottom' or 'front' or 'back' in self.boundary:
            self.velocity = {'u': 0, 'v': 0, 'w': 0}

        if 'left' in self.boundary:
            self.pressure = p_inlet

        if 'right' in self.boundary:
            self.pressure = p_outlet


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
