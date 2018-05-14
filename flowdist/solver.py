import numpy as np
#from CoolProp.CoolProp import PropsSI
from heattransfer import initial_conditions as htic


def define_grid(length_channel, height_channel, number_grid_x, number_grid_y):
    """Returns 2D element dimensions and numpy arrays containing the evenly spaced grid locations along the length and
    height (x and y)"""
    dx = length_channel/(number_grid_x - 1)
    dy = height_channel/(number_grid_y - 1)
    x = np.linspace(0.0, length_channel, num=number_grid_x)
    y = np.linspace(0.0, height_channel, num=number_grid_y)
    return dx, dy, x, y


def fluid_properties():
    '''Returns the fluid density, dynamic viscosity, and thermal conductivity in SI units. Ideally, for the flow field
    in 2D numpy arrays. These will be dependent on enthalpies (in J/kg), pressures (in Pa), and the fluid type.
    For now, the properties are hard coded constants for air at ambient conditions.'''

    #density = PropsSI('D', 'P', pressures, 'H', enthalpies, fluid)
    #viscosity_dynamic = PropsSI('V', 'P', pressures, 'H', enthalpies, fluid)
    #conductivity = PropsSI('L', 'P', pressures, 'H', enthalpies, fluid)

    return 1.225, 18.45e-6, 0.026


#h = PropsSI('H', 'X', 0, 'T', 300, 'air')

rho, mu, k = fluid_properties()

print(rho, mu, k)