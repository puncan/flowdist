import numpy as np
from CoolProp.CoolProp import PropsSI
#from fluidmechanics import FiniteVolume


def define_grid(length_channel, height_channel, width_channel, number_grid_x, number_grid_y, number_grid_z):
    """Returns 3D element dimensions and numpy arrays containing the evenly spaced grid locations along the length,
    height, and width (x, y, and z)"""
    dx = length_channel/(number_grid_x)
    dy = height_channel/(number_grid_y)
    dz = width_channel/(number_grid_z)
    x = np.linspace(0.0, length_channel - dx, num=number_grid_x)
    y = np.linspace(0.0, height_channel - dy, num=number_grid_y)
    z = np.linspace(0.0, width_channel - dz, num=number_grid_z)
    return [dx, dy, dz], [x, y, z]


def fluid_properties(pressures, temperatures, fluid):
    """Returns the fluid density, dynamic viscosity, and thermal conductivity in SI units. Ideally, for the flow field
    in 2D numpy arrays. These will be dependent on enthalpies (in J/kg), pressures (in Pa), and the fluid type.
    For now, the properties are hard coded constants for air at ambient conditions since CoolProp is not cooperating."""

    density = PropsSI('D', 'P', pressures, 'T', temperatures, fluid)
    viscosity_dynamic = PropsSI('V', 'P', pressures, 'T', temperatures, fluid)
    conductivity = PropsSI('L', 'P', pressures, 'T', temperatures, fluid)

    return density, viscosity_dynamic, conductivity

"""
size, position = define_grid(0.1, 0.001, 0.001, 100, 10, 10)

size = {'dx': size[0], 'dy': size[1], 'dz': size[2]}
dt = 0.01
position = {'x': position[0][1], 'y': position[1][1], 'z': position[2][1]}
boundary = None
velocity_est = {'u': 1, 'v': 0, 'z': 0}
velocity = velocity_est
pressure_est = 100000
pressure = pressure_est
temperature = 300
density, viscosity, conductivity = fluid_properties(pressure, temperature, 'air')

"""


# Let's try this with a small set of numpy arrays and clean it up later. First we define the channel geometry

length_channel = 0.1
height_channel = 0.001
depth_channel = 0.002
t = 0
t_end = 10
alpha = 0.1

# Now the grid should be defined by x, y, and z position at each location in an array
n_x = 11
n_y = 6
n_z = 5
dt = 0.1

size = (n_z, n_y, n_x)
dx = length_channel / (n_x - 1)
dy = height_channel / (n_y - 1)
dz = depth_channel / (n_z - 1)

position_x = np.array([[np.linspace(0, length_channel, n_x), ]*n_y]*n_z)
position_y = np.array([np.array([np.linspace(height_channel, 0, n_y), ]*n_x).transpose()]*n_z)
position_z = np.array([[np.linspace(0, depth_channel, n_z), ]*n_y]*n_x).transpose()

# Now we need to define some terms that will used in the linear coefficients, assume properties are constant for now

p_prop = 101325
temp_prop = 300
fluid_prop = 'air'
density = PropsSI('D', 'P', p_prop, 'T', temp_prop, fluid_prop)
viscosity_dynamic = PropsSI('V', 'P', p_prop, 'T', temp_prop, fluid_prop)
conductivity = PropsSI('L', 'P', p_prop, 'T', temp_prop, fluid_prop)

# now for some guesses!
u = np.ones(size)
v = np.zeros(size)
w = np.zeros(size)

u_t = np.ones(size)
v_t = np.zeros(size)
w_t = np.zeros(size)

u_est = np.ones(size)
v_est = np.zeros(size)
w_est = np.zeros(size)

p = p_prop*np.ones(size)
p_est = p_prop*np.ones(size)

# build the constants we use for solving for velocity and pressure
p_adv_u = (density/(viscosity_dynamic*dx))*u
p_adv_v = (density/(viscosity_dynamic*dy))*v
p_adv_w = (density/(viscosity_dynamic*dz))*w

a_t = density*dx*dy*dz/dt

d_u = u_t*a_t
d_v = v_t*a_t
d_w = w_t*a_t

