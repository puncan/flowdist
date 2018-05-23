import numpy as np
from CoolProp.CoolProp import PropsSI
#from fluidmechanics import FiniteVolume
import matplotlib.pyplot as plt


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


# Let's try this with a small set of numpy arrays and clean it up later. First we define the boundary conditions.
pressure_in = 101325 + 19.2  # Pa
pressure_out = 101325  # Pa
velocity_walls = 0

# then the channel geometry
length_channel = 1
height_channel = 0.005
depth_channel = 0.01

# the time over which we simulate
t = 0
t_end = 10

# Now the grid should be defined in space and time as well as the under-relaxation factors
n_x = 20
n_y = 11
n_z = 11
dt = 0.1
dx = length_channel / (n_x - 1)
dy = height_channel / (n_y - 1)
dz = depth_channel / (n_z - 1)

alpha_vel = 0.7
alpha_vel_est = 0.7
alpha_vel_corr = 0.7
alpha_p = 0.7
alpha_p_corr = 0.1
tolerance = 10**(-20)
size = (n_z, n_y, n_x)

# the positions of each node can now be represented
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

# now for some guesses! first we guess the velocities at the staggered grid points
residual_vel = 2*tolerance
residual_p = 2*tolerance
u = np.pad(0.25*np.ones((1, 1, n_x)), ((n_z//2, n_z//2), (n_y//2, n_y//2), (0, 0)), mode='linear_ramp', end_values=(0, 0))
v = 0.0*np.ones(size)
w = 0.0*np.ones(size)
p_est = np.pad(pressure_in*np.ones(
    (n_z, n_y, 1)), ((0, 0), (0, 0), (0, n_x - 1)), mode='linear_ramp', end_values=pressure_out)

plt.plot(u[0:, n_y//2, n_x//2])

u_old = np.pad(np.ones((1, 1, n_x)), ((n_z//2, n_z//2), (n_y//2, n_y//2), (0, 0)), mode='linear_ramp', end_values=(0, 0))
v_old = np.ones(size)
w_old = np.ones(size)

# the velocity field will be solved for based on the estimated pressure field, we need an initial guess of the
# velocity field on all the neighboring nodes. Let's just make copies of the velocity field we established earlier
# for the coefficients
u_e_est = np.pad(0.25*np.ones((1, 1, n_x)), ((n_z//2, n_z//2), (n_y//2, n_y//2), (0, 0)), mode='linear_ramp', end_values=(0, 0))
u_w_est = np.pad(0.25*np.ones((1, 1, n_x)), ((n_z//2, n_z//2), (n_y//2, n_y//2), (0, 0)), mode='linear_ramp', end_values=(0, 0))
u_n_est = np.pad(0.25*np.ones((1, 1, n_x)), ((n_z//2, n_z//2), (n_y//2, n_y//2), (0, 0)), mode='linear_ramp', end_values=(0, 0))
u_s_est = np.pad(0.25*np.ones((1, 1, n_x)), ((n_z//2, n_z//2), (n_y//2, n_y//2), (0, 0)), mode='linear_ramp', end_values=(0, 0))
u_f_est = np.pad(0.25*np.ones((1, 1, n_x)), ((n_z//2, n_z//2), (n_y//2, n_y//2), (0, 0)), mode='linear_ramp', end_values=(0, 0))
u_b_est = np.pad(0.25*np.ones((1, 1, n_x)), ((n_z//2, n_z//2), (n_y//2, n_y//2), (0, 0)), mode='linear_ramp', end_values=(0, 0))

v_e_est = 0*np.ones(size)
v_w_est = 0*np.ones(size)
v_n_est = 0*np.ones(size)
v_s_est = 0*np.ones(size)
v_f_est = 0*np.ones(size)
v_b_est = 0*np.ones(size)

w_e_est = 0*np.ones(size)
w_w_est = 0*np.ones(size)
w_n_est = 0*np.ones(size)
w_s_est = 0*np.ones(size)
w_f_est = 0*np.ones(size)
w_b_est = 0*np.ones(size)

# Then we'll need the main nodes for the first residual calculation

u_est = np.pad(0.25*np.ones((1, 1, n_x)), ((n_z//2, n_z//2), (n_y//2, n_y//2), (0, 0)), mode='linear_ramp', end_values=(0, 0))
v_est = 0*np.ones(size)
w_est = 0*np.ones(size)
u_est_old = np.pad(0.25*np.ones((1, 1, n_x)), ((n_z//2, n_z//2), (n_y//2, n_y//2), (0, 0)), mode='linear_ramp', end_values=(0, 0))
v_est_old = 0*np.ones(size)
w_est_old = 0*np.ones(size)

# velocities at the previous time step
u_t = 0.1 * np.ones(size)
v_t = 0.0 * np.ones(size)
w_t = 0.0 * np.ones(size)

# We also need guess values for the pressure correction equation
f_p_corr = 0
p_corr_old = f_p_corr*np.ones(size)
p_corr = 2*p_corr_old
p_corr_e = f_p_corr*np.ones(size)
p_corr_w = f_p_corr*np.ones(size)
p_corr_n = f_p_corr*np.ones(size)
p_corr_s = f_p_corr*np.ones(size)
p_corr_f = f_p_corr*np.ones(size)
p_corr_b = f_p_corr*np.ones(size)

u_max, v_max, w_max, p_max = (0, 0, 0, 0)
u_min, v_min, w_min, p_min = (0, 0, 0, 0)

while residual_vel > tolerance:

    # then we guess the pressure field at the regular grid points
    p_est_w = np.pad(p_est[:, :, 0:-1].copy(), ((0, 0), (0, 0), (1, 0)), mode='constant', constant_values=pressure_in)
    p_est_s = np.pad(p_est[:, 1:, :].copy(), ((0, 0), (0, 1), (0, 0)), 'edge')
    p_est_b = np.pad(p_est[1:, :, :].copy(), ((0, 1), (0, 0), (0, 0)), 'edge')

    # we need to get the velocities at the regular grid points based on the interpolation of the staggered grid values
    # these values will be handy for the coefficients coming up
    u_eo2 = (np.pad(u[:, :, 1:].copy(), ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=0) + u)/2
    u_wo2 = (np.pad(u[:, :, 0: -1].copy(), ((0, 0), (0, 0), (1, 0)), mode='constant', constant_values=0) + u)/2
    u_no2 = (np.pad(u[:, 0: -1, :].copy(), ((0, 0), (1, 0), (0, 0)), mode='constant', constant_values=0) + u)/2
    u_so2 = (np.pad(u[:, 1:, :].copy(), ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0) + u)/2
    u_fo2 = (np.pad(u[0: -1, :, :].copy(), ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0) + u)/2
    u_bo2 = (np.pad(u[1:, :, :].copy(), ((0, 1), (0, 0), (0, 0)), mode='constant', constant_values=0) + u)/2

    v_eo2 = (np.pad(v[:, :, 1:].copy(), ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=0) + v)/2
    v_wo2 = (np.pad(v[:, :, 0: -1].copy(), ((0, 0), (0, 0), (1, 0)), mode='constant', constant_values=0) + v)/2
    v_no2 = (np.pad(v[:, 0: -1, :].copy(), ((0, 0), (1, 0), (0, 0)), mode='constant', constant_values=0) + v)/2
    v_so2 = (np.pad(v[:, 1:, :].copy(), ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0) + v)/2
    v_fo2 = (np.pad(v[0: -1, :, :].copy(), ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0) + v)/2
    v_bo2 = (np.pad(v[1:, :, :].copy(), ((0, 1), (0, 0), (0, 0)), mode='constant', constant_values=0) + v)/2

    w_eo2 = (np.pad(w[:, :, 1:].copy(), ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=0) + w)/2
    w_wo2 = (np.pad(w[:, :, 0: -1].copy(), ((0, 0), (0, 0), (1, 0)), mode='constant', constant_values=0) + w)/2
    w_no2 = (np.pad(w[:, 0: -1, :].copy(), ((0, 0), (1, 0), (0, 0)), mode='constant', constant_values=0) + w)/2
    w_so2 = (np.pad(w[:, 1:, :].copy(), ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0) + w)/2
    w_fo2 = (np.pad(w[0: -1, :, :].copy(), ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0) + w)/2
    w_bo2 = (np.pad(w[1:, :, :].copy(), ((0, 1), (0, 0), (0, 0)), mode='constant', constant_values=0) + w)/2

    # build the constants we use for solving for velocity and pressure
    p_adv_u_eo2 = (density/(viscosity_dynamic*dx))*u_eo2
    p_adv_u_wo2 = (density/(viscosity_dynamic*dx))*u_wo2
    p_adv_u_no2 = (density/(viscosity_dynamic*dx))*u_no2
    p_adv_u_so2 = (density/(viscosity_dynamic*dx))*u_so2
    p_adv_u_fo2 = (density/(viscosity_dynamic*dx))*u_fo2
    p_adv_u_bo2 = (density/(viscosity_dynamic*dx))*u_bo2

    p_adv_v_eo2 = (density/(viscosity_dynamic*dy))*v_eo2
    p_adv_v_wo2 = (density/(viscosity_dynamic*dy))*v_wo2
    p_adv_v_no2 = (density/(viscosity_dynamic*dy))*v_no2
    p_adv_v_so2 = (density/(viscosity_dynamic*dy))*v_so2
    p_adv_v_fo2 = (density/(viscosity_dynamic*dy))*v_fo2
    p_adv_v_bo2 = (density/(viscosity_dynamic*dy))*v_bo2

    p_adv_w_eo2 = (density/(viscosity_dynamic*dz))*w_eo2
    p_adv_w_wo2 = (density/(viscosity_dynamic*dz))*w_wo2
    p_adv_w_no2 = (density/(viscosity_dynamic*dz))*w_no2
    p_adv_w_so2 = (density/(viscosity_dynamic*dz))*w_so2
    p_adv_w_fo2 = (density/(viscosity_dynamic*dz))*w_fo2
    p_adv_w_bo2 = (density/(viscosity_dynamic*dz))*w_bo2

    # time dependent source terms, for the first pass, these will be zero
    a_t = 0 # density*dx*dy*dz/dt
    d_u = 0 # u_t*a_t
    d_v = 0 # v_t*a_t
    d_w = 0 # w_t*a_t

    # the a coefficients for the x-momentum are
    a_x_e = (viscosity_dynamic/dx)*np.maximum(0, 1-(np.absolute(p_adv_u_eo2)/2)) + np.maximum(0, -(density*u_eo2))
    a_x_w = (viscosity_dynamic/dx)*np.maximum(0, 1+(np.absolute(p_adv_u_wo2)/2)) + np.maximum(0, (density*u_wo2))
    a_x_n = (viscosity_dynamic/dy)*np.maximum(0, 1-(np.absolute(p_adv_u_no2)/2)) + np.maximum(0, -(density*v_no2))
    a_x_s = (viscosity_dynamic/dy)*np.maximum(0, 1+(np.absolute(p_adv_u_so2)/2)) + np.maximum(0, (density*v_so2))
    a_x_f = (viscosity_dynamic/dz)*np.maximum(0, 1-(np.absolute(p_adv_u_fo2)/2)) + np.maximum(0, -(density*w_fo2))
    a_x_b = (viscosity_dynamic/dz)*np.maximum(0, 1+(np.absolute(p_adv_u_fo2)/2)) + np.maximum(0, (density*w_bo2))
    a_x = a_x_e + a_x_w + a_x_n + a_x_s + a_x_f + a_x_b + a_t

    a_y_e = (viscosity_dynamic/dx)*np.maximum(0, 1-(np.absolute(p_adv_v_eo2)/2)) + np.maximum(0, -(density*u_eo2))
    a_y_w = (viscosity_dynamic/dx)*np.maximum(0, 1+(np.absolute(p_adv_v_wo2)/2)) + np.maximum(0, (density*u_wo2))
    a_y_n = (viscosity_dynamic/dy)*np.maximum(0, 1-(np.absolute(p_adv_v_no2)/2)) + np.maximum(0, -(density*v_no2))
    a_y_s = (viscosity_dynamic/dy)*np.maximum(0, 1+(np.absolute(p_adv_v_so2)/2)) + np.maximum(0, (density*v_so2))
    a_y_f = (viscosity_dynamic/dz)*np.maximum(0, 1-(np.absolute(p_adv_v_fo2)/2)) + np.maximum(0, -(density*w_fo2))
    a_y_b = (viscosity_dynamic/dz)*np.maximum(0, 1+(np.absolute(p_adv_v_fo2)/2)) + np.maximum(0, (density*w_bo2))
    a_y = a_y_e + a_y_w + a_y_n + a_y_s + a_y_f + a_y_b + a_t

    a_z_e = (viscosity_dynamic/dx)*np.maximum(0, 1-(np.absolute(p_adv_w_eo2)/2)) + np.maximum(0, -(density*u_eo2))
    a_z_w = (viscosity_dynamic/dx)*np.maximum(0, 1+(np.absolute(p_adv_w_wo2)/2)) + np.maximum(0, (density*u_wo2))
    a_z_n = (viscosity_dynamic/dy)*np.maximum(0, 1-(np.absolute(p_adv_w_no2)/2)) + np.maximum(0, -(density*v_no2))
    a_z_s = (viscosity_dynamic/dy)*np.maximum(0, 1+(np.absolute(p_adv_w_so2)/2)) + np.maximum(0, (density*v_so2))
    a_z_f = (viscosity_dynamic/dz)*np.maximum(0, 1-(np.absolute(p_adv_w_fo2)/2)) + np.maximum(0, -(density*w_fo2))
    a_z_b = (viscosity_dynamic/dz)*np.maximum(0, 1+(np.absolute(p_adv_w_fo2)/2)) + np.maximum(0, (density*w_bo2))
    a_z = a_z_e + a_z_w + a_z_n + a_z_s + a_z_f + a_z_b + a_t

    # we will start with the Jacobi method from Chapter 7 of "An Introduction to Computational Fluid
    # Dynamics: The Finite Volume Method" by H K Versteeg and W Malalasekera
    # Eventually, we'll do some form of the Gauss-Seidel method but one step at a time, so to speak
    residual_vel_est = 2*tolerance

    while residual_vel_est > tolerance:
        u_est = (1 - alpha_vel_est)*u_est_old + alpha_vel_est*(
            a_x_e*u_e_est + a_x_w*u_w_est + a_x_n*u_n_est + a_x_s*u_s_est + a_x_f*u_f_est + a_x_b*u_b_est + d_u + (
                p_est_w - p_est)*dy*dz)/a_x

        v_est = (1 - alpha_vel_est)*v_est_old + alpha_vel_est*(
            a_y_e*v_e_est + a_y_w*v_w_est + a_y_n*v_n_est + a_y_s*v_s_est + a_y_f*v_f_est + a_y_b*v_b_est + d_v + (
                p_est_s - p_est)*dx*dz)/a_y

        w_est = (1 - alpha_vel_est)*w_est_old + alpha_vel_est*(
            a_z_e*w_e_est + a_z_w*w_w_est + a_z_n*w_n_est + a_z_s*w_s_est + a_z_f*w_f_est + a_z_b*w_b_est + d_w + (
                p_est_b - p_est)*dx*dy)/a_z

    # let's calculate the maximum residual for all the solved velocities
        residual_vel_est = np.amax(np.maximum(np.maximum(np.absolute(u_est - u_est_old), np.absolute(v_est - v_est_old)
                ), np.absolute(w_est - w_est_old)))
        #print('velocity_est residual = ', residual_vel_est)

    # now we need to update the neighboring nodes for the next iteration
        u_e_est = np.pad(u_est[:, :, 1:].copy(), ((0, 0), (0, 0), (0, 1)), mode='edge')
        u_w_est = np.pad(u_est[:, :, 0: -1].copy(), ((0, 0), (0, 0), (1, 0)), mode='edge')
        u_n_est = np.pad(u_est[:, 0: -1, :].copy(), ((0, 0), (1, 0), (0, 0)), mode='constant', constant_values=0)
        u_s_est = np.pad(u_est[:, 1:, :].copy(), ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)
        u_f_est = np.pad(u_est[0: -1, :, :].copy(), ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
        u_b_est = np.pad(u_est[1:, :, :].copy(), ((0, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)

        v_e_est = np.pad(v_est[:, :, 1:].copy(), ((0, 0), (0, 0), (0, 1)), mode='edge')
        v_w_est = np.pad(v_est[:, :, 0: -1].copy(), ((0, 0), (0, 0), (1, 0)), mode='edge')
        v_n_est = np.pad(v_est[:, 0: -1, :].copy(), ((0, 0), (1, 0), (0, 0)), mode='constant', constant_values=0)
        v_s_est = np.pad(v_est[:, 1:, :].copy(), ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)
        v_f_est = np.pad(v_est[0: -1, :, :].copy(), ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
        v_b_est = np.pad(v_est[1:, :, :].copy(), ((0, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)

        w_e_est = np.pad(w_est[:, :, 1:].copy(), ((0, 0), (0, 0), (0, 1)), mode='edge')
        w_w_est = np.pad(w_est[:, :, 0: -1].copy(), ((0, 0), (0, 0), (1, 0)), mode='edge')
        w_n_est = np.pad(w_est[:, 0: -1, :].copy(), ((0, 0), (1, 0), (0, 0)), mode='constant', constant_values=0)
        w_s_est = np.pad(w_est[:, 1:, :].copy(), ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)
        w_f_est = np.pad(w_est[0: -1, :, :].copy(), ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
        w_b_est = np.pad(w_est[1:, :, :].copy(), ((0, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)

        u_est_old = u_est
        v_est_old = v_est
        w_est_old = w_est

    # With this velocity field, we can start the pressure correction process. Start by getting the coefficients together
    a_p_e = (density*(dy*dz)**2)/a_x
    a_p_w = (density*(dy*dz)**2)/a_x_w
    a_p_n = (density*(dz*dx)**2)/a_y
    a_p_s = (density*(dz*dx)**2)/a_y_s
    a_p_f = (density*(dx*dy)**2)/a_z
    a_p_b = (density*(dx*dy)**2)/a_z_b
    a_p = a_p_e + a_p_w + a_p_n + a_p_s + a_p_f + a_p_b

    d_p = (density*u_e_est - density*u_est)*dy*dz + (density*v_s_est - density*v_est)*dz*dx + (
            density*w_b_est - density*w_est)*dx*dy  # + ((density - density)*dx*dy*dz/dt)
    """
    residual_p_corr = 2*tolerance
    while residual_p_corr > 10**(-10):
        # With these coefficients we can solve for the pressure correction term
        p_corr = (1 - alpha_p_corr)*p_corr_old + alpha_p_corr*(
            a_p_e*p_corr_e + a_p_w*p_corr_w + a_p_n*p_corr_n + a_p_s*p_corr_s + a_p_f*p_corr_f + a_p_b*p_corr_b + d_p)/a_p
        # print(p_corr)
        # now lets calculate the maximum residual for the pressure correction
        residual_p_corr = np.amax(np.absolute(p_corr - p_corr_old))
        # print('pressure correction residual = ', residual_p_corr)
        # as was done with the velocities, let's update the neighboring nodes
        p_corr_e = np.pad(p_corr[:, :, 1:].copy(), ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=0)
        p_corr_w = np.pad(p_corr[:, :, 0:-1].copy(), ((0, 0), (0, 0), (1, 0)), mode='constant', constant_values=0)
        p_corr_n = np.pad(p_corr[:, 0:-1, :].copy(), ((0, 0), (1, 0), (0, 0)), mode='constant', constant_values=0)
        p_corr_s = np.pad(p_corr[:, 1:, :].copy(), ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)
        p_corr_f = np.pad(p_corr[0:-1, :, :].copy(), ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
        p_corr_b = np.pad(p_corr[1:, :, :].copy(), ((0, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)

        p_corr_old = p_corr
    """
    # we can update the pressure field now
    p = p_est # + alpha_p*p_corr
    p_est = p
    residual_p = np.amax(np.absolute(alpha_p*p_corr))

    # np.save('p_corr', p_corr)
    # now we can use the pressure corrections to correct the velocity field
    u = u_est + alpha_vel*(p_corr_w - p_corr)*dy*dz/a_x
    v = v_est + alpha_vel*(p_corr_s - p_corr)*dx*dz/a_y
    w = w_est + alpha_vel*(p_corr_b - p_corr)*dx*dy/a_z
    residual_vel = np.amax(np.maximum(np.maximum(np.absolute(u - u_old), np.absolute(v - v_old)), np.absolute(
        w - w_old)))

    u_old = u
    v_old = v
    w_old = w

    u_max = np.amax(u)
    u_min = np.amin(u)
    v_max = np.amax(v)
    v_min = np.amin(v)
    w_max = np.amax(w)
    w_min = np.amin(w)
    p_max = np.amax(p)
    p_min = np.amin(p)

print('Residual_velocity = ' + '%e' % residual_vel, 'Residual_pressure = ' + '%e' % residual_p)
print('Max_velocities = ', u_max, v_max, w_max, 'Max_pressure = ', p_max)
print('Min_velocities = ', u_min, v_min, w_min, 'Min_pressure = ', p_min)

plt.plot(u[0:, n_y//2, n_x//2])
plt.show()