import numpy as np
from CoolProp.CoolProp import PropsSI as si
# import matplotlib.pyplot as plt
# import pint


def get_interface_pressure_and_velocity(guess_velocity, inlet_pressure, outlet_pressure, interface_fluid_temperature,
                                        channel_dimensions, nozzle_outlet_height, fluid, tolerance):
    
    channel_length, channel_height, channel_width, channel_roughness = channel_dimensions
    
    channel_aspect_ratio = min(channel_height / channel_width, channel_width / channel_height)
    hydraulic_diameter = 4 * channel_width * channel_height / (2 * (channel_width + channel_height))

    f_reynolds = 96 * (1 - 1.3553 * channel_aspect_ratio + 1.9467 * (channel_aspect_ratio ** 2) -
                       1.7012 * (channel_aspect_ratio ** 3) + 0.9564 * (channel_aspect_ratio ** 4) -
                       0.2537 * (channel_aspect_ratio ** 5))

    density = si('D', 'T', interface_fluid_temperature, 'P', outlet_pressure, fluid)
    viscosity = si('V', 'T', interface_fluid_temperature, 'P', outlet_pressure, fluid)

    velocity = guess_velocity
    interface_pressure = outlet_pressure
    res = 2 * tolerance

    while res > tolerance:
        reynolds = density * velocity * hydraulic_diameter / viscosity
        effective_diameter = (64 / f_reynolds) * hydraulic_diameter
        effective_reynolds = density * velocity * effective_diameter / viscosity

        friction_factor = (1 / (-1.8 * np.log((6.9 / effective_reynolds) +
                                              ((channel_roughness / effective_diameter) / 3.7) ** 1.11))) ** 2

        interface_pressure = inlet_pressure - (density * friction_factor * channel_length * velocity ** 2) / \
                                              (2 * hydraulic_diameter)

        nozzle_outlet_flow_area = channel_width * nozzle_outlet_height
        nozzle_hydraulic_diameter = 4 * nozzle_outlet_flow_area / (2 * (channel_width + nozzle_outlet_height))
        beta = nozzle_hydraulic_diameter / hydraulic_diameter
        if 0.2 > beta > 0.8:
            return "(nozzle hydraulic diameter/channel hydraulic diamter) should be between 0.2 and 0.8 but is " + \
                   str(beta)

        drag_coefficient = 0.9965 - 0.00653 * np.sqrt(beta) * np.sqrt((10 ** 6) / reynolds)
        alpha = drag_coefficient / np.sqrt(1 - (beta ** 4))
        velocity_new = alpha * nozzle_outlet_flow_area * np.sqrt(2 * (interface_pressure - outlet_pressure) / density) / \
            (channel_width * channel_height)
        res = np.abs((velocity_new - velocity) / velocity_new)
        velocity = (velocity_new + velocity) / 2

    return interface_pressure, velocity


def get_fluid_temperature(fluid_temperature_left, fluid_temperature_last, inlet_pressure, interface_pressure, heat_flux,
                          velocity, channel_dimensions, cell_length, location_along_channel, time_step):

    channel_length, channel_height, channel_width, channel_roughness = channel_dimensions
    fluid_pressure = inlet_pressure - ((inlet_pressure - interface_pressure)/channel_length)*location_along_channel
    density = si('D', 'T', fluid_temperature_last, 'P', fluid_pressure, fluid)
    heat_capacity = si('C', 'T', fluid_temperature_last, 'P', fluid_temperature_last, fluid)

    t_left_coefficient = density*velocity*heat_capacity*channel_height*channel_width
    t_last_coefficient = cell_length*channel_height*channel_width*heat_capacity/time_step
    source_term = heat_flux*cell_length*channel_width

    fluid_temperature = (t_left_coefficient*fluid_temperature_left + t_last_coefficient*fluid_temperature_last +
                         source_term)/(t_left_coefficient + t_last_coefficient)

    return fluid_temperature


def get_strip_temperature(strip_temperature_last, interface_fluid_temperature, interface_fluid_pressure, velocity,
                          channel_dimensions, nozzle_strip_dimensions, nozzle_outlet_height, time_step, fluid):

    channel_length, channel_height, channel_width, channel_roughness = channel_dimensions
    dead_state_nozzle_outlet_height, strip_length, strip_thickness, strip_coefficient, dead_state_strip_temperature = \
    nozzle_strip_dimensions

    density = si('D', 'T', interface_fluid_temperature, 'P', interface_fluid_pressure, fluid)
    heat_capacity = si('C', 'T', interface_fluid_temperature, 'P', interface_fluid_pressure, fluid)
    viscosity = si('V', 'T', interface_fluid_temperature, 'P', interface_fluid_pressure, fluid)
    conductivity = si('L', 'T', interface_fluid_temperature, 'P', interface_fluid_pressure, fluid)

    # interface hydraulic diameter
    interface_flow_area = channel_height * channel_width
    interface_flow_perimeter = 2 * (channel_height + channel_width)
    interface_hydraulic_diameter = 4 * interface_flow_area / interface_flow_perimeter

    # nozzle outlet hydraulic diameter
    nozzle_outlet_flow_area = nozzle_outlet_height * channel_width
    nozzle_outlet_flow_perimeter = 2 * (nozzle_outlet_height + channel_width)
    nozzle_outlet_hydraulic_diamter = 4 * nozzle_outlet_flow_area / nozzle_outlet_flow_perimeter

    mean_hydraulic_diameter = (interface_hydraulic_diameter + nozzle_outlet_hydraulic_diamter) / 2
    nozzle_outlet_velocity = velocity * channel_height / nozzle_outlet_height
    mean_velocity = (velocity + nozzle_outlet_velocity) / 2

    mean_reynolds_number = density * mean_velocity * mean_hydraulic_diameter / viscosity
    prandtl_number = heat_capacity * viscosity / conductivity

    # assuming turbulent flow, we use the Dittus-Boelter equation
    if interface_fluid_temperature >= strip_temperature_last:
        mean_nusselt_number = 0.0265 * (mean_reynolds_number**(4/5)) * (prandtl_number**0.3)
    else:
        mean_nusselt_number = 0.0243 * (mean_reynolds_number**(4/5)) * (prandtl_number**0.4)

    heat_transfer_coefficient = conductivity * mean_nusselt_number / mean_hydraulic_diameter

    # assuming half copper, half SS for the bimetallic strip properties
    strip_density = (8960 + 7700) / 2
    strip_heat_capacity = (376.8 + 502) / 2

    # using lumped capacitance assumption
    exp_arg = -heat_transfer_coefficient*time_step/(strip_density * strip_thickness * strip_heat_capacity)
    return (strip_temperature_last - interface_fluid_temperature) * np.exp(exp_arg) + interface_fluid_temperature


def get_nozzle_outlet_height(nozzle_strip_dimensions, nozzle_outlet_height, strip_temperature):

    dead_state_nozzle_outlet_height, strip_length, strip_thickness, strip_coefficient, dead_state_strip_temperature = \
        nozzle_strip_dimensions

    deflection = (strip_coefficient * (strip_length**2) * (strip_temperature - dead_state_strip_temperature)) / \
        strip_thickness

    return nozzle_outlet_height + 2 * deflection


def get_surface_temperature(fluid_temperature, inlet_pressure, interface_pressure, velocity, channel_dimensions,
                            location_along_channel, heat_flux):

    channel_length, channel_height, channel_width, channel_roughness = channel_dimensions

    fluid_pressure = inlet_pressure - ((inlet_pressure - interface_pressure) / channel_length) * location_along_channel

    density = si('D', 'T', fluid_temperature, 'P', fluid_pressure, fluid)
    heat_capacity = si('C', 'T', fluid_temperature, 'P', fluid_pressure, fluid)
    viscosity = si('V', 'T', fluid_temperature, 'P', fluid_pressure, fluid)
    conductivity = si('L', 'T', fluid_temperature, 'P', fluid_pressure, fluid)

    channel_flow_area = channel_width * channel_height
    channel_flow_perimeter = 2 * (channel_width + channel_height)
    hydraulic_diameter = 4 * channel_flow_area / channel_flow_perimeter

    reynolds_number = density * velocity * hydraulic_diameter / viscosity
    prandtl_number = heat_capacity * viscosity / conductivity

    # assuming turbulent flow, we use the Dittus-Boelter equation
    nusselt_number = 0.0243 * (reynolds_number ** (4 / 5)) * (prandtl_number ** 0.4)

    heat_transfer_coefficient = conductivity * nusselt_number / hydraulic_diameter
    return fluid_temperature + (heat_flux / heat_transfer_coefficient)


# inputs
# channel dimensions
channel_length = 0.1
channel_height = 0.001
channel_width = 0.002
channel_roughness = 0.002e-3  # smooth stainless steel, epsilon
channel_dimensions = [channel_length, channel_height, channel_width, channel_roughness]

# nozzle and bimetallic strip dimensions
dead_state_nozzle_outlet_height = 0.0008
strip_length = 0.001
strip_thickness = 0.00025
strip_coefficient = 18e-6
dead_state_strip_temperature = 325
nozzle_strip_dimensions = [dead_state_nozzle_outlet_height, strip_length, strip_thickness, strip_coefficient,
                           dead_state_strip_temperature]
nozzle_outlet_height = dead_state_nozzle_outlet_height

# initial conditions
initial_fluid_temperature = 300

# boundary conditions
inlet_temperature = 300
inlet_pressure = 5050000
outlet_pressure = 5000000

# the minimum and maximum heat fluxes and type of flux distribution, maybe should be a dict
input_heat_flux = 1e6  # ([0, 1e5], 'parabolic')
total_time_of_simulation = 5

# fluid
fluid = 'water'

# model parameters
tolerance = 1e-10
number_of_nodes = 100
number_of_time_steps = 500

# trivial calculations from inputs
cell_length = channel_length/number_of_nodes
location_along_channel = np.linspace(0, channel_length, number_of_nodes)
time_step = total_time_of_simulation/number_of_time_steps

# array initializations
fluid_temperature = np.zeros((2, number_of_nodes))
fluid_temperature[0, :] = initial_fluid_temperature
fluid_temperature[1, 0] = inlet_temperature
heat_flux = input_heat_flux*np.ones(number_of_nodes)
heat_flux[0] = 0.0
strip_temperature = initial_fluid_temperature
velocity = 0.1

time = 0
while time < total_time_of_simulation:
    interface_pressure, velocity = get_interface_pressure_and_velocity(velocity, inlet_pressure, outlet_pressure,
                                                                       initial_fluid_temperature, channel_dimensions,
                                                                       nozzle_outlet_height, fluid, tolerance)

    for i in np.arange(1, number_of_nodes):
        fluid_temperature[1, i] = get_fluid_temperature(fluid_temperature[1, i - 1], fluid_temperature[0, i],
                                                        inlet_pressure, interface_pressure, heat_flux[i], velocity,
                                                        channel_dimensions, cell_length, location_along_channel[i],
                                                        time_step)

    interface_fluid_temperature = fluid_temperature[1, -1]
    fluid_temperature[0, :] = fluid_temperature[1, :]
    fluid_temperature[:, 0] = inlet_temperature

    strip_temperature = get_strip_temperature(strip_temperature, interface_fluid_temperature,
                                              interface_pressure, velocity, channel_dimensions, nozzle_strip_dimensions,
                                              nozzle_outlet_height, time_step, fluid)

    nozzle_outlet_height = get_nozzle_outlet_height(nozzle_strip_dimensions, nozzle_outlet_height, strip_temperature)

    surface_temperature = get_surface_temperature(fluid_temperature[0, :], inlet_pressure, interface_pressure, velocity,
                                                  channel_dimensions, location_along_channel, heat_flux)
    time += time_step
    print(surface_temperature[-1], fluid_temperature[0, -1], strip_temperature, velocity, nozzle_outlet_height, time)
    # print(nozzle_outlet_height)

# plt.plot(location_nodes, temperature_fluid)

#ani = animation.FuncAnimation(fig=fig, func=animate, init_func=init, interval=25, blit=True)
#plt.ylim([295, 480])
# plt.show()"""
