import numpy as np
import yaml
from CoolProp.CoolProp import PropsSI as si
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def animation_init():
    line.set_ydata([np.nan] * len(surface_temperature[0, :]))
    return line,


def animate(i):
    line.set_ydata(surface_temperature[i, :])
    return line,


def get_interface_pressure_and_velocity(inputs, velocity, interface_fluid_temperature, nozzle_outlet_height):

    density = si('D', 'T', interface_fluid_temperature, 'P', inputs['outlet_pressure'], inputs['fluid'])
    viscosity = si('V', 'T', interface_fluid_temperature, 'P', inputs['outlet_pressure'], inputs['fluid'])

    interface_pressure = 0
    res = 2 * inputs['tolerance']

    while res > inputs['tolerance']:
        reynolds = density * velocity * inputs['channel_hydraulic_diameter'] / viscosity
        effective_reynolds = density * velocity * inputs['effective_diameter'] / viscosity

        friction_factor = (1 / (-1.8 *
                                np.log((6.9 / effective_reynolds) +
                                       ((inputs['channel_roughness'] /
                                         inputs['effective_diameter']) / 3.7) ** 1.11))) ** 2

        interface_pressure = inputs['inlet_pressure'] - (density * friction_factor * inputs['channel_length'] *
                                                         velocity ** 2) / (2 * inputs['channel_hydraulic_diameter'])

        nozzle_outlet_flow_area = inputs['channel_width'] * nozzle_outlet_height
        nozzle_hydraulic_diameter = 4 * nozzle_outlet_flow_area / (2 * (inputs['channel_width'] + nozzle_outlet_height))
        beta = nozzle_hydraulic_diameter / inputs['channel_hydraulic_diameter']
        if 0.2 > beta > 0.8:
            return "(nozzle hydraulic diameter/channel hydraulic diamter) should be between 0.2 and 0.8 but is " + \
                   str(beta)

        drag_coefficient = 0.9965 - 0.00653 * np.sqrt(beta) * np.sqrt((10 ** 6) / reynolds)
        alpha = drag_coefficient / np.sqrt(1 - (beta ** 4))
        velocity_new = alpha * nozzle_outlet_flow_area * np.sqrt(2 * (interface_pressure - inputs['outlet_pressure']) /
                                                                 density) / inputs['channel_flow_area']
        res = np.abs((velocity_new - velocity) / velocity_new)
        velocity = (velocity_new + velocity) / 2

    return interface_pressure, velocity


def get_fluid_temperature(inputs, location_along_channel, fluid_temperature_left, fluid_temperature_last,
                          interface_pressure, velocity):

    fluid_pressure = inputs['inlet_pressure'] - ((inputs['inlet_pressure'] - interface_pressure) /
                                                 inputs['channel_length']) * location_along_channel
    density = si('D', 'T', fluid_temperature_last, 'P', fluid_pressure, inputs['fluid'])
    heat_capacity = si('C', 'T', fluid_temperature_last, 'P', fluid_pressure, inputs['fluid'])

    t_left_coefficient = density * velocity * heat_capacity * inputs['channel_flow_area']
    t_last_coefficient = inputs['cell_length'] * inputs['channel_flow_area'] * heat_capacity / inputs['time_step']

    fluid_temperature = (t_left_coefficient*fluid_temperature_left + t_last_coefficient*fluid_temperature_last +
                         inputs['source_term'])/(t_left_coefficient + t_last_coefficient)

    return fluid_temperature


def get_strip_temperature(inputs, strip_temperature_last, interface_fluid_temperature, interface_fluid_pressure,
                          velocity, nozzle_outlet_height):

    density = si('D', 'T', interface_fluid_temperature, 'P', interface_fluid_pressure, inputs['fluid'])
    heat_capacity = si('C', 'T', interface_fluid_temperature, 'P', interface_fluid_pressure, inputs['fluid'])
    viscosity = si('V', 'T', interface_fluid_temperature, 'P', interface_fluid_pressure, inputs['fluid'])
    conductivity = si('L', 'T', interface_fluid_temperature, 'P', interface_fluid_pressure, inputs['fluid'])

    # nozzle outlet hydraulic diameter
    nozzle_outlet_flow_area = nozzle_outlet_height * inputs['channel_width']
    nozzle_outlet_flow_perimeter = 2 * (nozzle_outlet_height + inputs['channel_width'])
    nozzle_outlet_hydraulic_diameter = 4 * nozzle_outlet_flow_area / nozzle_outlet_flow_perimeter

    mean_hydraulic_diameter = (inputs['channel_hydraulic_diameter'] + nozzle_outlet_hydraulic_diameter) / 2
    nozzle_outlet_velocity = velocity * inputs['channel_height'] / nozzle_outlet_height
    mean_velocity = (velocity + nozzle_outlet_velocity) / 2

    mean_reynolds_number = density * mean_velocity * mean_hydraulic_diameter / viscosity
    prandtl_number = heat_capacity * viscosity / conductivity

    # assuming turbulent flow, we use the Dittus-Boelter equation
    if interface_fluid_temperature >= strip_temperature_last:
        mean_nusselt_number = 0.0265 * (mean_reynolds_number**(4/5)) * (prandtl_number**0.3)
    else:
        mean_nusselt_number = 0.0243 * (mean_reynolds_number**(4/5)) * (prandtl_number**0.4)

    heat_transfer_coefficient = conductivity * mean_nusselt_number / mean_hydraulic_diameter

    # using lumped capacitance assumption
    exp_arg = - heat_transfer_coefficient * inputs['time_step'] / \
              (inputs['strip_density'] * inputs['strip_thickness'] * inputs['strip_heat_capacity'])

    return (strip_temperature_last - interface_fluid_temperature) * np.exp(exp_arg) + interface_fluid_temperature


def get_nozzle_outlet_height(inputs, nozzle_outlet_height, strip_temperature):

    deflection = (inputs['strip_coefficient'] * (inputs['strip_length'] ** 2) *
                  (strip_temperature - inputs['dead_state_strip_temperature'])) / inputs['strip_thickness']

    return nozzle_outlet_height + 2 * deflection


def get_surface_temperature(inputs, fluid_temperature, interface_pressure, velocity, locations_along_channel):

    fluid_pressure = inputs['inlet_pressure'] - ((inputs['inlet_pressure'] -
                                                  interface_pressure) / inputs['channel_length']) * \
                                                locations_along_channel

    density = si('D', 'T', fluid_temperature, 'P', fluid_pressure, inputs['fluid'])
    heat_capacity = si('C', 'T', fluid_temperature, 'P', fluid_pressure, inputs['fluid'])
    viscosity = si('V', 'T', fluid_temperature, 'P', fluid_pressure, inputs['fluid'])
    conductivity = si('L', 'T', fluid_temperature, 'P', fluid_pressure, inputs['fluid'])

    reynolds_number = density * velocity * inputs['channel_hydraulic_diameter'] / viscosity
    prandtl_number = heat_capacity * viscosity / conductivity

    # assuming turbulent flow, we use the Dittus-Boelter equation
    nusselt_number = 0.0243 * (reynolds_number ** (4 / 5)) * (prandtl_number ** 0.4)

    heat_transfer_coefficient = conductivity * nusselt_number / inputs['channel_hydraulic_diameter']
    return fluid_temperature + (inputs['input_heat_flux'] / heat_transfer_coefficient)

# get input values and convert to floats except for fluid string
with open('input.yaml', 'r') as f:
    inputs = yaml.safe_load(f)

for key, val in inputs.items():
    if key != 'fluid':
        inputs[key] = float(val)

# add derived constants to input dictionary
inputs['channel_flow_area'] = inputs['channel_width'] * inputs['channel_height']
channel_flow_perimeter = 2 * (inputs['channel_width'] + inputs['channel_height'])
channel_aspect_ratio_f_reynolds = min(inputs['channel_height'] / inputs['channel_width'],
                                         inputs['channel_width'] / inputs['channel_height'])
f_reynolds = 96 * (1 - 1.3553 * channel_aspect_ratio_f_reynolds + 1.9467 * (channel_aspect_ratio_f_reynolds ** 2) -
                   1.7012 * (channel_aspect_ratio_f_reynolds ** 3) + 0.9564 * (channel_aspect_ratio_f_reynolds ** 4) -
                   0.2537 * (channel_aspect_ratio_f_reynolds ** 5))

inputs['channel_hydraulic_diameter'] = 4 * inputs['channel_flow_area'] / channel_flow_perimeter
inputs['effective_diameter'] = (64 / f_reynolds) * inputs['channel_hydraulic_diameter']
inputs['cell_length'] = inputs['channel_length'] / inputs['number_of_nodes']
inputs['time_step'] = inputs['total_time_of_simulation'] / inputs['number_of_time_steps']
inputs['source_term'] = inputs['input_heat_flux'] * inputs['cell_length'] * inputs['channel_width']

location_along_channel = np.linspace(0, inputs['channel_length'], inputs['number_of_nodes'])

# initializations
fluid_temperature = np.zeros((2, int(inputs['number_of_nodes'])))
fluid_temperature[0, :] = inputs['initial_fluid_temperature']
fluid_temperature[1, 0] = inputs['inlet_temperature']
strip_temperature = inputs['initial_fluid_temperature']
velocity = inputs['velocity']
interface_fluid_temperature = inputs['initial_fluid_temperature']
surface_temperature = np.zeros((int(inputs['number_of_time_steps'] + 1), int(inputs['number_of_nodes'])))
nozzle_outlet_height = inputs['dead_state_nozzle_outlet_height']

time = 0
j = 0
while time < inputs['total_time_of_simulation']:
    interface_pressure, velocity = get_interface_pressure_and_velocity(inputs, velocity, interface_fluid_temperature,
                                                                       nozzle_outlet_height)
    for i in np.arange(1, int(inputs['number_of_nodes'])):
        fluid_temperature[1, i] = get_fluid_temperature(inputs, location_along_channel[i], fluid_temperature[1, i - 1],
                                                        fluid_temperature[0, i], interface_pressure, velocity)

    interface_fluid_temperature = fluid_temperature[1, -1]
    fluid_temperature[0, :] = fluid_temperature[1, :]
    fluid_temperature[:, 0] = inputs['inlet_temperature']

    strip_temperature = get_strip_temperature(inputs, strip_temperature, interface_fluid_temperature,
                                              interface_pressure, velocity, nozzle_outlet_height)

    nozzle_outlet_height = get_nozzle_outlet_height(inputs, nozzle_outlet_height, strip_temperature)

    surface_temperature[j, :] = get_surface_temperature(inputs, fluid_temperature[0, :], interface_pressure, velocity,
                                                        location_along_channel)
    time += inputs['time_step']
    j += 1

fig, ax = plt.subplots()
line, = ax.plot(location_along_channel, surface_temperature[0, :], color='black')
ax.set(xlim=(0, inputs['channel_length']),
       ylim=(np.amin(surface_temperature[1:-1, :]) - 5, np.amax(surface_temperature) + 5))
ax.set_xlabel('Distance Along Channel (m)')
ax.set_ylabel('Channel Surface Temperature (K)')
interval = 1000 * inputs['time_step']
ax.grid(True)
ani = animation.FuncAnimation(fig, animate, frames=int(inputs['number_of_time_steps']), init_func=animation_init,
                              interval=int(interval), blit=True)

plt.show()
