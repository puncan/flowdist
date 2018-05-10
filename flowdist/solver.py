import numpy as np


def define_grid(length_channel, number_gridpoints):
    """Returns grid spacing and numpy array containing the axial locations. 1D for now."""
    grid_spacing = length_channel/(number_gridpoints - 1)
    positions_axial = np.linspace(0.0, length_channel, num=number_gridpoints)
    return grid_spacing, positions_axial
