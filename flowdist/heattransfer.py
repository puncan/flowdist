import numpy as np


def initial_conditions(temperature, grid):
    """The initial temperature distribution is defined by a scalar or function of x and y.
    The grid should be a numpy array"""

    try:
        return temperature(grid)
    except TypeError:
        return temperature * np.ones(np.shape(grid))



