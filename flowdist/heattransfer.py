import numpy as np


def initial_conditions(temperature, grid):
    """The initial temperature distribution in degrees kelvin is defined by a scalar or function of x and y.
    The grid must be a numpy array"""

    em1 = 'Temperature must be in degrees Kelvin'
    em2 = 'Temperature must be scalar or function of x and y coordinates and grid must be numpy array'

    try:
        t = temperature(grid)
        if np.min(t) < 250:
            return em1
        else:
            return t
    except TypeError:
        try:
            t = temperature * np.ones(np.shape(grid))
            if np.min(t) < 250:
                return em1
            else:
                return t
        except TypeError:
            return em2
