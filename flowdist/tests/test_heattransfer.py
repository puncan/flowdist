from ..heattransfer import initial_conditions
import numpy as np

em1 = 'Temperature must be in degrees Kelvin'
em2 = 'Temperature must be scalar or function of x and y coordinates and grid must be numpy array'

def test_initial_conditions_1():

    def coldtempfunc(grid):
        return 200 - 0.5 * grid

    assert initial_conditions(coldtempfunc, np.array([1, 2, 3])) == em1


def test_initial_conditions_2():

    def goodtempfunc(grid):
        return 300 + 0.5 * grid

    assert np.sum(initial_conditions(goodtempfunc, np.array([1, 2, 3]))) == np.sum(np.array([300.5, 301, 301.5]))


def test_initial_conditions_3():
    assert initial_conditions(100, np.array([1, 2, 3])) == em1


def test_initial_conditions_4():
    assert np.sum(initial_conditions(300, [1, 2, 3])) == np.sum([300, 300, 300])


def test_initial_conditions_5():
    assert initial_conditions('duck', np.array([1, 2, 3])) == em2
