from ..fluidmechanics import initial_conditions
import numpy as np

em1 = 'Initial velocity should be positive'
em2 = 'Velocity must be scalar or function of x and y coordinates and grid must be numpy array'


def test_initial_conditions_1():

    def negvelfunc(grid):
        return - 0.5 * grid

    assert initial_conditions(negvelfunc, np.array([1, 2, 3])) == em1


def test_initial_conditions_2():

    def posvelfunc(grid):
        return 5 + 0.5 * grid

    assert np.sum(initial_conditions(posvelfunc, np.array([1, 2, 3]))) == np.sum(np.array([5.5, 6, 6.5]))


def test_initial_conditions_3():
    assert initial_conditions(-1, np.array([1, 2, 3])) == em1


def test_initial_conditions_4():
    assert np.sum(initial_conditions(10, [1, 2, 3])) == np.sum([10, 10, 10])


def test_initial_conditions_5():
    assert initial_conditions('duck', np.array([1, 2, 3])) == em2
