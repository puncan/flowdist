from ..heattransfer import initial_conditions
import numpy as np


def test_initial_conditions():
    assert np.allclose(300, np.array([1, 2, 3]))



