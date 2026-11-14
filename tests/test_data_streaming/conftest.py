'''Local configuration for the data-streaming acceptance tests.

The helpers live in _streaming.py; only what pytest has to discover for itself
is here. Markers and the --runslow option come from the repository-root
conftest.py, which applies to everything under tests/.
'''

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def _close_figures():
    '''No test here is about a figure staying open.'''

    yield
    plt.close('all')
