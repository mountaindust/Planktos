'''File for configuring py.test tests'''

import pytest
from pathlib import Path
from planktos import _dataio

# Exploratory / visual scripts and the perf benchmark live here. They are not
# automated tests (and several need external data or produce plots/videos), so
# keep them out of collection entirely.
collect_ignore = ['tests/manual']
# vtk is a mandatory dependency (see setup.cfg) that _dataio imports
# unconditionally, so it is present whenever planktos imports. Fall back to True
# if _dataio does not expose an explicit VTK availability flag.
VTK = getattr(_dataio, 'VTK', True)

def pytest_addoption(parser):
    '''Adds parser options'''
    parser.addoption('--runslow', action='store_true', default=False,
                     help='run slow tests')
    parser.addoption('--runstreaming', action='store_true', default=False,
                     help='run the data-streaming acceptance tests '
                          '(tests/test_data_streaming/)')

def pytest_collection_modifyitems(config, items):
    '''If test is marked with the pytest.mark.slow decorator, mark it to be
    skipped, unless the --runslow option has been passed.'''
    if not config.getoption("--runslow"):
        # --runslow not given in cli: skip slow tests
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    # The data-streaming acceptance tests are a goal line for work in progress,
    # not a regression suite, and they run whole simulations -- so they are
    # opt-in rather than part of the default run. --runstreaming and --runslow
    # are independent: the slow members of that directory (the example scripts,
    # the cross-version check, the movie renders) need both.
    if not config.getoption("--runstreaming"):
        skip_streaming = pytest.mark.skip(
            reason="need --runstreaming option to run")
        for item in items:
            if "streaming" in item.keywords:
                item.add_marker(skip_streaming)
    # skip vtk tests if unable to import vtk
    if not VTK:
        skip_vtk = pytest.mark.skip(reason="could not load VTK")
        for item in items:
            if "vtk" in item.keywords:
                item.add_marker(skip_vtk)
    # skip comsol tests if unable to find comsol data
    path = Path('tests/data/comsol/')
    if not path.is_dir():
        skip_vtu = pytest.mark.skip(reason="could not load VTU data")
        for item in items:
            if "vtu" in item.keywords:
                item.add_marker(skip_vtu)

# Note: the slow/vtk/vtu markers are registered in pytest.ini (single source of
# truth). Do not re-register them here.