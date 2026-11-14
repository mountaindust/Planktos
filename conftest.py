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
    parser.addoption('--runexamples', action='store_true', default=False,
                     help='run the example scripts as written')

def pytest_collection_modifyitems(config, items):
    '''Skip tests marked slow unless --runslow has been passed, and tests marked
    examples unless --runexamples has been passed.'''
    if not config.getoption("--runslow"):
        # --runslow not given in cli: skip slow tests
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    # The example scripts run whole simulations and take minutes between them,
    # so they have their own flag, independent of --runslow.
    if not config.getoption("--runexamples"):
        skip_examples = pytest.mark.skip(
            reason="need --runexamples option to run")
        for item in items:
            if "examples" in item.keywords:
                item.add_marker(skip_examples)
    # skip vtk tests if unable to import vtk
    if not VTK:
        skip_vtk = pytest.mark.skip(reason="could not load VTK")
        for item in items:
            if "vtk" in item.keywords:
                item.add_marker(skip_vtk)
    # skip comsol tests if unable to find comsol data
    path = Path('tests/data/comsol_vtu/rect_channelflow_uvw.vtu')
    if not path.is_file():
        skip_vtu = pytest.mark.skip(reason="could not load VTU data")
        for item in items:
            if "vtu" in item.keywords:
                item.add_marker(skip_vtu)

# Note: the slow/vtk/vtu markers are registered in pytest.ini (single source of
# truth). Do not re-register them here.