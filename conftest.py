'''File for configuring py.test tests'''

import pytest
from pathlib import Path
from planktos import _dataio
VTK = _dataio.VTK

def pytest_addoption(parser):
    '''Adds parser options'''
    parser.addoption('--runslow', action='store_true', default=False, 
                     help='run slow tests')
    
def pytest_collection_modifyitems(config, items):
    '''If test is marked with the pytest.mark.slow decorator, mark it to be
    skipped, unless the --runslow option has been passed.'''
    if not config.getoption("--runslow"):
        # --runslow not given in cli: skip slow tests
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
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
    
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "vtk: mark test as requiring vtk data"
    )
    config.addinivalue_line(
        "markers",
        "vtu: mark test as requiring vtu data"
    )