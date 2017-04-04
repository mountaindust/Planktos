'''File for configuring py.test tests'''

import pytest

def pytest_addoption(parser):
    parser.addoption('--runslow', action='store_true', help='run slow tests')