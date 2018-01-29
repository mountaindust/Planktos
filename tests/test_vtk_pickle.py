#! /usr/bin/env python3

'''Test pickle load of VTK data.
Requires pickle data to already be generated and present in IBAMR_test_data.'''

import numpy as np
import framework

def test_IBAMR_pickle_load():
    '''Test loading pickled IBAMR fluid data into the environment'''
    pathname = 'tests/IBAMR_test_data'
    prefix = 'IBAMR_'
    envir = framework.environment() # default environment, 2D.

    ##### Load the pickled flow #####
    envir.read_pickled_vtk_data(pathname,prefix)
    envir.set_boundary_conditions(('zero','zero'), ('zero','zero'), ('noflux','noflux'))