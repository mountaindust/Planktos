'''
Script for visual-inspection tests of 2D FTLE
'''

from sys import platform
if platform == 'darwin': # OSX backend does not support blitting
    import matplotlib
    matplotlib.use('Qt5Agg')
import numpy as np
import sys
sys.path.append('..')
import Planktos

# TODO: get a less-difficult geometry
envir = Planktos.environment()
envir.read_IB2d_vtk_data('data/channel_cyl', 5.0e-5, 1000, d_start=1)
envir.read_IB2d_vertex_data('data/channel_cyl/channel.vertex')

# Test basic tracer particles on a masked mesh.
envir.calculate_FTLE((30,30), testdir='x1')

envir.plot_envir()