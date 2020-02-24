#! /usr/bin/env python3

import sys
sys.path.append('..')
from sys import platform
if platform == 'darwin': # OSX backend does not support blitting
    import matplotlib
    matplotlib.use('TkAgg')
import numpy as np
import Planktos, data_IO

# High-viscous enviornment (Re=10)
envir = Planktos.environment(Lx=.5,Ly=.5,Lz=1, rho=1000, mu=10)
# Specify static velocity along the top of the domain
U = 0.09818
# Specify pressure gradient
dpdx = 0.0234759411877

envir.set_brinkman_flow(alpha=37.24, a=0.15, res=101, U=U, dpdx=dpdx)
envir.add_swarm()
s = envir.swarms[0]

# Specify amount of jitter (mean, covariance)
# Set std as 1 cm = 0.01 m
s.shared_props['cov'] *= 0.01**2

print('Moving swarm...')
for ii in range(240):
    s.move(0.1)


##########              Plot!               ###########
s.plot_all('brine_shrimp_Brinkman.mp4', fps=20)
#s.plot_all()