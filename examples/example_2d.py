#! /usr/bin/env python3

from sys import platform
if platform == 'darwin': # OSX backend does not support blitting
    import matplotlib
    matplotlib.use('Qt5Agg')
import numpy as np
import sys
sys.path.append('..')
import Planktos

envir = Planktos.environment(rho=1000, mu=1000)
# envir.set_brinkman_flow(alpha=66, a=15, U=range(1,6), 
#                         dpdx=np.ones(5)*0.22306, res=101, tspan=[0, 10])
U=0.1*np.array(list(range(0,5))+list(range(5,-5,-1))+list(range(-5,8,3)))
envir.set_brinkman_flow(alpha=66, h_p=1.5, U=U, dpdx=np.ones(20)*0.22306,
                        res=101, tspan=[0, 20])

print(len(U))

envir.add_swarm()
s = envir.swarms[0]
s.shared_props['cov'] *= 0.01

print('Moving swarm...')
for ii in range(240):
    s.move(0.1)

#s.plot_all('ex_2d.mp4', fps=20)
s.plot_all()