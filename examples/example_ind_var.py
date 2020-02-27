#! /usr/bin/env python3
'''
This script is an example of how to specify individual variation for the agents
in a swarm. In this case, the covariance matrix for the brownian motion varies
from agent to agent.
'''

from sys import platform
if platform == 'darwin': # OSX backend does not support blitting
    import matplotlib
    matplotlib.use('Qt5Agg')
import numpy as np
import sys
sys.path.append('..')
import Planktos

envir = Planktos.environment(rho=1000, mu=1000)
# envir.set_brinkman_flow(alpha=66, a=15, res=100, U=range(1,6), 
#                         dpdx=np.ones(5)*0.22306, tspan=[0, 10])
U=0.1*np.array(list(range(0,5))+list(range(5,-5,-1))+list(range(-5,8,3)))
envir.set_brinkman_flow(alpha=66, a=1.5, res=101, U=U, 
                        dpdx=np.ones(20)*0.22306, tspan=[0, 20])

print(len(U))

s = envir.add_swarm()
# for each of the 100 (default) agents, specify an unbiased random walk with
#   variance in each direction between 0 and 0.1, chosen uniformly.
s.add_prop('cov', [np.eye(2)*0.1*np.random.rand() for ii in range(100)])

print('Moving swarm...')
for ii in range(240):
    s.move(0.1)

#s.plot_all('ex_2d.mp4', fps=20)
s.plot_all()