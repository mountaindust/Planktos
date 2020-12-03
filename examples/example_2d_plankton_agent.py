#! /usr/bin/env python3
'''
This example runs a model with plankton agents (defined in plankton_agent.py)
as an example of non-trivial agent behavior specification and implementation.
'''

import numpy as np
from plankton_agent import plankton

p = plankton(swarm_size=100, init='random')
U=0.1*np.array(list(range(0,5))+list(range(5,-5,-1))+list(range(-5,8,3)))
p.envir.set_brinkman_flow(alpha=66, h_p=1.5, U=U, dpdx=np.ones(20)*0.22306,
                          res=100, tspan=[0, 20])

print('Moving plankton...')
for ii in range(50):
    p.move(0.5)

p.plot_all()