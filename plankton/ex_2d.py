#! /usr/bin/env python3

import numpy as np
from plankton import plankton

p = plankton(swarm_size=100, init='random')
U=list(range(0,5))+list(range(5,-5,-1))+list(range(-3,6,2))
p.envir.set_brinkman_flow(alpha=66, a=15, res=100, U=U, 
                          dpdx=np.ones(20)*0.22306, tspan=[0, 20])

print('Moving plankton...')
for ii in range(50):
    p.move(0.5)

p.plot_all()