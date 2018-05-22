#! /usr/bin/env python3

import numpy as np
import sys
sys.path.append('..')
import framework

envir = framework.environment(Lz=10, rho=1000, mu=1000)
U=0.1*np.array(list(range(0,5))+list(range(5,-5,-1))+list(range(-5,8,3)))

envir.set_brinkman_flow(alpha=66, a=1.5, res=101, U=U, 
                        dpdx=np.ones(20)*0.22306, tspan=[0, 20])
envir.add_swarm()
s = envir.swarms[0]

print('Moving swarm...')
for ii in range(50):
    s.move(0.5)

s.plot_all()