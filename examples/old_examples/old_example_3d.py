#! /usr/bin/env python3

import numpy as np
import sys
sys.path.append('..')
import planktos

envir = planktos.environment(Lz=10, rho=1000, mu=1000)
U=0.1*np.array(list(range(0,5))+list(range(5,-5,-1))+list(range(-5,8,3)))

envir.set_brinkman_flow(alpha=66, h_p=1.5, U=U, dpdx=np.ones(20)*0.22306, 
                        res=101, tspan=[0, 20])
envir.add_swarm()
s = envir.swarms[0]
s.shared_props['cov'] *= 0.01

# s.plot()

print('Moving swarm...')
for ii in range(240):
    s.move(0.1)

#s.plot_all('ex_3d.mp4', fps=20)
s.plot_all()