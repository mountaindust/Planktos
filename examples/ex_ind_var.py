#! /usr/bin/env python3
'''
This provides an example of how to specify individual variation for the agents
in a swarm. In this case, the covariance matrix for the brownian motion varies
from agent to agent.
'''

import numpy as np
import planktos

# Let's go ahead and create a default environment with some Brinkman flow in 2D.
envir = planktos.environment(rho=1000, mu=1000)
envir.set_brinkman_flow(alpha=66, h_p=1.5, U=1, dpdx=1, res=101)

# Let's add a very default swarm. Another way of adding a swarm to an existing 
#   environment is by using the envir.add_swarm method. You can pass any 
#   parameters for the swarm to the add_swarm method just like you would the
#   planktos.swarm constructor. You just skip the envir argument!
swrm = envir.add_swarm()

# Now let's specify some individualized behavior. For each of the 100 agents, 
#   let's specify that their variance for the unbiased random walk be a number
#   between 0 and 0.1, chosen via a uniform random distribution.

# Under the hood, this goes into the swrm.prop pandas DataFrame as a column of
#   2x2 covariance matrices, one for each agent. But we can accomplish this 
#   quickly and easily by using the add_prop method of the swarm object, which 
#   recognizes both that our property is varying from agent to agent (because
#   it is a list of matrices and not a single matrix) and that it has the same 
#   name as a shared parameter 'cov', and so the old property should be deleted!
swrm.add_prop('cov', [np.eye(2)*0.1*np.random.rand() for ii in range(100)])

print('Moving swarm...')
for ii in range(180):
    swrm.move(0.1)

# Let's output this to a video. Notice how some agents have more jitter than others!
swrm.plot_all('ex_2d_ind_var.mp4', fps=20)
# swrm.plot_all()