#! /usr/bin/env python3
'''
This example/tutorial builds on the ex_ib2d_sticky. Its
primary purpose is to show off the moving boundaries functionality of Planktos. 
This example/tutorial also implements the sticky boundary
condition option. 

NOTE: IN ORDER TO RUN THIS EXAMPLE, YOU MUST HAVE THE REQUIRED DATA!
It can be downloaded from: 
https://drive.google.com/drive/folders/1LGfOdLX0WJrC-v9khSpivQLhlZ_RsJIN?usp=sharing

Put it into the ib2d_jellyfish_data folder in this example directory, and you should be 
good to go!

This example loads data generated from an IB2d jellyfish example for the generated 
fluid velocity field AND loads the lag points data, creating an ibmesh out of it by
attaching nearby points. 

In the ex_ib2d_ibmesh, agents cannot move through the cylinder or the channel boundaries.
In this example, the agents can move through the moving jellyfish mesh.
'''

import numpy as np
import planktos

#import numpy.ma which is used for masked arrays
#   in this example, we will use masked arrays to "remove" particles that have stuck to the jellyfish
import numpy.ma as ma

# Let's begin by loading the same fluid and mesh as used in ex_ib2d_ibmesh.py
envir = planktos.Environment()
envir.read_IB2d_fluid_data('examples/ib2d_jellyfish_data', 1.25e-5  , 1600)

#In the ex_ib2d_ibmesh we read in the vertex data to get an immersed mesh. 
#   This approach does not work for moving boundaries, 
#   so we will read in the lag points data instead.
#   We also need to manually remove any indexs corresponding to lag points which are connected to wrong points. 
#   See the documention for read_IB2d_mesh_data in _swarm.py for more detail about brk_idx_list and other option. 
envir.read_IB2d_mesh_data('examples/ib2d_jellyfish_data', dt=1.25e-5  , print_dump=1600, brk_idx_list=[241])

# In this example, we are going to create agents that stick to immersed
#   boundaries (jellyfish) whenever they come in contact, and then remove those agents!

# To do this, we need to define some new agent behavior. It will rely on a 
#   user-defined agent property, which we will call 'stick'.
class permstick(planktos.Swarm):
    def apply_agent_model(self, dt):
        # first, we will get the value of the 'stick' property. It is expected
        #   to be different across agents, and to have boolean value. If it is
        #   False, we aren't sticking. If it is True, we will stay put!
        stick = self.get_prop('stick')

        # The most computationally efficient way to do this would be to only
        #   get movement vectors for the agents that are not stuck. But for
        #   brevity, we will get the movement for all of them and then adjust.
        
        all_move = planktos.motion.Euler_brownian_motion(self, dt)

        # Multiplying by a boolean array is like multiplying by 1s and 0s. So
        #   we just need to expand the dimension of the stick array (it's shape
        #   (N,) while all_move is shape (N,2)... this causes a broadcasting 
        #   error unless we expand stick to shape (N,1)), multiply and add!
        #   Putting a ~ in front of stick will negate the boolean values in
        #   the array.
        return np.expand_dims(~stick,1)*all_move +\
               np.expand_dims(stick,1)*self.positions
    
    # After an agent runs into an immersed structure, we want it to be removed from the plot
        #for all future times. There is an attribute of the Swarm object called 
    #   ib_collision which is an array of bool, one for each agent. If the agent 
    #   collided with an immersed structure in the most recent move, it is set 
    #   to True for that agent. Otherwise, it is False. We'll use that to 
    #   dynamically update our 'stick' property after the move is over.
    #   We also will use that to mask the positions of the particles that have 
    #   stuck to the jellyfish. This will remove the stuck particles from the plot.
    
    # To do this, we will override after_move, a method that gets called 
    #   after all the agents have moved.
    def after_move(self, dt):
        swrm.props.loc[swrm.ib_collision, 'stick'] = True
        self.positions[self.ib_collision] = ma.masked

# Now we create the Swarm similar to ex_ib2d_sticky.py.
# We will set store_prop_history=True because we want to keep track of agent 
#   property changes through time. We will also set the default ib condition to 
#   'sticky'. We could alternatively pass it in each time to the move method 
#   below.
#We will also set the initial conditions to be random positions near the jellyfish 
#   in order to the moving boundary functionality.
        
N=50 #number of particles
#create Nx2 array of random initial positions near the jellyfish
ICs= np.zeros((N,2))
ICs[:,0]= np.random.uniform(1.1,1.3,N) #x-positions of the particles
ICs[:,1]= np.random.uniform(0.01,1.3,N) #y-positions of the particles

swrm = permstick(swarm_size=N, envir=envir, 
                 init=ICs, store_prop_history=True, 
                 ib_condition='sticky')

swrm.shared_props['cov'] *= 0.001
swrm.shared_props['mu'] = [0.2,0.1]
# We also need to initialize our new agent properties. We'll set stick to False 
#   starting out, so that none of them will be stuck at the beginning
swrm.props['stick'] = np.full(N, False) # creates a length 100 array of False
# An equivalent way to do this: swrm.add_prop('stick', np.full(100, False), shared=False)


# Now we move the swarm. We'll use the 'sticky' option for immersed boundary
#   collisions instead of the default sliding option. This means that 
#   whenever an agent runs into an immersed structure, it will stop its movement 
#   for that time step at the point of intersection. It would be free to move in 
#   the next time step however, which is why our after_move updates a property
#   for us that is then used in apply_agent_model.

for ii in range(42):
    swrm.move(0.025, ib_collisions='sticky')
    # if np.any(swrm.ib_collision): # uncomment to display whenever something is getting stuck!
    #     swrm.plot()
    
swrm.plot_all(movie_filename='mvbnd_sticky.mp4', fps=6, fluid='vort',
              plot_heading=False)


