'''
Examples of computing FTLE in 2D. 3D is largely untested.
'''

import planktos
from planktos import motion

# The Planktos FTLE solver works by creating a grid of agents in a swarm and 
#   then solving equations of motion over a certain integration time, T. The 
#   idea is that, unlike other FTLE solvers, Planktos can find this FTLE data 
#   with respect to motion rules as arbitrary as the agents' behavior. The 
#   tracer particle results have been compared to results in VisIt.

# First, create an environment and apply the fluid velocity field from our 
#   time-varying IB2d channel flow data. Again, it can be downloaded from
#   https://drive.google.com/drive/folders/104ekG8cEJYuvk6NR8pTGn4wEISILUcuH?usp=sharing.

envir = planktos.environment(char_L=0.1, rho=1, mu=0.001, U=15)

envir.read_IB2d_vtk_data('ib2d_data', 5.0e-5, 1000)
# Another option is to try a static flow field, for example by just using 
#   cycle number 20 (in which vortices have formed). Just uncomment one or the
#   other depending on what you want to calculate over!
# envir.read_IB2d_vtk_data('ib2d_data', 5.0e-5, 1000, d_start=20, d_finish=20)

# Recall that there are vertex data associated the fluid velocity data. We can
#   optionally load these here. Note that things will be a lot slower, and in 
#   our tests of this particular example, it doesn't make much difference whether 
#   the agents know about the mesh structures or not:
# envir.read_IB2d_vertex_data('ib2d_data/channel.vertex')

# If you want a gut-check of what you have at this point, you can always plot it:
# envir.plot_flow()

# Now, we calculate the FTLE. By default, this is done with tracer particles. 
#   The first argument to this method gives the grid dimensions. The second
#   argument gives the integration time, and the third argument defines the 
#   temporal step-size to take. See the documentation of the method for full 
#   details.

# If you have the vertex data loaded, you can omit grid points that would fall 
#   inside closed structures like our cylinder by choosing a test direction for
#   seeing if points are inside a structure. See the documentation of the 
#   calculate_FTLE method for details. If you want to see how this works, 
#   uncomment the following two lines:
# s = envir.add_swarm(900, init='grid', grid_dim=(30,30), testdir='x1')
# s.plot()

# Otherwise, these will calculate the FTLE!
##############     Basic Tracer Particles     ##############
envir.calculate_FTLE((512,128),T=0.1,dt=0.001) # w/o vertex data loaded
# envir.calculate_FTLE((102,25),T=0.1, dt=0.001, testdir='x1') # w/ vertex data loaded
############################################################

# The result is saved in the environment object and can be plotted like this:
envir.plot_2D_FTLE()





################################################################################
# OK, but what about something other than tracer particles?
# The next most complicated thing one can do is to use a different set of ODEs
#   from tracer particles... for example, ODEs for inertial particles. There is
#   a generator for such an ode function in the planktos.motion library, so
#   let's use that.

envir = planktos.environment(char_L=0.1, rho=1, mu=0.001, U=15)
# NOTE: We'll start with cycle 20 in the VTK data here, instead of from the 
# beginning. It produces a more exciting result!
envir.read_IB2d_vtk_data('ib2d_data', 5.0e-5, 1000, d_start=20, d_finish=None)

##############     Use a Passed in ODE Generator    ##############
# Note: intertial particles requires certain parameters to be present in the 
#   swarm object that generates the ODE functions (which in turn define the 
#   parameters in the ODEs). We can specify these using a dictionary and the
#   props parameter of calculate_FTLE.
envir.calculate_FTLE((512,128),T=0.1,dt=0.001,ode_gen=motion.inertial_particles,
                      props={'R':2/3, 'diam':0.01})
envir.plot_2D_FTLE()





################################################################################
# Finally, we can calculate the FTLE using behavior rules pulled from a passed 
#   in swarm object subclass. All this example does is reproduce the tracer
#   particle result using a swarm and Euler steps (we can't solve arbitrary 
#   sub-class motion using anything except Euler steps), but even so, being able 
#   to do so gives us a lot of freedom for analysis! Note that the passed-in 
#   swarm is not used or changed itself in any way - a copy is made, and then 
#   that is operated on.

envir = planktos.environment(char_L=0.1, rho=1, mu=0.001, U=15)
envir.read_IB2d_vtk_data('ib2d_data', 5.0e-5, 1000, d_start=20, d_finish=None)

##############     FTLE with passed in swarm     ##############
class ftle_swrm(planktos.swarm):
    
    def get_positions(self, dt, params=None):
       return self.positions + self.get_fluid_drift()*dt

swrm = ftle_swrm(envir=envir)
envir.calculate_FTLE((512,128),T=0.1,dt=0.001, swrm=swrm)
envir.plot_2D_FTLE()


