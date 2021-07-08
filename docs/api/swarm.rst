swarm class
===========

The swarm class acts as a collection of agents that share a similar motion model. 
All agents must be grouped into a swarm, and every swarm must also have an 
associated environment object. Internally, information about agent locations, 
velocities, and accelerations within the swarm are stored as masked NumPy arrays 
where each row is an agent and each column is a spatial dimension. The mask 
refers to whether or not the agent has left the spatial domain. Any swarm 
properties that do not vary from agent to agent are stored within the 
shared_props attribute, implemented as a Python dictionary. Properties that 
vary between agents are stored within the props attribute as a pandas DataFrame.

Created on Tues Jan 24 2017

Author: Christopher Strickland

Email: cstric12@utk.edu

.. autoclass:: planktos.swarm
    :members:
