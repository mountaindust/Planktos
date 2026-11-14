Planktos API
============

Simulations in Planktos are carried out using the Environment class, which 
contains all information about the environment (both spatial and temporal), and 
the Swarm class, which is a collection of agents sharing a similar motion model.

Planktos provides:
    1. An Environment class for agent Swarms to move around in
    2. A Swarm class for agents
    3. A FluidData class holding the fluid velocity field, which is what
       ``Environment.flow`` is
    4. Run archives, which stream agent state to disk as a run proceeds and
       read it back afterwards
    5. Supporting functions to solve eqns of motion, etc.

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    Environment
    Swarm
    FluidData
    RunArchive
    motion