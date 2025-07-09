'''
========
Planktos
========

Provides
  1. An Environment class for agent swarms to move around in
  2. A Swarm class for agents
  3. Supporting functions to handle data I/O, solving eqns of motion, etc.
'''

__author__ = "W. Christopher Strickland"
__email__ = "cstric12@utk.edu"
__copyright__ = "Copyright 2025, Christopher Strickland"
__version__ = '1.0.0'

from .environment import Environment
from .swarm import Swarm
# from . import motion
# from . import dataio

__all__ = ["Environment", "Swarm", "motion", "dataio", "geom"]
Environment.__module__ = "environment"
Swarm.__module__ = "swarm"