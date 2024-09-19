'''
========
Planktos
========

Provides
  1. An environment class for agent swarms to move around in
  2. A swarm class for agents
  3. Supporting functions to handle data I/O, solving eqns of motion, etc.
'''

__author__ = "W. Christopher Strickland"
__email__ = "cstric12@utk.edu"
__copyright__ = "Copyright 2017, Christopher Strickland"
__version__ = '0.7.0'

from .environment import environment
from .swarm import swarm
# from . import motion
# from . import dataio

__all__ = ["environment", "swarm", "motion", "dataio", "geo"]
environment.__module__ = "environment"
swarm.__module__ = "swarm"