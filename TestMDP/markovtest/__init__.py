"""
Markov Property Testing for Sequential Decision Processes

Example:
    >>> from markovtest import test
    >>> from markovtest.environments import MovingTigerKthOrder
    >>> 
    >>> env = MovingTigerKthOrder(k=7)
    >>> data = env.generate(n_trajectories=100, trajectory_length=50)
    >>> p_value = test(data, J=6)
"""

from .testing import test
from . import environments

__version__ = '0.1.0'

__all__ = [
    'test',
    'environments',
]