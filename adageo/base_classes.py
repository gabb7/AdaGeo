"""
Some base classes used in this AdaGeo implementation.

Gabriele Abbati, Machine Learning Research Group, University of Oxford
January 2018
"""


# Libraries
from abc import ABC, abstractmethod


class ObservedSpaceSampler(ABC):

    def __init__(self, objective_function):
        self.objective = objective_function
        self.dim_observed = None

    @abstractmethod
    def sample(self, n_samples: int = 100, n_burn: int = 10000,
               thin_factor: int = 100):
        """
        Abstract method that allows you to sample from the objective function.
        It needs to be implemented
        :param n_samples: number of needed observed samples;
        :param n_burn: number of burn-in iterations;
        :param thin_factor: sampling thinning factor.
        """
        pass
