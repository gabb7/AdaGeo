"""
Abstract class for the AdaGeo Optimizer

Gabriele Abbati, Machine Learning Research Group, University of Oxford
January 2018
"""


# Libraries
from abc import ABC, abstractmethod
from adageo.base_classes import AdaGeoAlgorithm, ObservedSpaceOptimizer


class AdaGeoOptimizer(AdaGeoAlgorithm, ABC):

    def __init__(self, objective_function,
                 obs_optimizer: ObservedSpaceOptimizer):
        """
        Constructor.
        :param objective_function: function from which we want to sample from;
        :param obs_optimizer: sampler that will act on the observed space.
        """
        AdaGeoAlgorithm.__init__(self, objective_function)
        self.obs_optimizer = obs_optimizer
        return

    @abstractmethod
    def perform_step(self):
        """
        Performs a single optimization step
        """
        pass

    @abstractmethod
    def optimize(self):
        """
        Main optimization function
        """
        pass
