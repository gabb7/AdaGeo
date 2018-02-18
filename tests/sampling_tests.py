"""
Some densities that implement the Samplable abstract class in base_classes.py.
These are used to show how the AdaGeo sampling framework can be used.

Gabriele Abbati, Machine Learning Research Group, University of Oxford
January 2018
"""


# Libraries
import autograd.numpy as np
from autograd import grad
from adageo.base_classes import Samplable


# noinspection PyUnresolvedReferences
class BananaDensity(Samplable):
    """
    Banana-shaped density:
        p(theta) \propto \exp ( - theta_0^2 / 200.0 - (theta_1 - b * theta_0^2 +
        100 * b)^2 / 2.0 - sum_{j=2}^{D-1} theta_j^2)
    """

    def __init__(self, dim_observed: int):
        """
        Constructor.
        :param dim_observed: dimensionality of the density.
        """
        self.dim_observed = dim_observed
        self.b = 0.1
        self.density_gradient = grad(self.p)
        return

    def p(self, x: np.array) -> float:
        """
        Returns the value of the unnormalized probability distribution computed
        at x.
        :param x: numpy array containing the desired coordinates.
        :return: probability at x.
        """
        p = - x[0]**2 / 200.0 - 0.5 * (x[1] - self.b * x[0]**2 +
                                       100.0 * self.b)**2
        p_sum = - 0.5 * np.sum(x[2:-1]**2)
        p = p + p_sum
        return np.exp(p)

    def get_gradient(self, x: np.array) -> np.array:
        """
        Computes the gradient of the probability distribution with respect to x,
        which in this case is the parameter vector.
        :param x: numpy array containing the desired coordinates.
        :return: numpy.array containing the gradient of the probability
        distribution.
        """
        return self.density_gradient(x)


# noinspection PyUnresolvedReferences
class LogBananaDensity(BananaDensity):
    """
    Banana-shaped log-density:
        p(theta) \propto ( - theta_0^2 / 200.0 - (theta_1 - b * theta_0^2 +
        100 * b)^2 / 2.0 - sum_{j=2}^{D-1} theta_j^2)
    """

    def p(self, x: np.array) -> float:
        """
        Returns the value of the unnormalized probability log-distribution
        computed at x.
        :param x: numpy array containing the desired coordinates.
        :return: log-probability at x.
        """
        p = - x[0]**2 / 200.0 - 0.5 * (x[1] - self.b * x[0]**2 +
                                       100.0 * self.b)**2
        p_sum = - 0.5 * np.sum(x[2:-1]**2)
        p = p + p_sum
        return p
