"""
Abstract class for the AdaGeo Sampler

Gabriele Abbati, Machine Learning Research Group, University of Oxford
January 2018
"""


# Libraries
from abc import ABC
import numpy as np
from adageo.base_classes import ObservedSpaceSampler


class AdaGeoSampler(ABC):

    def __init__(self, objective_function,
                 obs_sampler: ObservedSpaceSampler):
        """
        Constructor.
        :param objective_function: function from which we want to sample from;
        :param obs_sampler: sampler that will act on the observed space.
        """
        self.objective = objective_function
        self.obs_sampler = obs_sampler
        self.observed_samples = None
        self.dim_observed = None
        self.theta = None
        self.omega = None
        return

    def load_observed_samples(self, filename: str):
        """
        Load observed space samples, previously produced with a generic sampler
        acting on the observed space.
        :param filename: string containing the location of the samples.
        """
        self.observed_samples = np.load(filename)
        self.dim_observed = self.observed_samples.shape[1]
        self.theta = np.copy(self.observed_samples[-1, :])
        return

    def sample_observed_space(self, n_samples: int = 100, n_burn: int = 10000,
                              thin_factor: int = 100):
        """
        Uses the sampler in self.obs_sampler to sample directly from the
        observed space, collecting in this way the samples the latent space is
        built on.
        :param n_samples: number of needed observed samples;
        :param n_burn: number of burn-in iterations;
        :param thin_factor: sampling thinning factor.
        """
        self.observed_samples =\
            self.obs_sampler.sample(n_samples=n_samples,
                                    n_burn=n_burn,
                                    thin_factor=thin_factor)
        self.observed_samples = np.reshape(
            self.observed_samples, [n_samples, self.obs_sampler.dim_observed])
        self.dim_observed = self.obs_sampler.dim_observed
        return
