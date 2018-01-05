"""
Abstract class for the AdaGeo Sampler

Gabriele Abbati, Machine Learning Research Group, University of Oxford
January 2018
"""


# Libraries
from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import sqrtm
from adageo.base_classes import AdaGeoAlgorithm, ObservedSpaceSampler


class AdaGeoSampler(AdaGeoAlgorithm, ABC):

    def __init__(self, objective_function,
                 obs_sampler: ObservedSpaceSampler,
                 epsilon: float = 1e-2, rate_decay: float = 0.0,):
        """
        Constructor.
        :param objective_function: function from which we want to sample from;
        :param obs_sampler: sampler that will act on the observed space.
        """
        AdaGeoAlgorithm.__init__(self, objective_function)
        self.obs_sampler = obs_sampler
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.rate_decay = rate_decay
        self.n_it = 0
        return

    def sample_observed_space(self, n_samples: int = 100, n_burn: int = 10000,
                              thin_factor: int = 100) -> None:
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

    def update_learning_rate(self) -> None:
        """
        Updates the learning rate with decay given as argument to the
        constructor.
        """
        self.epsilon = self.initial_epsilon / (1. + self.rate_decay * self.n_it)
        return

    @abstractmethod
    def perform_step(self):
        """
        Performs a single update step of the Markov Chain used for sampling
        """
        pass

    def run_burn_in(self, n_burn: int):
        """
        Performs the necessary burn-in iterations, before the actual sampling.
        :param n_burn: number of burn-in iterations.
        """
        for n in range(n_burn):
            self.perform_step()
        return

    def sample(self, dim_latent, n_samples_latent: int = 100,
               n_burn_latent: int = 10000, thin_factor_latent: int = 100,
               n_samples_observed: int = 100, n_burn_observed: int = 10000,
               thin_factor_observed: int = 100, ard: bool = False) -> np.array:
        """
        Samples using the AdaGeo-method.
        :param dim_latent: dimension of the latent space used in the
        optimization;
        :param n_samples_latent: number of needed samples (latent space);
        :param n_burn_latent: number of burn-in iterations (latent space);
        :param thin_factor_latent: sampling thinning factor (latent space);
        :param n_samples_observed: number of needed samples (observed space);
        :param n_burn_observed: number of burn-in iterations (observed space);
        :param thin_factor_observed: sampling thinning factor (observed space);
        :param ard: indicating whether to use the Automatic Relevance
        Determination (ARD) kernel or not.
        """
        self.n_it = 0
        self.dim_latent = dim_latent
        self.sample_observed_space(n_samples_observed, n_burn_observed,
                                   thin_factor_observed)
        self.build_latent_space(self.dim_latent, ard=ard)
        self.initialize_from_last_theta()
        self.run_burn_in(n_burn_latent)
        samples = np.copy(self.theta)
        for n in range(n_samples_latent-1):
            self.run_burn_in(thin_factor_latent)
            samples = np.concatenate((samples, self.theta), axis=0)
        return samples


class AdaGeoSGLD(AdaGeoSampler):

    def __init__(self, objective_function,
                 obs_sampler: ObservedSpaceSampler, epsilon=1e-2):
        """
        Constructor.
        :param objective_function: function from which we want to sample from;
        :param obs_sampler: sampler that will act on the observed space.
        """
        AdaGeoSampler.__init__(self, objective_function, obs_sampler)
        self.epsilon = epsilon
        return

    def perform_step(self):
        """
        Performs a single step with the AdaGeo stochastic gradient Langevin
        dynamics update rule in the latent space.
        """
        self.n_it = self.n_it + 1
        self.update_learning_rate()
        observed_gradient = self.get_observed_gradient(self.theta)
        latent_gradient = self.compute_latent_gradient(observed_gradient)
        eta = np.random.normal(0.0, np.sqrt(self.epsilon), self.dim_latent)
        self.omega = self.omega + self.epsilon / 2.0 * latent_gradient + eta
        self.theta = self.gplvm_model.predict(self.omega)[0]
        return


class AdaGeoSGRLD(AdaGeoSGLD):

    def perform_step(self):
        """
        Performs a single step with the AdaGeo stochastic gradient Riemannian
        Langevin dynamics update rule in the latent space.
        """
        self.n_it = self.n_it + 1
        self.update_learning_rate()
        observed_gradient = self.get_observed_gradient(self.theta)
        latent_gradient = self.compute_latent_gradient(observed_gradient)
        g_omega = self.gplvm_model.predict_wishart_embedding(self.omega)[0]
        g_inv_omega = np.linalg.inv(g_omega)
        mu = np.dot(g_inv_omega, latent_gradient[0, :])
        epsilon_derivative = 1e-4
        for k in range(self.dim_latent):
            increment = np.copy(self.omega)
            increment[0, k] = increment[0, k] + epsilon_derivative
            g_derivative =\
                (self.gplvm_model.predict_wishart_embedding(increment)[0] -
                 g_omega) / epsilon_derivative
            tmp_mu = np.dot(g_inv_omega, np.dot(g_derivative, g_inv_omega))
            mu = mu - 2.0 * tmp_mu[:, k]
            mu = mu + g_inv_omega[:, k] * np.trace(np.dot(g_inv_omega,
                                                          g_derivative))
        g_inv_sqrt_omega = sqrtm(g_inv_omega)
        eta = np.random.normal(0.0, np.sqrt(self.epsilon), self.dim_latent)
        self.omega = self.omega + self.epsilon / 2.0 * mu +\
            np.dot(g_inv_sqrt_omega, eta)
        self.theta = self.gplvm_model.predict(self.omega)[0]
        return
