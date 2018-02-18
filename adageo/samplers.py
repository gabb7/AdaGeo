"""
Abstract class for the AdaGeo Sampler

Gabriele Abbati, Machine Learning Research Group, University of Oxford
January 2018
"""


# Libraries
from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import sqrtm
from adageo.base_classes import AdaGeoAlgorithm, ObservedSpaceSampler, Samplable


class MetropolisHastings(ObservedSpaceSampler):
    """
    Implementation of the Metropolis - Hasting sampler.
    """

    def __init__(self, objective_function: Samplable, dim_observed: int,
                 radius: float):
        """
        Constructor.
        :param objective_function: function which we want to sample from;
        :param dim_observed: size of the sampled parameter vector;
        :param radius: radius of the uniform distribution which the proposals
        are sampled from.
        """
        ObservedSpaceSampler.__init__(self, objective_function, dim_observed)
        self.radius = radius
        return

    def propose_sample(self) -> np.array:
        """
        Returns a proposed update, drawn from a uniform distribution centered on
        self.theta and with radius self.radius.
        :return: the proposed update, as a numpy.array.
        """
        norm = np.random.normal
        normal_deviates = norm(size=self.dim_observed)
        r = np.sqrt((normal_deviates**2).sum(axis=0))
        update = normal_deviates / r * self.radius
        return self.theta + update

    def accept_sample(self, proposal: np.array) -> bool:
        """
        Decides whether to accept the proposal returned by the previous
        function, according to the Metropolis - Hastings algorithm.
        :param proposal: possible new sample;
        :return: boolean indicating whether to accept or not.
        """
        ratio = self.objective.p(proposal) / self.objective.p(self.theta)
        if np.random.uniform() < ratio:
            return True
        return False

    def perform_step(self) -> None:
        """
        Performs a single update step of the Markov Chain used for sampling
        """
        proposal = self.propose_sample()
        n = 1.0
        while not self.accept_sample(proposal):
            proposal = self.propose_sample()
            n += 1.0
        self.theta = np.copy(proposal)
        return n

    def sample(self, n_samples: int = 100, n_burn: int = 10000,
               thin_factor: int = 100) -> np.array:
        """
        Sample from the objective function.
        :param n_samples: number of needed observed samples;
        :param n_burn: number of burn-in iterations;
        :param thin_factor: sampling thinning factor.
        """
        self.n_it = 0
        self.run_burn_in(n_burn)
        samples = ObservedSpaceSampler.sample(self, n_samples=n_samples,
                                              n_burn=0, thin_factor=thin_factor)
        return samples


class SGLD(ObservedSpaceSampler):
    """
    Implementation of the Stochastic Gradient Langevin Dynamics.
    """

    def __init__(self, objective_function: Samplable, dim_observed: int,
                 epsilon: float = 1e-2, rate_decay: float = 0.0):
        """
        Constructor.
        :param objective_function: function which we want to sample from;
        :param dim_observed: size of the sampled parameter vector;
        :param epsilon: step size of the sampler;
        :param rate_decay: regulates how the step decreases in time. 0.0 if no
        decrease is needed.
        """
        ObservedSpaceSampler.__init__(self, objective_function, dim_observed)
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.rate_decay = rate_decay
        return

    def update_learning_rate(self) -> None:
        """
        Updates the learning rate with decay given as argument to the
        constructor.
        """
        self.epsilon = self.initial_epsilon / (1. + self.rate_decay * self.n_it)
        return

    def perform_step(self) -> None:
        """
        Performs a single update step of the Markov Chain used for sampling
        """
        self.update_learning_rate()
        observed_gradient = self.objective.get_gradient(self.theta)
        eta = np.random.normal(0.0, np.sqrt(self.epsilon), self.dim_observed)
        self.theta = self.theta + self.epsilon / 2.0 * observed_gradient + eta
        return


class AdaGeoSampler(AdaGeoAlgorithm, ABC):
    """
    Abstract base class for a AdaGeo sampler.
    """

    def __init__(self, objective_function,
                 obs_sampler: ObservedSpaceSampler,
                 epsilon: float = 1e-2, rate_decay: float = 0.0):
        """
        Constructor.
        :param objective_function: function from which we want to sample from;
        :param obs_sampler: sampler that will act on the observed space;
        :param epsilon: step size of the sampler;
        :param rate_decay: regulates how the step decreases in time. 0.0 if no
        decrease is needed.
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
    def perform_step(self) -> None:
        """
        Performs a single update step of the Markov Chain used for sampling
        """
        pass

    def run_burn_in(self, n_burn: int) -> None:
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
        :return: a matrix with dimensions [n_samples, dim_observed] containing
        the samples gathered from the observed space.
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
    """
    Adageo - Stochastic gradient Langevin dynamics
    """

    def perform_step(self) -> None:
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
    """
    Adageo - Stochastic gradient Riemannian Langevin dynamics
    """

    def perform_step(self) -> None:
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
