"""
AdaGeo Optimizers

Gabriele Abbati, Machine Learning Research Group, University of Oxford
January 2018
"""


# Libraries
from abc import ABC, abstractmethod
from adageo.base_classes import AdaGeoAlgorithm, ObservedSpaceOptimizer,\
    Optimizable
import numpy as np


class AdaGeoOptimizer(AdaGeoAlgorithm, ABC):

    def __init__(self, objective_function: Optimizable,
                 obs_optimizer: ObservedSpaceOptimizer,
                 learning_rate: float = 1e-2, rate_decay: float = 1e-3,
                 t_observed: int = 15, t_latent: int = 15):
        """
        Constructor.
        :param objective_function: function from which we want to optimize;
        :param obs_optimizer: optimizer that will act on the observed space;
        :param learning_rate: learning rate in the update rule on the latent
        space;
        :param rate_decay: how fast the learning rate decays (needed for SGD);
        :param t_observed: how many iterations in the observed space;
        :param t_latent: how many iterations in the latent space.
        """
        AdaGeoAlgorithm.__init__(self, objective_function)
        self.obs_optimizer = obs_optimizer
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.rate_decay = rate_decay
        self.t_observed = t_observed
        self.t_latent = t_latent
        self.n_it = 0
        return

    def get_observed_gradient(self, theta: np.array) -> np.array:
        """
        Returns the value of the gradients in the observed space computed at
        theta;
        :param theta: coordinates at which the gradient is computed;
        :return: numpy array containing the gradients at theta.
        """
        return self.objective.get_gradient(theta)

    def update_learning_rate(self) -> None:
        """
        Updates the learning rate with decay given as argument to the
        constructor.
        """
        self.learning_rate = self.initial_learning_rate / (1. + self.rate_decay
                                                           * self.n_it)
        return

    @abstractmethod
    def perform_step(self) -> None:
        """
        Performs a single optimization step in the latent space.
        """
        pass

    def optimize(self, dim_latent: int, n_iterations: int = 10,
                 ard: bool = False) -> None:
        """
        Main optimization function.
        :param dim_latent: dimension of the latent space used in the
        optimization;
        :param n_iterations: number of iterations of the WHOLE method, meaning
        how many times t_observed and t_latent updates are executed;
        :param ard: indicating whether to use the Automatic Relevance
        Determination (ARD) kernel or not.
        """
        self.dim_latent = dim_latent
        for n in range(n_iterations):
            self.observed_samples = self.objective.optimize(self.t_observed)
            self.build_latent_space(self.dim_latent, ard=ard)
            self.initialize_from_last_theta()
            for t in range(self.t_latent):
                self.perform_step()
        return


class AdaGeoGradientDescent(AdaGeoOptimizer):

    def __init__(self, objective_function: Optimizable,
                 obs_optimizer: ObservedSpaceOptimizer,
                 learning_rate: float = 1e-2, rate_decay: float = 1e-3,
                 t_observed: int = 15, t_latent: int = 15):
        """
        Constructor.
        :param objective_function: function from which we want to optimize;
        :param obs_optimizer: optimizer that will act on the observed space;
        :param learning_rate: learning rate in the update rule on the latent
        space;
        :param rate_decay: how fast the learning rate decays (needed for SGD);
        :param t_observed: how many iterations in the observed space;
        :param t_latent: how many iterations in the latent space.
        """
        AdaGeoOptimizer.__init__(self, objective_function, obs_optimizer,
                                 learning_rate, rate_decay, t_observed,
                                 t_latent)
        return

    def perform_step(self) -> None:
        """
        Performs a single optimization step using the vanilla (stochastic)
        gradient descent.
        """
        self.n_it = self.n_it + 1
        self.update_learning_rate()
        observed_gradient = self.get_observed_gradient(self.theta)
        latent_gradient = self.compute_latent_gradient(observed_gradient)
        self.omega = self.omega - self.learning_rate * latent_gradient[0, :]
        self.theta = self.gplvm_model.predict(self.omega)[0]
        return


class AdaGeoMomentumDescent(AdaGeoOptimizer):

    def __init__(self, objective_function: Optimizable,
                 obs_optimizer: ObservedSpaceOptimizer,
                 learning_rate: float = 1e-2, rate_decay: float = 1e-3,
                 momentum_factor: float = .9, t_observed: int = 15,
                 t_latent: int = 15):
        """
        Constructor.
        :param objective_function: function from which we want to optimize;
        :param obs_optimizer: optimizer that will act on the observed space;
        :param learning_rate: learning rate in the update rule on the latent
        space;
        :param rate_decay: how fast the learning rate decays (needed for SGD);
        :param momentum_factor: factor multiplying the Nesterov momentum;
        :param t_observed: how many iterations in the observed space;
        :param t_latent: how many iterations in the latent space.
        """
        AdaGeoOptimizer.__init__(self, objective_function, obs_optimizer,
                                 learning_rate, rate_decay, t_observed,
                                 t_latent)
        self.momentum_factor = momentum_factor
        self.momentum = None
        return

    def perform_step(self) -> None:
        """
        Performs a single optimization step using the Nesterov momentum
        (stochastic) gradient descent.
        """
        self.n_it = self.n_it + 1
        self.update_learning_rate()
        observed_gradient = self.get_observed_gradient(self.theta)
        latent_gradient = self.compute_latent_gradient(observed_gradient)
        self.momentum = self.momentum_factor * self.momentum + \
            self.learning_rate * latent_gradient[0, :]
        self.omega = self.omega - self.momentum
        self.theta = self.gplvm_model.predict(self.omega)[0]
        return

    def optimize(self, dim_latent: int, n_iterations: int = 10,
                 ard: bool = False) -> None:
        """
        Main optimization function.
        :param dim_latent: dimension of the latent space used in the
        optimization;
        :param n_iterations: number of iterations of the WHOLE method, meaning
        how many times t_observed and t_latent updates are executed;
        :param ard: indicating whether to use the Automatic Relevance
        Determination (ARD) kernel or not.
        """
        self.dim_latent = dim_latent
        for n in range(n_iterations):
            self.observed_samples = self.objective.optimize(self.t_observed)
            self.build_latent_space(self.dim_latent, ard=ard)
            self.initialize_from_last_theta()
            self.momentum = np.zeros(self.omega.shape)
            for t in range(self.t_latent):
                self.perform_step()
        return


class AdaGeoAdaGrad(AdaGeoOptimizer):

    def __init__(self, objective_function: Optimizable,
                 obs_optimizer: ObservedSpaceOptimizer,
                 learning_rate: float = 1e-2, smoothing_factor: float = 1e-8,
                 t_observed: int = 15, t_latent: int = 15):
        """
        Constructor.
        :param objective_function: function from which we want to optimize;
        :param obs_optimizer: optimizer that will act on the observed space;
        :param learning_rate: learning rate in the update rule on the latent
        space;
        :param smoothing_factor: smoothing factor in the AdaGrad update rule;
        :param t_observed: how many iterations in the observed space;
        :param t_latent: how many iterations in the latent space.
        """
        AdaGeoOptimizer.__init__(self, objective_function, obs_optimizer,
                                 learning_rate=learning_rate,
                                 t_observed=t_observed, t_latent=t_latent)
        self.smoothing_factor = smoothing_factor
        self.grad_accumulator = None
        return

    def perform_step(self) -> None:
        """
        Performs a single optimization step using the Nesterov momentum
        (stochastic) gradient descent.
        """
        self.n_it = self.n_it + 1
        self.update_learning_rate()
        observed_gradient = self.get_observed_gradient(self.theta)
        latent_gradient = self.compute_latent_gradient(observed_gradient)
        self.grad_accumulator = self.grad_accumulator + \
            latent_gradient[0, :] * latent_gradient[0, :]
        adagrad_rates = self.learning_rate / np.sqrt(self.grad_accumulator +
                                                     self.smoothing_factor)
        self.omega = self.omega - adagrad_rates * latent_gradient[0, :]
        self.theta = self.gplvm_model.predict(self.omega)[0]
        return

    def optimize(self, dim_latent: int, n_iterations: int = 10,
                 ard: bool = False) -> None:
        """
        Main optimization function.
        :param dim_latent: dimension of the latent space used in the
        optimization;
        :param n_iterations: number of iterations of the WHOLE method, meaning
        how many times t_observed and t_latent updates are executed;
        :param ard: indicating whether to use the Automatic Relevance
        Determination (ARD) kernel or not.
        """
        self.dim_latent = dim_latent
        for n in range(n_iterations):
            self.observed_samples = self.objective.optimize(self.t_observed)
            self.build_latent_space(self.dim_latent, ard=ard)
            self.initialize_from_last_theta()
            self.grad_accumulator = np.zeros(self.omega.shape)
            for t in range(self.t_latent):
                self.perform_step()
        return


class AdaGeoNaturalGradient(AdaGeoOptimizer):

    def __init__(self, objective_function: Optimizable,
                 obs_optimizer: ObservedSpaceOptimizer,
                 learning_rate: float = 1e-2, rate_decay: float = 1e-3,
                 t_observed: int = 15, t_latent: int = 15):
        """
        Constructor.
        :param objective_function: function from which we want to optimize;
        :param obs_optimizer: optimizer that will act on the observed space;
        :param learning_rate: learning rate in the update rule on the latent
        space;
        :param rate_decay: how fast the learning rate decays (needed for SGD);
        :param t_observed: how many iterations in the observed space;
        :param t_latent: how many iterations in the latent space.
        """
        AdaGeoOptimizer.__init__(self, objective_function, obs_optimizer,
                                 learning_rate, rate_decay, t_observed,
                                 t_latent)
        return

    def perform_step(self) -> None:
        """
        Performs a single optimization step using the vanilla (stochastic)
        gradient descent.
        """
        self.n_it = self.n_it + 1
        self.update_learning_rate()
        observed_gradient = self.get_observed_gradient(self.theta)
        natural_gradient = self.compute_natural_latent_gradient(
            observed_gradient)
        self.omega = self.omega - self.learning_rate * natural_gradient[0, :]
        self.theta = self.gplvm_model.predict(self.omega)[0]
        return
