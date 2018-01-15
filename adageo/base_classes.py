"""
Some base classes used in this AdaGeo implementation.

Gabriele Abbati, Machine Learning Research Group, University of Oxford
January 2018
"""


# Libraries
from abc import ABC, abstractmethod
import numpy as np
import GPy


class Samplable(ABC):

    @abstractmethod
    def p(self, x: np.array) -> float:
        """
        Returns the value of the probability distribution we want to sample
        from computed at x (can be unnormalized)
        :param x: numpy array containing the desired coordinates
        :return: probability at x
        """
        pass

    def get_gradient(self, x: np.array) -> np.array:
        """
        Returns the gradients of the objective function we want to sample from
        computed at x (if implementable)
        :param x: numpy array containing the desired coordinates
        :return: function gradients at x
        """
        pass


class Optimizable(ABC):

    @abstractmethod
    def f(self, x: np.array) -> float:
        """
        Returns the value of the objective function we want to optimize
        computed at x
        :param x: numpy array containing the desired coordinates
        :return: value of the objective function at x
        """
        pass

    @abstractmethod
    def get_gradient(self, x: np.array) -> np.array:
        """
        Returns the gradients of the objective function we want to optimize
        computed at x
        :param x: numpy array containing the desired coordinates
        :return: function gradients at x
        """
        pass


class ObservedSpaceSampler(ABC):

    def __init__(self, objective_function: Samplable, dim_observed: int):
        """
        Constructor.
        :param objective_function: function which we want to sample from.
        :param dim_observed: size of the sampled parameter vector.
        """
        self.objective = objective_function
        self.dim_observed = dim_observed
        self.theta = None
        self.initialize_theta()
        self.n_it = 0
        return

    def initialize_theta(self):
        """
        Initializes the parameter vector from a N(0,1) distribution
        """
        self.theta = np.random.multivariate_normal(np.zeros(self.dim_observed),
                                                   np.eye(self.dim_observed))
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
        samples = []
        for n in range(n_samples):
            self.n_it = self.n_it + 1
            self.run_burn_in(thin_factor)
            samples.append(self.theta)
        return np.asarray(samples)


class ObservedSpaceOptimizer(ABC):

    def __init__(self, objective_function: Optimizable):
        """
        Constructor.
        :param objective_function: function which we want to optimize.
        """
        self.objective = objective_function
        return

    @abstractmethod
    def optimize(self, n_iterations) -> np.array:
        """
        Abstract method that allows you to optimize the objective function.
        It needs to be implemented. Returns a matrix with dimensions
        [n_iterations, dim_observed] in which every row shows the value of
        theta for each optimization step.
        :param n_iterations: number of optimization iterations
        """
        pass


class AdaGeoAlgorithm(object):

    def __init__(self, objective_function: Optimizable):
        """
        Constructor.
        :param objective_function: function which we want to sample from or
        optimize.
        """
        self.objective = objective_function
        self.observed_samples = None
        self.dim_observed = None
        self.dim_latent = None
        self.theta = None
        self.omega = None
        self.gplvm_model = None
        self.jacobian = None
        self.metric_tensor = None
        return

    def load_observed_samples(self, filename: str) -> None:
        """
        Load observed space samples, previously produced with a generic sampler
        acting on the observed space.
        :param filename: string containing the location of the samples.
        """
        self.observed_samples = np.load(filename)
        self.dim_observed = self.observed_samples.shape[1]
        self.theta = np.copy(self.observed_samples[-1, :])
        return

    def build_latent_space(self, dim_latent: int, ard: bool = False,
                           likelihood_variance: float = 0.1) -> None:
        """
        Builds the latent representation for the parameters theta acquired so
        far, saved in self.observed_samples.
        :param dim_latent: dimension of the latent space;
        :param ard: boolean indicating the activation of the ARD kernel;
        :param likelihood_variance: variance associated with the likelihood in
        the GP model.
        """
        self.dim_latent = dim_latent
        kernel = GPy.kern.RBF(input_dim=self.dim_latent, ARD=ard) + GPy.kern.\
            Bias(input_dim=self.dim_latent)
        self.gplvm_model = GPy.models.GPLVM(Y=self.observed_samples,
                                            input_dim=self.dim_latent,
                                            kernel=kernel)
        self.gplvm_model.likelihood.variance = likelihood_variance
        self.gplvm_model.optimize()
        return

    def draw_latent_omega_prior(self) -> None:
        """
        Draw omega from a Gaussian N(0,1) prior on the latent space, as a
        way to set the starting point for the sampling / optimization.
        """
        self.omega = np.reshape(np.random.multivariate_normal(
                     np.zeros(self.dim_latent), np.eye(self.dim_latent)),
                     [1, self.dim_latent])
        self.theta = self.gplvm_model.predict(self.omega)[0]
        self.dim_observed = self.theta.shape[1]
        return

    def initialize_from_last_theta(self) -> None:
        """
        Initialize omega as the point in the latent space that corresponds to
        the last observed sample.
        """
        xx = GPy.plotting.gpy_plot.plot_util.get_x_y_var(self.gplvm_model)[0]
        self.omega = np.reshape(xx[-1, :], [1, self.dim_latent])
        self.theta = self.gplvm_model.predict(self.omega)[0]
        return

    def get_observed_gradient(self, theta: np.array) -> np.array:
        """
        Returns the value of the gradients in the observed space computed at
        theta;
        :param theta: coordinates at which the gradient is computed;
        :return: numpy array containing the gradients at theta.
        """
        return self.objective.get_gradient(theta)

    def compute_jacobian(self) -> None:
        """
        Computes the Jacobian matrix of the GP-LVM mapping. In this case we
        limit to a point estimation using the mean of the matrix. Remember the
        GP-LVM yields the full distribution over the Jacobian of the
        transformation.
        """
        self.jacobian = self.gplvm_model.predict_jacobian(
            self.omega)[0][0, :, :]
        return

    def compute_metric_tensor(self) -> None:
        """
        Computes the metric tensor of the latent space identified by the GP-LVM.
        In this case we limit to a point estimation using the mean of the
        tensor. Remember the GP-LVM yields the full distribution over the
        metric tensor.
        """
        self.metric_tensor = self.gplvm_model.predict_wishart_embedding(
            self.omega)[0]
        return

    def compute_latent_gradient(self, observed_gradient: np.array) -> np.array:
        """
        Computes the gradient in the latent space by multiplying the one in the
        observed space with the Jacobian matrix. Once again so far only a point
        estimate using the mean is employed.
        :param observed_gradient: gradient computed in the observed space;
        :return: the corresponding gradient in the latent space.
        """
        self.compute_jacobian()
        latent_gradient = np.dot(self.jacobian, observed_gradient[0, :])
        return np.reshape(latent_gradient, [1, self.dim_latent])

    def compute_natural_latent_gradient(
            self, observed_gradient: np.array) -> np.array:
        """
        Compute the natural gradient in the latent space using the metric
        tensor yielded by the GP-LVM mapping:
        natural_gradient = inv(metric_tensor) * gradient
        :param observed_gradient: gradient computed in the observed space;
        :return: the corresponding natural gradient in the latent space.
        """
        self.compute_metric_tensor()
        latent_gradient = self.compute_latent_gradient(observed_gradient)
        natural_gradient = np.dot(np.linalg.inv(self.metric_tensor),
                                  latent_gradient)
        return natural_gradient
