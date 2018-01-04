"""
Some base classes used in this AdaGeo implementation.

Gabriele Abbati, Machine Learning Research Group, University of Oxford
January 2018
"""


# Libraries
from abc import ABC, abstractmethod
import numpy as np
import GPy


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


class AdaGeoAlgorithm(object):

    def __init__(self, objective_function):
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

    def build_latent_space(self, dim_latent, ard=False,
                           likelihood_variance=0.1) -> None:
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

    def initialize_from_last_omega(self) -> None:
        """
        Initialize omega as the point in the latent space that corresponds to
        the last observed sample.
        """
        xx = GPy.plotting.gpy_plot.plot_util.get_x_y_var(self.gplvm_model)[0]
        self.omega = np.reshape(xx[-1, :], [1, self.dim_latent])
        self.theta = self.gplvm_model.predict(self.omega)[0]
        return

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
        gradient_latent = np.dot(self.jacobian, observed_gradient[0, :])
        return np.reshape(gradient_latent, [1, self.dim_latent])
