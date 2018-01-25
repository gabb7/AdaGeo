"""
Python script that test the AdaGeo optimizer while training a neural network
with one layer implementing logistic regression on the MNIST dataset.

Gabriele Abbati, Machine Learning Research Group, University of Oxford
January 2018
"""

# Libraries
from tests.optimization_tests import LogRegressionMNIST
from adageo.optimizers import AdaGeoGradientDescent, AdaGeoMomentumDescent,\
    AdaGeoAdaGrad, AdaGeoNaturalGradientDescent
from adageo.optimizers import GradientDescent, MomentumDescent, AdaGradDescent


test_network = LogRegressionMNIST(n_batch=750)
test_network.build_model()

observed_optimizer = GradientDescent(test_network, test_network.dim_observed,
                                     learning_rate=1e-1, rate_decay=0.001)

adageo_optimizer = AdaGeoMomentumDescent(test_network, observed_optimizer,
                                         learning_rate=5e-1, rate_decay=0.01,
                                         momentum_factor=0.6,
                                         t_observed=20, t_latent=20)

adageo_optimizer.optimize(dim_latent=9, n_iterations=2, ard=True)
