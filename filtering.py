import numpy as np
from numpy.linalg import inv

from . import transformation as tf, probability_representations as pr

class KalmanFilterStep:
    def __init__(dynamics_model, measurement_model):
        self.dynamics_model = dynamics_model
        self.measurement_model = measurement_model

    def propagate_prior(self, prior):
        return x.shove_through_transform(self.dynamics)

    @property
    def prior_measurement_mapping(self):
        return self.dynamics_model.compose_with(self.measurement_model)

    @property
    def updator(self):
        mapping = self.prior_measurement_mapping
        product = GaussianProduct.lift_noisy_transform(mapping)
        return product.update
        

class UnscentedKalmanFilter:
    def __init__():
        pass

class ParticleFilter:
    def __init__():
        pass
