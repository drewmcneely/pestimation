import numpy as np
from numpy.linalg import inv

from . import transformation as tf, probability_representations as pr

class KalmanFilter:
    def __init__(dynamics, measurement_model):
        self.dynamics = dynamics
        self.measurement_model = measurement_model
    
    def propagate_prior(x):
        return x.shove_through_transform(self.dynamics)

    def update_estimate_with_measurement(estimate, measurement):
        prior = estimate.shove_through_transform(self.dynamics)
        measurement_prediction = prior.shove_through_transform(self.measurement_model)
        cross_covariance = prior.cross_covariance(self.measurement_model)
        
        innovation = measurement - measurement_prediction.mean
        filter_gain = cross_covariance @ inv(measurement_prediction.covariance)

        posterior_mean = prior.mean + filter_gain @ innovation
        posterior_covariance = prior_covariance - filter_gain @ cross_covariance.T

        return pr.GaussianRepresentation(posterior_mean, posterior_covariance)

class UnscentedKalmanFilter:
    def __init__():
        pass

class ParticleFilter:
    def __init__():
        pass
