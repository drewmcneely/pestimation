import numpy as np
from numpy.linalg import inv

import transformation as tf, probability_representations as pr

import functools as ft
from itertools import accumulate, repeat

class KalmanFilterStep:
    def __init__(self, dynamics_model, measurement_model):
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
        return pr.GaussianProduct.updator_from_transform(mapping)

class KalmanFilter:
    def run_model_list(prior, measurements, kfsteps):
        updators = [step.updator for step in kfsteps]
        updators = zip(measurements, updators)

        def f(x, updator):
            z = updator[0]
            update = updator[1]
            return update(x, z)

        return accumulate(updators, func=f, initial=prior)

    def run_single_model(prior, measurements, model):
        updator = model.updator
        states = [prior]
        for meas in measurements:
            states += [updator(states[-1], meas)]
        return states


class UnscentedKalmanFilter:
    def __init__():
        pass

class ParticleFilter:
    def __init__():
        pass
