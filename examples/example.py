# This takes the example from https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/
# and implements it using my complicated library...

import numpy as np

import probability_representations as pr
import transformation as tr
import systems as sy
import filtering as ft

import matplotlib.pyplot as plt

dt = 0.5
theta = 20*np.pi/180
F_real = np.array([[np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]])
F_guess = np.array([[np.cos(theta) + 0.3, -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]])
Q = pr.covariance_from_stds([0.5, 0.2])

H = np.array([[1, 0]])
R = pr.covariance_from_stds([2])

x0 = np.array([1, 2])
x0_guess = np.array([1.4, 2.6])
P0 = pr.covariance_from_stds([2,3])
prior = pr.GaussianRepresentation(x0_guess, P0)

dynamics = tr.LinearTransformation(F_real)
dynamics_model = tr.NoisyLinearTransformation(F_guess, Q)
measurement_model = tr.NoisyLinearTransformation(H, R)

real_model = sy.SimulationStep(dynamics, measurement_model)
(states, measurements) = real_model.run_simulation(x0, 20)

kf_model = ft.KalmanFilterStep(dynamics_model, measurement_model)

solution = ft.KalmanFilter.run_single_model(prior, measurements, kf_model)
