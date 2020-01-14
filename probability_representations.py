import numpy as np
from numpy.linalg import cholesky

def covariance_from_stds(stds):
    return np.diag(np.square(stds))

class GaussianRepresentation:
    def __init__(self, mean, covariance):
        self.mean = mean
        self.covariance = covariance

    def shove_through_transform(self, transform):
        new_mean = transform.push_point(self.mean)
        new_covariance = transform.push_tangent_bivector(self.mean, self.covariance)
        return GaussianRepresentation(new_mean, new_covariance)

    @property
    def dimension(self): return len(self.mean)

    def __str__(self):
        return "Mean:\n" + str(self.mean) + "\n\nCovariance:\n" + str(self.covariance)

class SigmaRepresentation:
    def __init__(self, sigma_points, mean_weights, covariance_weights):
        """sigma_points is a list of points.
        """
        self.sigma_points = sigma_points
        self.mean_weights = mean_weights
        self.covariance_weights = covariance_weights

    @classmethod
    def from_gaussian_representation(cls, g, alpha=0.001, beta=2, kappa=0):
        dim = g.dimension
        lamb = alpha**2 * (dim + kappa) - dim

        S = cholesky(g.covariance).T
        S = list(S)
        sigma_points = [g.mean]
        sigma_points += [g.mean + np.sqrt(dim + lamb) * si for si in S]
        sigma_points += [g.mean - np.sqrt(dim + lamb) * si for si in S]

        mean_weights = [0.5 / (dim + lamb)] * len(sigma_points)
        mean_weights[0] *= 2*lamb

        covariance_weights = mean_weights.copy()
        covariance_weights[0] += 1 - alpha**2 + beta

        return SigmaRepresentation(sigma_points, mean_weights, covariance_weights)


    @property 
    def dimension(self): return len(self.sigma_points[0])

    @property
    def mean(self):
        return sum([w * xi for (w, xi) in zip(self.mean_weights, self.sigma_points)])

    @property
    def covariance(self):
        mean = self.mean
        logs = [xi - mean for xi in self.sigma_points]
        outers = [np.outer(x,x) for x in logs]
        weights = self.covariance_weights
        return sum( [w*o for (w,o) in zip(weights, outers)] )

    def shove_through_transform(self, transform):
        new_points = [transform.push_point(p) for p in self.sigma_points]
        return SigmaRepresentation(new_points, self.mean_weights, self.covariance_weights)

    def shove_through_noisy_transform(self, transform):
        pass

    def __str__(self):
        return "Mean:\n" + str(self.mean) + "\n\nCovariance:\n" + str(self.covariance)

class SampleRepresentation:
    def __init__(self, samples):
        """samples must be a List.
        """
        self.samples = samples
        self.num_samples = len(samples)

    @classmethod
    def from_gaussian_representation(cls, gr, num=100):
        b = gr.mean
        dim = len(b)
        A = np.linalg.cholesky(gr.covariance)

        rands = np.random.normal(size=(num, dim))
        normalsamples = list(rands)
        samples = [A@x + b for x in normalsamples]
        return cls(samples)

    @property
    def mean(self): return sum(self.samples) / len(self.samples)

    @property
    def covariance(self):
        mean = self.mean
        samples = self.samples
        logs = [x - mean for x in samples]
        outers = [np.outer(x,x) for x in logs]

        return sum(outers) / (len(samples) - 1)

    def shove_through_transform(self, transform):
        new_points = [transform.push_point(p) for p in self.samples]
        return SampleRepresentation(new_points)

    def __str__(self):
        return "Mean:\n" + str(self.mean) + "\n\nCovariance:\n" + str(self.covariance)

class WeightedSampleRepresentation(SampleRepresentation):
    def __init__(self, samples, weights=None):
        super().__init__(samples)
        if weights==None:
            weights = [1.0 / self.num_samples for s in samples]
        self.weights = weights

    @property
    def mean(self):
        return sum([w * xi for (w, xi) in zip(self.weights, self.samples)])

    @property
    def covariance(self):
        mean = self.mean
        logs = [xi - mean for xi in self.samples]
        outers = [np.outer(x,x) for x in logs]
        weights = self.weights
        return sum( [w*o for (w,o) in zip(weights, outers)] )

    def shove_through_transform(self, transform):
        new_points = super().shove_through_transform(transform).samples
        return WeightedSampleRepresentation(new_points, self.weights)

    def __str__(self):
        return "Mean:\n" + str(self.mean) + "\n\nCovariance:\n" + str(self.covariance)
