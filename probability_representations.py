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

    def shove_through_transform_unscented(self, transform):
        sigma_r = SigmaRepresentation.from_gaussian_representation(self)

    @property
    def dimension(self): return len(self.mean)

    @classmethod
    def just_noise(cls, covariance):
        dim = covariance.shape[0]
        return cls(np.zeros(dim), covariance)

    @classmethod
    def from_sigma_representation(cls, sigma_r):
        return cls(sigma_r.mean, sigma_r.covariance)

    def confidence_ellipse(p=.99):
        s = -2*np.log(1-p)
        evals, evecs = np.linalg.eig(cov*s)
        sqevals = np.diag(np.sqrt(evals))

        a = np.dot(evecs, sqevals)
        e = np.array([np.dot(a, [np.cos(t), np.sin(t)]) for t in np.linspace(0, 2*np.pi)])
        xs = e[:,0] + mean[0]
        ys = e[:,1] + mean[1]
        return (xs, ys)


    def __str__(self):
        return "Mean:\n" + str(self.mean) + "\n\nCovariance:\n" + str(self.covariance)

class GaussianProduct:
    def __init__(self, mean1, mean2, cov1, cov2, cross_cov=None):
        dim1 = len(mean1)
        dim2 = len(mean2)

        if cross_covariance==None:
            cross_covariance = np.zeros((dim1, dim2))


        self.dim1 = dim1
        self.dim2 = dim2

        self.mean1 = mean1
        self.mean2 = mean2

        self.cov1 = cov1
        self.cov2 = cov2

        self.cross_cov = cross_cov

    @property
    def dimension(self): return self.dim1 + self.dim2

    @property
    def mean(self): return np.block([self.mean1, self.mean2])

    @property
    def covariance(self):
        cov1 = self.cov1
        cov2 = self.cov2
        cross = self.cross_cov
        return np.block([[cov1, cross],[cross.T, cov2]])

    def as_augented_gaussian(self):
        return GaussianRepresentation(self.mean, self.covariance)

    @classmethod
    def from_augmented_gaussian(cls, g, dim1):
        mean1 = g.mean[:dim1]
        mean2 = g.mean[dim1:]

        cov1 = g.covariance[:dim1, :dim1]
        cov2 = g.covariance[dim1:, dim1:]
        cross_cov = g.covariance[dim1:, :dim1]
        return cls(mean1, mean2, cov1, cov2, cross_cov)

    @classmethod
    def from_gaussians(cls, g1, g2, cross_cov=None):
        mean1 = g1.mean
        mean2 = g2.mean
        cov1 = g1.covariance
        cov2 = g2.covariance
        return cls(mean1, mean2, cov1, cov2, cross_cov)

    @property
    def swap(self):
        mean1 = self.mean2
        mean2 = self.mean1

        cov1 = self.cov2
        cov2 = self.cov1
        cross_cov = self.cross_cov.T

        return GaussianProduct(mean1, mean2, cov1, cov2, cross_cov)

    def update(self, z):
        xbar = self.mean1
        zbar = self.mean2
        innov = z - zbar

        Pxx = self.cov1
        Pzz = self.cov2
        Pxz = self.cross_cov

        Pzz_inv = np.linalg.inv(Pzz)

        mean = xbar + Pxz @ Pzz_inv @ innov
        covariance = Pxx - Pxz @ Pzz_inv @ Pxz.T
        return GaussianRepresentation(mean, covariance)

    @classmethod
    def from_noisy_transform(cls, g1, f):
        g2 = g1.shove_through_transform(f)
        cross_cov = g1.covariance @ f.matrix.T

        return cls.from_gaussians(g1, g2, cross_cov)

    @classmethod
    def lift_noisy_transform(cls, trans):
        def func(g): return cls.from_noisy_transform(g, trans)
        return func


class SigmaRepresentation:
    def __init__(self, sigma_points, mean_weights, covariance_weights):
        """sigma_points is a list of points.
        """
        self.sigma_points = sigma_points
        self.mean_weights = mean_weights
        self.covariance_weights = covariance_weights

    @classmethod
    def from_gaussian_representation(cls, g):
        (ps, mws, cws) = symmetric_set_from_humphreys(g.dimension, g.mean, g.covariance)
        return cls(ps, mws, cws)

    # The following routines are collected from Table I of menegaz2015unscented

    def symmetric_sigma_points(mean, covariance, const):
        sconst = sqrt(const)

        S = list(cholesky(covariance).T)
        deltas = [sconst * si for si in S]

        sigma_points = [mean]
        sigma_points += [mean + d for d in deltas]
        sigma_points += [mean - d for d in deltas]

        return sigma_points

    def symmetric_weights(w0, w_rest):
        weights = [w0]
        weights += [w_rest] * (2*n)

        return weights

    def symmetric_set(mean, covariance, sigma_const, w0, w_rest):
        sigma_points = symmetric_sigma_points(mean, covariance, sigma_const)
        weights = symmetric_weights(w0, w_rest)
        return (sigma_points, weights)

    def symmetric_set_of_1(kappa):
        # let kappa > -n
        npk = n + kappa
        sigma_const = npk
        w0 = kappa / npk
        w_rest = 0.5/npk
        return symmetric_set(mean, covariance, sigma_const, w0, w_rest)

    def symmetric_set_of_7(w0):
        # let w0 < 1
        w_rest = (1-w0)/(2*n)
        sigma_const = n/(1-w0)
        return symmetric_set(mean, covariance, sigma_const, w0, w_rest)

    def symmetric_set_from_humphreys(n, mean, covariance, alpha=0.001, beta=2, kappa=0):
        lamb = alpha**2 * (n + kappa) - n
        npl = n + lamb
        sigma_const = npl

        w0m = lamb / npl
        wm_rest = 0.5/npl

        w0c = w0m + 1 - alpha**2 + beta
        wc_rest = wm_rest

        sigma_points = symmetric_sigma_points(mean, covariance, sigma_const)
        mean_weights = symmetric_weights(w0m, wm_rest)
        covariance_weights = symmetric_weights(w0c, wc_rest)

        return (sigma_points, mean_weights, covariance_weights)

    def symmetric_set_of_35(alpha=0.001, beta=2, kappa=0):
        lamb = alpha**2 * (n + kappa) - n
        npl = n + lamb
        sigma_const = npl

        w0m = lamb / npl
        wm_rest = w0m

        w0c = w0m + 1 - alpha**2 + beta
        wc_rest = wm_rest

        sigma_points = symmetric_sigma_points(mean, covariance, sigma_const)
        mean_weights = symmetric_weights(w0m, wm_rest)
        covariance_weights = symmetric_weights(w0c, wc_rest)

        return (sigma_points, mean_weights, covariance_weights)

    def reduced_set_of_8():
        pass

    def spherical_simplex_set_of_9():
        pass

    def simplex_set_of_34():
        pass

    def minimum_set_of_12():
        pass

    def fifth_order_set_of_36():
        pass

    def set_of_37():
        pass

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

    def shove_through_noisy_transform_augmented(self, transform):
        pass

    def confidence_ellipse(p=.99):
        return GaussianRepresentation.from_sigma_representation(self).confidence_ellipse()

    def __str__(self):
        return "Mean:\n" + str(self.mean) + "\n\nCovariance:\n" + str(self.covariance)


class RiemannianSigmaRepresentation:
    def __init__(self, sigma_points, mean_weights, covariance_weights, manifold):
        """sigma_points is a list of points.
        """
        self.sigma_points = sigma_points
        self.mean_weights = mean_weights
        self.covariance_weights = covariance_weights
        self.manifold = manifold

    @property
    def mean(self):
        manifold = self.manifold
        max_iter=256
        precision=1e-5

        y = self.sigma_points[0]
        for i in range(max_iter):
            y_prev = y
            xys = [manifold.metric.log(point=p, base_point=y_prev) for p in self.sigma_points]
            weightedxys = [w*xy for w, xy in zip(self.mean_weights, xys)]
            v = sum(weightedxys)
            y = manifold.metric.exp(base_point=y_prev, tangent_vec=v)

            #tangent_vec = manifold.projection_to_tangent_space(
            #        vector=euclidean_grad, base_point=x)
            #x = manifold.metric.exp(base_point=x, tangent_vec=tangent_vec)[0]
            if (np.sqrt(manifold.metric.squared_norm(v)) <= precision):
                break

        return y

    @property
    def covariance(self):
        logs = [self.manifold.metric.log(point=p, base_point=self.mean) for p in self.sigma_points]
        weightedouterlogs = [w * np.outer(l[0], l[0]) for w, l in zip(self.covariance_weights, logs)]
        return sum(weightedouterlogs)

    @classmethod
    def from_gaussian_representation(cls):
        pass

    def confidence_ellipse(p=.99):
        mean = self.mean
        cov = self.covariance
        manifold = self.manifold

        s = -2*np.log(1-p)
        evals, evecs = np.linalg.eig(cov*s)
        sqevals = np.sqrt(evals)
        idxs = ~np.isnan(sqevals)
        sqevals = sqevals[idxs]
        evecs = evecs[:,idxs]
        ev_list = zip(sqevals, evecs.T)
        ev_list = sorted(ev_list, key=lambda a:a[0], reverse=True)
        sqevals, evecs = zip(*ev_list)
        es = [sqevals[0]*evecs[0]*np.cos(t) + 
            sqevals[1]*evecs[1]*np.sin(t) for t in np.linspace(0, 2*np.pi, num=500)]

        ems = [manifold.metric.exp(base_point=mean, tangent_vec=e) for e in es]
        return ems


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


class RiemannianGaussianRepresentation:
    def __init__(self):
        raise NotImplementedError


class RiemannianSampleRepresentation:
    def __init__(self):
        raise NotImplementedError


class RiemannianWeightedSampleRepresentation(SampleRepresentation):
    def __init__(self):
        raise NotImplementedError
