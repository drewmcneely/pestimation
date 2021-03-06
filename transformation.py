import numpy as np
from scipy.integrate import solve_ivp
import abc

def vectorized(func):
    def wrapper(z):
        return np.array(func(*z))
    return wrapper


class Transformation(abc.ABC):
    @abc.abstractmethod
    def push_point(self, x): pass

    @abc.abstractmethod
    def push_tangent_bivector(self, x, P): pass


class LinearTransformation(Transformation):
    def __init__(self, matrix):
        self.matrix = matrix

    @property
    def domain_dimension(self):
        return self.matrix.shape[1]

    @property
    def codomain_dimension(self):
        return self.matrix.shape[0]

    def push_point(self, x): return self.matrix @ x
    def simulate(self, x): return self.push_point(x)
    def push_tangent_vector(self, v): return self.matrix @ v
    def push_tangent_bivector(self, x, P): return self.matrix @ P @ self.matrix.T

    def shove_gaussian(self, g):
        mean = self.push_point(g.mean)
        covariance = self.push_tangent_bivector(g.covariance)

        return GaussianRepresentation(mean, covariance)

    @classmethod
    def compose(cls, T1, T2):
        return cls(T2.matrix @ T1.matrix)


class NoisyLinearTransformation(LinearTransformation):
    def __init__(self, matrix, noise):
        self.matrix = matrix
        self.noise = noise

    @property
    def underlying_transform(self):
        return LinearTransformation(self.matrix)

    def push_point(self, x):
        return self.underlying_transform.push_point(x)

    def simulate(self, x):
        dim = self.codomain_dimension
        noise_mean = np.zeros(dim)
        noise_cov = self.noise
        noise_vector = np.random.multivariate_normal(noise_mean, noise_cov)

        return self.push_point(x) + noise_vector

    def shove_gaussian(self, g):
        g1 = self.underlying_transform.shove_gaussian(g)
        g1.covariance += self.noise
        return g1

    def compose_with(T1, T2):
        #Note:
        #Same as treating T1 as a Gauss(F, Q) and shoving through T2!
        F = T1.matrix
        Q = T1.noise

        H = T2.matrix
        R = T2.noise

        matrix = H @ F
        noise = H @ Q @ H.T + R

        return NoisyLinearTransformation(matrix, noise)

    @classmethod
    def from_noisemap(cls, F, Gamma, Q):
        return cls(F, Gamma @ Q @ Gamma.T)

class AffineTransformation(LinearTransformation):
    def __init__(self, A, b):
        super().__init__(A)
        self.b = b

    def push_point(self, x): return super().push_point(x) + self.b


class NonlinearTransformation:
    def __init__(self, mapping):
        """
        mapping must be a function taking an X and returning a Y
        """
        self.mapping = mapping

    def push_point(self, x): return self.mapping(x)


class NoisyNonlinearTransformation:
    def __init__(self):
        pass


class NonlinearExtendedTransformation(NonlinearTransformation):
    def __init__(self, mapping, jacobian, inverse_mapping=None):
        """ Represents a smooth transform from the space X to the space Y.

        mapping must be a function taking an X and returning a Y
        jacobian must be a function taking an X and returning a dim(Y) by dim(X) matrix
        jacobian represents the jacobian of mapping evaluated at x

        self.differential is simply a wrapping of the jacobian into a LinearTransformation
        """
        super().__init__(mapping)
        self.jacobian = jacobian

        def differential(x): return LinearTransformation(jacobian(x))
        self.differential = differential
        self.inverse_mapping = inverse_mapping

    @property
    def inverse(self):
        def j(y): return np.linalg.inv(self.jacobian(self.inverse_mapping(y)))
        return NonlinearExtendedTransformation(self.inverse_mapping, j, self.mapping)

    def push_tangent_vector(self, x, v): return self.differential(x).push_tangent_vector(x, v)
    def push_tangent_bivector(self, x, P): return self.differential(x).push_tangent_bivector(x, P)


class NoisyNonlinearExtendedTransformation(NonlinearExtendedTransformation):
    def __init__(self, mapping, jacobian, noise_covariance, noise_mapping_matrix=None):
        codomain_dimension = noise_covariance.shape[0]
        super().__init__(mapping, jacobian)

        if noise_mapping_matrix==None:
            noise_mapping_matrix = np.identity(codomain_dimension)

        noise_mapping = LinearTransformation(noise_mapping_matrix)
        self.noise_uncertainty = noise_mapping.push_tangent_bivector(noise_covariance)

    def push_tangent_bivector(self, x, P): return super().push_tangent_bivector(x,P) + self.noise_uncertainty


class NoisyNonlinearDifferentialEquation:
    def __init__(self, derivative_model, jacobian, noise_covariance=None, noise_mapping_matrix=None):
        """
        derivative_model must be a function that takes a (t, x) and returns a vector in TX
        jacobian must be a function that takes a (t,x) and returns a matrix
        jacobian must be the jacobian of derivative_model
        noise_mapping_matrix is just a matrix for now
        noise_covariance is just a matrix for now
        """

        self.state_dimension = noise_mapping_matrix.shape[0]

        if noise_covariance==None:
            noise_covariance = np.identity(self.state_dimension)

        self.noise_dimension = noise_mapping_matrix.shape[1]

        self.f_tilde = derivative_model
        self.A = jacobian

        self.D = noise_mapping_matrix
        self.Q_tilde = noise_covariance

    def Fdot(self, t, x, F): return self.A(t, x) @ F
    def Gammadot(self, t, x, Gamma):
        return self.A(t,x) @ Gamma + self.D

    @staticmethod
    def augment_state(x, F, Gamma):
        return np.concatenate((x, F.flatten(), Gamma.flatten()))

    def parse_augmented_state(self, X):
        dim_x = self.state_dimension
        dim_F = dim_x**2
        dim_Gamma = dim_x * self.noise_dimension

        x = X[:dim_x]
        F = X[dim_x:-dim_Gamma].reshape((dim_x, dim_x))
        Gamma = X[-dim_Gamma:].reshape((dim_x, self.noise_dimension))

        return (x, F, Gamma)

    def ode_func(self, t, X):
        (x, F, Gamma) = self.parse_augmented_state(X)
        xdot = self.f_tilde(t,x)
        Fdot = self.Fdot(t,x,F)
        Gammadot = self.Gammadot(t,x,Gamma)
        return self.augment_state(xdot, Fdot, Gammadot)

    def initial_augmented_state(self, x0):
        F0 = np.identity(self.state_dimension)
        Gamma0 = np.zeros((self.state_dimension, self.noise_dimension))
        return self.augment_state(x0, F0, Gamma0)

    def propagate(self, x0, T):
        fun = self.ode_func
        t_span = (0,T)
        y0 = self.initial_augmented_state(x0)
        result = solve_ivp(fun, t_span, y0)
        X = result.y[:,-1].flatten()
        (x, F, Gamma) = self.parse_augmented_state(X)
        Q = self.Q_tilde / T
        return (x,F,Gamma,Q)
