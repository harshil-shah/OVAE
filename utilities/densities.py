import numpy as np
import theano.tensor as T


def log_bernoulli(X, P):

    log_densities = T.sum((X * T.log(P)) + ((1 - X) * T.log(1 - P)), axis=1)

    return log_densities


def log_scalar_gaussian(x, means, covs, chol=False):

    if chol:
        covs **= 2

    log_normalisers = - 0.5 * T.log(2. * np.pi * covs)

    log_densities = - 0.5 * (((x - means)**2) / covs)

    return log_normalisers + log_densities


def log_diagonal_gaussian(X, means, covs, chol=False):

    if chol:
        covs **= 2

    log_normalisers = - 0.5 * T.sum(T.log(2. * np.pi * covs), axis=1)

    log_densities = - 0.5 * T.sum(((X - means)**2) / covs, axis=1)

    return log_normalisers + log_densities
