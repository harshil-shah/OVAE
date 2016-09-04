import sys

import numpy as np
import theano
import theano.tensor as T
from theano.tensor import exp
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import lasagne
from lasagne.nonlinearities import linear, sigmoid, tanh

from VB.vae import OIWAE
from VB.run import Run

import utilities.densities
import utilities.nn
import utilities.utilities


main_dir = sys.argv[1]
out_dir = sys.argv[2]


srng = RandomStreams()


class MVBernoulli(object):

    def __init__(self, h_dim, x_dim, hid_dim):

        self.h_dim = h_dim
        self.x_dim = x_dim
        self.hid_dim = hid_dim

        self.bernoulli_probs_nn = self.bernoulli_probs_nn_fn()

    def get_params(self):

        bernoulli_probs_params = lasagne.layers.get_all_params(self.bernoulli_probs_nn)

        return bernoulli_probs_params

    def get_param_values(self):

        bernoulli_probs_params = lasagne.layers.get_all_param_values(self.bernoulli_probs_nn)

        return bernoulli_probs_params

    def set_param_values(self, param_values):

        lasagne.layers.set_all_param_values(self.bernoulli_probs_nn, param_values)

    def bernoulli_probs_nn_fn(self):

        nonlinearities = [tanh, tanh, sigmoid]

        nn = utilities.nn.nn(self.h_dim, [self.hid_dim, self.hid_dim, self.x_dim], nonlinearities)

        return nn

    def log_p_h(self, H):

        log_p_h = utilities.densities.log_diagonal_gaussian(H, T.zeros_like(H), T.ones_like(H))

        return log_p_h

    def log_p_x(self, H, X, det=False):

        bernoulli_probs = lasagne.layers.get_output(self.bernoulli_probs_nn, H, deterministic=det)

        log_p_x = utilities.densities.log_bernoulli(X, bernoulli_probs)

        return log_p_x

    def generate_output_prior(self, probs=True):

        h = srng.normal([self.h_dim])

        bernoulli_probs = lasagne.layers.get_output(self.bernoulli_probs_nn, h, deterministic=True)

        if probs:
            return bernoulli_probs.eval()
        else:
            return [int(np.random.uniform() <= p) for p in bernoulli_probs.eval()]

    def generate_output_posterior(self, h, probs=True):

        bernoulli_probs = lasagne.layers.get_output(self.bernoulli_probs_nn, h, deterministic=True)

        if probs:
            return bernoulli_probs.eval()
        else:
            return [int(np.random.uniform() <= p) for p in bernoulli_probs.eval()]


class MeanField(object):

    def __init__(self, h_dim, x_dim, hid_dim, tau):

        self.h_dim = h_dim
        self.x_dim = x_dim
        self.hid_dim = hid_dim

        self.tau = theano.shared(tau, 'tau')

        self.means_h_nn = self.means_h_nn_fn()
        self.covs_h_nn = self.covs_h_nn_fn()

    def get_params(self):

        means_h_params = lasagne.layers.get_all_params(self.means_h_nn)
        covs_h_params = lasagne.layers.get_all_params(self.covs_h_nn)

        return means_h_params + covs_h_params + [self.tau]

    def get_param_values(self):

        means_h_params = lasagne.layers.get_all_param_values(self.means_h_nn)
        covs_h_params = lasagne.layers.get_all_param_values(self.covs_h_nn)

        return [means_h_params, covs_h_params, self.tau.get_value()]

    def set_param_values(self, param_values):

        [means_h_params, covs_h_params, tau] = param_values

        lasagne.layers.set_all_param_values(self.means_h_nn, means_h_params)
        lasagne.layers.set_all_param_values(self.covs_h_nn, covs_h_params)

        self.tau.set_value(tau)

    def means_h_nn_fn(self):

        nonlinearities = [tanh, tanh, linear]

        nn = utilities.nn.nn(self.x_dim, [self.hid_dim, self.hid_dim, self.h_dim], nonlinearities)

        return nn

    def covs_h_nn_fn(self):

        nonlinearities = [tanh, tanh, exp]

        nn = utilities.nn.nn(self.x_dim, [self.hid_dim, self.hid_dim, self.h_dim], nonlinearities)

        return nn

    def get_samples_latents_regular(self, X, det=False):

        if X.ndim == 1:
            X = X.reshape((1, X.shape[0]))
            e = srng.normal([self.h_dim])
        else:
            e = srng.normal([X.shape[0], self.h_dim])

        means = lasagne.layers.get_output(self.means_h_nn, X, deterministic=det)
        covs = lasagne.layers.get_output(self.covs_h_nn, X, deterministic=det)

        samples = means + (T.sqrt(covs) * e)

        return samples

    def get_samples_latents_overdisp(self, X, dim, det=False):

        if X.ndim == 1:
            X = X.reshape((1, X.shape[0]))
            e = srng.normal([1])
        else:
            e = srng.normal([X.shape[0]])

        means = lasagne.layers.get_output(self.means_h_nn, X, deterministic=det)
        covs = lasagne.layers.get_output(self.covs_h_nn, X, deterministic=det)

        tau_dim = self.tau[dim]

        means_dim = means[:, dim]
        covs_dim = T.switch(tau_dim < 1, 1, tau_dim) * covs[:, dim].nonzero_values()

        samples = means_dim + (T.sqrt(covs_dim) * e)

        return samples

    def entropies_latents(self, H, X, det=False):

        if X.ndim == 1:
            X = X.reshape((1, X.shape[0]))

        means = lasagne.layers.get_output(self.means_h_nn, X, deterministic=det)
        covs = lasagne.layers.get_output(self.covs_h_nn, X, deterministic=det)

        entropies = -utilities.densities.log_diagonal_gaussian(H, means, covs)

        return entropies

    def importance_weights_latents(self, H, X, dim, det=False):

        if X.ndim == 1:
            X = X.reshape((1, X.shape[0]))

        means = lasagne.layers.get_output(self.means_h_nn, X, deterministic=det)
        covs = lasagne.layers.get_output(self.covs_h_nn, X, deterministic=det)

        tau_dim = self.tau[dim]

        means_dim = means[:, dim]
        covs_dim = T.switch(tau_dim < 1, 1, tau_dim) * covs[:, dim].nonzero_values()

        nums = utilities.densities.log_scalar_gaussian(H, means_dim, covs_dim)
        dens = utilities.densities.log_scalar_gaussian(H, means_dim,
                                                       T.switch(tau_dim < 1, 1, tau_dim) * covs_dim)

        imp_wts = nums / dens

        return imp_wts


solver = OIWAE

d_h = 50
d_hid = 200
d_x = 784

solver_kwargs = {'generative_model': MVBernoulli,
                 'recognition_model': MeanField,
                 'h_dim': d_h,
                 'x_dim': d_x,
                 'hid_dim': d_hid,
                 'tau': 2,
                 }

pre_trained = False

train = True

training_iterations = 300000
training_batch_size = 20
training_num_samples = 5

update = lasagne.updates.adam
update_kwargs = {'learning_rate': 1e-4}

val_freq = 50
val_batch_size = 20
val_num_samples = 5


generate_output = True
num_outputs = 100


test = True
test_batch_size = 10
test_num_samples = 5000


if __name__ == '__main__':

    run = Run(solver=solver, solver_kwargs=solver_kwargs, dataset='mnist', main_dir=main_dir, out_dir=out_dir,
              load_param_dir=param_dir, pre_trained=pre_trained)

    if train:
        run.train(vae_or_iwae='iwae', n_iter=training_iterations, batch_size=training_batch_size,
                  num_samples=training_num_samples, update=update, update_kwargs=update_kwargs, val_freq=val_freq,
                  val_batch_size=val_batch_size, val_num_samples=val_num_samples, overdisp=True)

    if generate_output:
        run.generate_output(num_outputs)

    if test:
        run.test(test_batch_size, test_num_samples)
