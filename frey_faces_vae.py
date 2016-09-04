import sys

import theano.tensor as T
from theano.tensor import exp
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import lasagne
from lasagne.nonlinearities import linear, sigmoid, tanh

from VB.vae import VAE
from VB.run import Run

import utilities.densities
import utilities.nn
import utilities.utilities


main_dir = sys.argv[1]
out_dir = sys.argv[2]


srng = RandomStreams()


class MVGaussian(object):

    def __init__(self, h_dim, x_dim, hid_dim):

        self.h_dim = h_dim
        self.x_dim = x_dim
        self.hid_dim = hid_dim

        self.means_nn = self.means_nn_fn()
        self.covs_nn = self.covs_nn_fn()

    def get_params(self):

        means_params = lasagne.layers.get_all_params(self.means_nn)
        covs_params = lasagne.layers.get_all_params(self.covs_nn)

        return means_params + covs_params

    def get_param_values(self):

        means_params = lasagne.layers.get_all_param_values(self.means_nn)
        covs_params = lasagne.layers.get_all_param_values(self.covs_nn)

        return [means_params, covs_params]

    def set_param_values(self, param_values):

        [means_params, covs_params] = param_values

        lasagne.layers.set_all_param_values(self.means_nn, means_params)
        lasagne.layers.set_all_param_values(self.covs_nn, covs_params)

    def means_nn_fn(self):

        nonlinearities = [tanh, tanh, sigmoid]

        nn = utilities.nn.nn(self.h_dim, [self.hid_dim, self.hid_dim, self.x_dim], nonlinearities)

        return nn

    def covs_nn_fn(self):

        nonlinearities = [tanh, tanh, exp]

        nn = utilities.nn.nn(self.h_dim, [self.hid_dim, self.hid_dim, self.x_dim], nonlinearities)

        return nn

    def log_p_h(self, H):

        log_p_h = utilities.densities.log_diagonal_gaussian(H, T.zeros_like(H), T.ones_like(H))

        return log_p_h

    def log_p_x(self, H, X, det=False):

        means = lasagne.layers.get_output(self.means_nn, H, deterministic=det)
        covs = lasagne.layers.get_output(self.covs_nn, H, deterministic=det)

        log_p_x = utilities.densities.log_diagonal_gaussian(X, means, covs)

        return log_p_x

    def generate_output_prior(self):

        h = srng.normal([self.h_dim])
        e = srng.normal([self.x_dim])

        means = lasagne.layers.get_output(self.means_nn, h, deterministic=True)
        covs = lasagne.layers.get_output(self.covs_nn, h, deterministic=True)

        sample = means + (T.sqrt(covs) * e)

        return sample.eval()

    def generate_output_posterior(self, h):

        e = srng.normal([self.x_dim])

        means = lasagne.layers.get_output(self.means_nn, h, deterministic=True)
        covs = lasagne.layers.get_output(self.covs_nn, h, deterministic=True)

        sample = means + (T.sqrt(covs) * e)

        return sample.eval()


class MeanField(object):

    def __init__(self, h_dim, x_dim, hid_dim):

        self.h_dim = h_dim
        self.x_dim = x_dim
        self.hid_dim = hid_dim

        self.means_h_nn = self.means_h_nn_fn()
        self.covs_h_nn = self.covs_h_nn_fn()

    def get_params(self):

        means_h_params = lasagne.layers.get_all_params(self.means_h_nn)
        covs_h_params = lasagne.layers.get_all_params(self.covs_h_nn)

        return means_h_params + covs_h_params

    def get_param_values(self):

        means_h_params = lasagne.layers.get_all_param_values(self.means_h_nn)
        covs_h_params = lasagne.layers.get_all_param_values(self.covs_h_nn)

        return [means_h_params, covs_h_params]

    def set_param_values(self, param_values):

        [means_h_params, covs_h_params] = param_values

        lasagne.layers.set_all_param_values(self.means_h_nn, means_h_params)
        lasagne.layers.set_all_param_values(self.covs_h_nn, covs_h_params)

    def means_h_nn_fn(self):

        nonlinearities = [tanh, tanh, linear]

        nn = utilities.nn.nn(self.x_dim, [self.hid_dim, self.hid_dim, self.h_dim], nonlinearities)

        return nn

    def covs_h_nn_fn(self):

        nonlinearities = [tanh, tanh, exp]

        nn = utilities.nn.nn(self.x_dim, [self.hid_dim, self.hid_dim, self.h_dim], nonlinearities)

        return nn

    def get_samples_latents(self, X, det=False):

        if X.ndim == 1:
            X = X.reshape((1, X.shape[0]))
            e = srng.normal([self.h_dim])
        else:
            e = srng.normal([X.shape[0], self.h_dim])

        means = lasagne.layers.get_output(self.means_h_nn, X, deterministic=det)
        covs = lasagne.layers.get_output(self.covs_h_nn, X, deterministic=det)

        samples = means + (T.sqrt(covs) * e)

        return samples

    def entropies_latents(self, H, X, det=False):

        if X.ndim == 1:
            X = X.reshape((1, X.shape[0]))

        means = lasagne.layers.get_output(self.means_h_nn, X, deterministic=det)
        covs = lasagne.layers.get_output(self.covs_h_nn, X, deterministic=det)

        entropies = -utilities.densities.log_diagonal_gaussian(H, means, covs)

        return entropies


solver = VAE

d_h = 25
d_hid = 100
d_x = 560

solver_kwargs = {'generative_model': MVGaussian,
                 'recognition_model': MeanField,
                 'h_dim': d_h,
                 'x_dim': d_x,
                 'hid_dim': d_hid,
                 }

pre_trained = False

train = True

training_iterations = 100000
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

    run = Run(solver=solver, solver_kwargs=solver_kwargs, dataset='frey_faces', main_dir=main_dir, out_dir=out_dir,
              pre_trained=pre_trained)

    if train:
        run.train(vae_or_iwae='vae', n_iter=training_iterations, batch_size=training_batch_size,
                  num_samples=training_num_samples, update=update, update_kwargs=update_kwargs, val_freq=val_freq,
                  val_batch_size=val_batch_size, val_num_samples=val_num_samples)

    if generate_output:
        run.generate_output(num_outputs)

    if test:
        run.test(test_batch_size, test_num_samples)
