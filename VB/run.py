import os
import cPickle
import time
import numpy as np

from lasagne.updates import adam

import data
import utilities.utilities


class Run(object):

    def __init__(self, solver, solver_kwargs, dataset, main_dir, out_dir, load_param_dir=None, pre_trained=False):

        self.solver_kwargs = solver_kwargs

        self.vb = solver(**self.solver_kwargs)

        self.main_dir = main_dir
        self.out_dir = out_dir
        self.load_param_dir = load_param_dir

        self.pre_trained = pre_trained

        if dataset == 'mnist':
            self.X_train = data.load_binarized_mnist(dir=self.main_dir, dataset='train')
            self.X_test = data.load_binarized_mnist(dir=self.main_dir, dataset='test')
        elif dataset == 'frey_faces':
            self.X_train, self.X_test = data.load_frey_faces(dir=self.main_dir)

        if self.pre_trained:
            with open(os.path.join(self.load_param_dir, 'gen_params.save'), 'rb') as f:
                self.vb.generative_model.set_param_values(cPickle.load(f))
            with open(os.path.join(self.load_param_dir, 'recog_params.save'), 'rb') as f:
                self.vb.recognition_model.set_param_values(cPickle.load(f))

    def train(self, vae_or_iwae, n_iter, batch_size, num_samples, update=adam, update_kwargs=None, val_freq=None,
              val_batch_size=0, val_num_samples=0, overdisp=False):

        if self.pre_trained:
            with open(os.path.join(self.load_param_dir, 'updates.save'), 'rb') as f:
                saved_update = cPickle.load(f)
        else:
            saved_update = None

        optimiser, updates = self.vb.optimiser(num_samples=num_samples, update=update, update_kwargs=update_kwargs,
                                               saved_update=saved_update)

        if val_freq is not None:

            elbo_fn = self.vb.elbo_fn(val_num_samples)

        for i in range(n_iter):

            start = time.clock()

            batch_indices = np.random.choice(self.X_train.shape[0], batch_size, replace=False)
            batch = self.X_train[batch_indices]

            if overdisp:
                overdisp_dim = np.random.choice(self.solver_kwargs['h_dim'])
                if vae_or_iwae == 'iwae':
                    overdisp_s = np.random.choice(num_samples)
                    elbo = optimiser(batch, overdisp_dim, overdisp_s)
                else:
                    elbo = optimiser(batch, overdisp_dim)
            else:
                elbo = optimiser(batch)

            print 'Iteration ' + str(i + 1) + ': ELBO = ' + str(elbo / batch_size) + ' (time taken = ' + \
                  str(time.clock() - start) + ' seconds)'

            if val_freq is not None and i % val_freq == 0:

                val_batch_indices = np.random.choice(self.X_test.shape[0], val_batch_size, replace=False)
                val_batch = self.X_test[val_batch_indices]

                print 'Test set ELBO = ' + str(elbo_fn(val_batch) / val_batch_size)

        print

        with open(os.path.join(self.out_dir, 'gen_params.save'), 'wb') as f:
            cPickle.dump(self.vb.generative_model.get_param_values(), f, protocol=cPickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'recog_params.save'), 'wb') as f:
            cPickle.dump(self.vb.recognition_model.get_param_values(), f, protocol=cPickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'updates.save'), 'wb') as f:
            cPickle.dump(updates, f, protocol=cPickle.HIGHEST_PROTOCOL)

    def generate_output(self, prior=True, posterior=True, num_outputs=100):

        if prior:

            outputs = []

            for i in range(num_outputs):
                outputs.append(self.vb.generate_output_prior())

            np.save(os.path.join(self.out_dir, 'generated_output_prior.npy'), outputs)

        if posterior:

            outputs = []

            for i in range(num_outputs):

                n = np.random.choice(self.X_train.shape[0])

                outputs.append(self.vb.generate_output_posterior(self.X_train[n]))

            np.save(os.path.join(self.out_dir, 'generated_output_posterior.npy'), outputs)

    def test(self, batch_size, num_samples):

        elbo_fn = self.vb.elbo_fn(num_samples)

        elbo = 0

        batches_complete = 0

        for batch in utilities.utilities.chunker(self.X_test, batch_size):

            start = time.clock()

            elbo += elbo_fn(batch)

            batches_complete += 1

            print 'Tested batches ' + str(batches_complete) + ' of ' + str(round(self.X_test.shape[0] / batch_size)) \
                  + '; test set ELBO so far = ' + str(elbo) \
                  + ' / ' + str(elbo / (batches_complete * batch_size)) + ' per obs.' \
                  + ' (time taken = ' + str(time.clock() - start) + ' seconds)'

        print 'Test set ELBO = ' + str(elbo)
