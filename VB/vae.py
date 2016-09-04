import numpy as np
import theano
import theano.tensor as T


class VAE(object):

    def __init__(self, generative_model, recognition_model, h_dim, x_dim, hid_dim):

        self.generative_model = generative_model(h_dim, x_dim, hid_dim)
        self.recognition_model = recognition_model(h_dim, x_dim, hid_dim)

    def elbo_fn(self, num_samples):

        batch = T.matrix('batch')
        batch_rep = T.repeat(batch, num_samples, axis=0)

        h_rep = self.recognition_model.get_samples_latents(batch_rep, True)

        log_p_h = self.generative_model.log_p_h(h_rep)
        log_p_x = self.generative_model.log_p_x(h_rep, batch_rep, True)
        entropies_h = self.recognition_model.entropies_latents(h_rep, batch_rep, True)

        elbos_rep = log_p_h + log_p_x + entropies_h
        elbos_matrix = elbos_rep.reshape((batch.shape[0], num_samples))

        elbo = T.sum(T.mean(elbos_matrix, axis=1, keepdims=True))

        elbo_fn = theano.function(inputs=[batch],
                                  outputs=elbo,
                                  )

        return elbo_fn

    def optimiser(self, num_samples, update, update_kwargs, saved_update=None):

        batch = T.matrix('batch')
        batch_rep = T.repeat(batch, num_samples, axis=0)

        h_rep = self.recognition_model.get_samples_latents(batch_rep)

        log_p_h = self.generative_model.log_p_h(h_rep)
        log_p_x = self.generative_model.log_p_x(h_rep, batch_rep)
        entropies_h = self.recognition_model.entropies_latents(h_rep, batch_rep)

        elbos_rep = log_p_h + log_p_x + entropies_h
        elbos_matrix = elbos_rep.reshape((batch.shape[0], num_samples))

        elbo = T.sum(T.mean(elbos_matrix, axis=1, keepdims=True))

        params = self.generative_model.get_params() + self.recognition_model.get_params()
        grads = T.grad(-elbo, params)

        update_kwargs['loss_or_grads'] = grads
        update_kwargs['params'] = params

        updates = update(**update_kwargs)

        if saved_update is not None:
            for u, v in zip(updates, saved_update.keys()):
                u.set_value(v.get_value())

        optimiser = theano.function(inputs=[batch],
                                    outputs=elbo,
                                    updates=updates,
                                    )

        return optimiser, updates

    def generate_output_prior(self):

        return self.generative_model.generate_output_prior()

    def generate_output_posterior(self, x):

        h = self.recognition_model.get_samples_latents(x, True)

        return self.generative_model.generate_output_posterior(h)


class OVAE(object):

    def __init__(self, generative_model, recognition_model, h_dim, x_dim, hid_dim, tau):

        self.generative_model = generative_model(h_dim, x_dim, hid_dim)
        self.recognition_model = recognition_model(h_dim, x_dim, hid_dim, tau*np.ones(h_dim))

    def elbo_fn(self, num_samples):

        batch = T.matrix('batch')
        batch_rep = T.repeat(batch, num_samples, axis=0)

        h_rep = self.recognition_model.get_samples_latents_regular(batch_rep, True)

        log_p_h = self.generative_model.log_p_h(h_rep)
        log_p_x = self.generative_model.log_p_x(h_rep, batch_rep, True)
        entropies_h = self.recognition_model.entropies_latents(h_rep, batch_rep, True)

        elbos_rep = log_p_h + log_p_x + entropies_h
        elbos_matrix = elbos_rep.reshape((batch.shape[0], num_samples))

        elbo = T.sum(T.mean(elbos_matrix, axis=1, keepdims=True))

        elbo_fn = theano.function(inputs=[batch],
                                  outputs=elbo,
                                  )

        return elbo_fn

    def optimiser(self, num_samples, update, update_kwargs, saved_update=None):

        latent_dim = T.bscalar('latent_dim')

        batch = T.matrix('batch')
        batch_rep = T.repeat(batch, num_samples, axis=0)

        h_regular_rep = self.recognition_model.get_samples_latents_regular(batch_rep)
        h_overdisp_rep = self.recognition_model.get_samples_latents_overdisp(batch_rep, latent_dim)
        h_rep = T.set_subtensor(h_regular_rep[:, latent_dim], h_overdisp_rep)

        log_p_h = self.generative_model.log_p_h(h_rep)
        log_p_x = self.generative_model.log_p_x(h_rep, batch_rep)
        entropies_h = self.recognition_model.entropies_latents(h_rep, batch_rep)

        imp_wts = self.recognition_model.importance_weights_latents(h_overdisp_rep, batch_rep, latent_dim)

        elbos_rep = imp_wts * (log_p_h + log_p_x + entropies_h)
        elbos_matrix = elbos_rep.reshape((batch.shape[0], num_samples))

        elbo = T.sum(T.mean(elbos_matrix, axis=1, keepdims=True))

        params = self.generative_model.get_params() + self.recognition_model.get_params()[:-1]
        grads = T.grad(-elbo, params)

        tau = self.recognition_model.get_params()[-1]

        all_grads, _ = theano.scan(lambda s, E: T.grad(-T.sum(E[s]), params),
                                   sequences=[T.arange(elbos_matrix.T.shape[0])],
                                   non_sequences=[elbos_matrix.T],
                                   )

        variance = T.sum([T.sum(T.var(g, axis=0)) for g in all_grads])

        grad_tau = T.grad(variance, tau)

        grads += [grad_tau]

        params = self.generative_model.get_params() + self.recognition_model.get_params()

        update_kwargs['loss_or_grads'] = grads
        update_kwargs['params'] = params

        updates = update(**update_kwargs)

        if saved_update is not None:
            for u, v in zip(updates, saved_update.keys()):
                u.set_value(v.get_value())

        optimiser = theano.function(inputs=[batch, latent_dim],
                                    outputs=elbo,
                                    updates=updates,
                                    )

        return optimiser, updates

    def generate_output_prior(self):

        return self.generative_model.generate_output_prior()

    def generate_output_posterior(self, x):

        h = self.recognition_model.get_samples_latents_regular(x, True)

        return self.generative_model.generate_output_posterior(h)


class IWAE(object):

    def __init__(self, generative_model, recognition_model, h_dim, x_dim, hid_dim):

        self.generative_model = generative_model(h_dim, x_dim, hid_dim)
        self.recognition_model = recognition_model(h_dim, x_dim, hid_dim)

    def elbo_fn(self, num_samples):

        batch = T.matrix('batch')
        batch_rep = T.repeat(batch, num_samples, axis=0)

        h_rep = self.recognition_model.get_samples_latents(batch_rep, True)

        log_p_h = self.generative_model.log_p_h(h_rep)
        log_p_x = self.generative_model.log_p_x(h_rep, batch_rep, True)
        entropies_h = self.recognition_model.entropies_latents(h_rep, batch_rep, True)

        log_iw_rep = log_p_h + log_p_x + entropies_h

        log_iw_matrix = log_iw_rep.reshape((batch.shape[0], num_samples))
        log_iw_max = T.max(log_iw_matrix, axis=1, keepdims=True)
        log_iw_minus_max = log_iw_matrix - log_iw_max

        elbo = T.sum(log_iw_max + T.log(T.mean(T.exp(log_iw_minus_max), axis=1, keepdims=True)))

        elbo_fn = theano.function(inputs=[batch],
                                  outputs=elbo,
                                  )

        return elbo_fn

    def optimiser(self, num_samples, update, update_kwargs, saved_update=None):

        batch = T.matrix('batch')
        batch_rep = T.repeat(batch, num_samples, axis=0)

        h_rep = self.recognition_model.get_samples_latents(batch_rep)

        log_p_h = self.generative_model.log_p_h(h_rep)
        log_p_x = self.generative_model.log_p_x(h_rep, batch_rep)
        entropies_h = self.recognition_model.entropies_latents(h_rep, batch_rep)

        log_w_rep = log_p_h + log_p_x + entropies_h
        log_w_matrix = log_w_rep.reshape((batch.shape[0], num_samples))

        log_w_minus_max = log_w_matrix - T.max(log_w_matrix, axis=1, keepdims=True)
        w_matrix = T.exp(log_w_minus_max)
        w_normalized_matrix = w_matrix / T.sum(w_matrix, axis=1, keepdims=True)
        w_normalized_rep = T.reshape(w_normalized_matrix, log_w_rep.shape)

        params = self.generative_model.get_params() + self.recognition_model.get_params()

        dummy_vec = T.vector(dtype=theano.config.floatX)
        grads = theano.clone(T.grad(-T.dot(log_w_rep, dummy_vec), params), replace={dummy_vec: w_normalized_rep})

        update_kwargs['loss_or_grads'] = grads
        update_kwargs['params'] = params

        updates = update(**update_kwargs)

        if saved_update is not None:
            for u, v in zip(updates, saved_update.keys()):
                u.set_value(v.get_value())

        optimiser = theano.function(inputs=[batch],
                                    outputs=T.dot(log_w_rep, w_normalized_rep),
                                    updates=updates,
                                    )

        return optimiser, updates

    def generate_output_prior(self):

        return self.generative_model.generate_output_prior()

    def generate_output_posterior(self, x):

        h = self.recognition_model.get_samples_latents(x, True)

        return self.generative_model.generate_output_posterior(h)


class OIWAE(object):

    def __init__(self, generative_model, recognition_model, h_dim, x_dim, hid_dim, tau):

        self.generative_model = generative_model(h_dim, x_dim, hid_dim)
        self.recognition_model = recognition_model(h_dim, x_dim, hid_dim, tau*np.ones(h_dim))

    def elbo_fn(self, num_samples):

        batch = T.matrix('batch')
        batch_rep = T.repeat(batch, num_samples, axis=0)

        h_rep = self.recognition_model.get_samples_latents_regular(batch_rep, True)

        log_p_h = self.generative_model.log_p_h(h_rep)
        log_p_x = self.generative_model.log_p_x(h_rep, batch_rep, True)
        entropies_h = self.recognition_model.entropies_latents(h_rep, batch_rep, True)

        log_iw_rep = log_p_h + log_p_x + entropies_h

        log_iw_matrix = log_iw_rep.reshape((batch.shape[0], num_samples))
        log_iw_max = T.max(log_iw_matrix, axis=1, keepdims=True)
        log_iw_minus_max = log_iw_matrix - log_iw_max

        elbo = T.sum(log_iw_max + T.log(T.mean(T.exp(log_iw_minus_max), axis=1, keepdims=True)))

        elbo_fn = theano.function(inputs=[batch],
                                  outputs=elbo,
                                  )

        return elbo_fn

    def optimiser(self, num_samples, update, update_kwargs, saved_update=None):

        overdisp_dim = T.bscalar('overdisp_dim')
        overdisp_s = T.bscalar('overdisp_s')

        batch = T.matrix('batch')
        batch_rep = T.repeat(batch, num_samples, axis=0)

        h_regular_rep = self.recognition_model.get_samples_latents_regular(batch_rep)
        h_overdisp = self.recognition_model.get_samples_latents_overdisp(batch, overdisp_dim)
        h_rep = T.set_subtensor(h_regular_rep[overdisp_s::num_samples, overdisp_dim], h_overdisp)

        log_p_h = self.generative_model.log_p_h(h_rep)
        log_p_x = self.generative_model.log_p_x(h_rep, batch_rep)
        entropies_h = self.recognition_model.entropies_latents(h_rep, batch_rep)

        log_w_rep = log_p_x + entropies_h + log_p_h
        log_w_matrix = log_w_rep.reshape((batch.shape[0], num_samples))

        v = self.recognition_model.importance_weights_latents(h_overdisp, batch, overdisp_dim)

        log_u_matrix = T.repeat(v, num_samples).reshape((batch.shape[0], num_samples)) + log_w_matrix
        log_u_rep = log_u_matrix.flatten()

        log_u_minus_max = log_u_matrix - T.max(log_u_matrix, axis=1, keepdims=True)
        u_matrix = T.exp(log_u_minus_max)
        u_normalized_matrix = u_matrix / T.sum(u_matrix, axis=1, keepdims=True)
        u_normalized_rep = T.reshape(u_normalized_matrix, log_w_rep.shape)

        params = self.generative_model.get_params() + self.recognition_model.get_params()[:-1]

        dummy_vec = T.vector(dtype=theano.config.floatX)
        grads = theano.clone(T.grad(-T.dot(log_u_rep, dummy_vec), params), replace={dummy_vec: u_normalized_rep})

        tau = self.recognition_model.get_params()[-1]

        all_grads, _ = theano.scan(lambda s, log_u, u_norm: theano.clone(T.grad(-T.dot(log_u[s], dummy_vec), params),
                                                                         replace={dummy_vec: u_norm[s]}),
                                   sequences=[T.arange(log_u_matrix.T.shape[0])],
                                   non_sequences=[log_u_matrix.T, u_normalized_matrix.T],
                                   )

        variance = T.sum([T.sum(T.var(g, axis=0)) for g in all_grads])

        grad_tau = T.grad(variance, tau)

        grads += [grad_tau]

        params = self.generative_model.get_params() + self.recognition_model.get_params()

        update_kwargs['loss_or_grads'] = grads
        update_kwargs['params'] = params

        updates = update(**update_kwargs)

        if saved_update is not None:
            for u_matrix, v in zip(updates, saved_update.keys()):
                u_matrix.set_value(v.get_value())

        optimiser = theano.function(inputs=[batch, overdisp_dim, overdisp_s],
                                    outputs=T.dot(log_u_rep, u_normalized_rep),
                                    updates=updates,
                                    )

        return optimiser, updates

    def generate_output_prior(self):

        return self.generative_model.generate_output_prior()

    def generate_output_posterior(self, x):

        h = self.recognition_model.get_samples_latents_regular(x, True)

        return self.generative_model.generate_output_posterior(h)
