import lasagne


def nn(input_shape, dims, nonlinearities, W_inits=False, b_inits=False, batch_norms=False):

    if type(input_shape) is int:
        input_shape = (input_shape,)

    if type(nonlinearities) is not list:
        nonlinearities = [nonlinearities] * len(dims)

    if type(W_inits) is not list:
        W_inits = [W_inits] * len(dims)

    if type(b_inits) is not list:
        b_inits = [b_inits] * len(dims)

    if type(batch_norms) is not list:
        batch_norms = [batch_norms] * len(dims)

    nn = lasagne.layers.InputLayer(shape=(None,) + input_shape)

    for l in range(len(dims)):

        W_init = W_inits[l] if W_inits[l] else lasagne.init.GlorotUniform()
        b_init = b_inits[l] if b_inits[l] else lasagne.init.Constant(0.)

        nn = lasagne.layers.DenseLayer(nn, num_units=dims[l], W=W_init, b=b_init, nonlinearity=nonlinearities[l])

        if batch_norms[l]:
            nn = lasagne.layers.batch_norm(nn)

    return nn
