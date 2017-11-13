import numpy as np

STUDENT = {'name': 'Tamir Moshiashvili',
           'ID': '316131259'}


def softmax(x):
    """
    Compute the softmax vector.
    x: a n-dim vector (numpy array)
    returns: an n-dim vector (numpy array) of softmax values
    """
    x -= np.max(x)  # For numeric stability, we use the identify we proved in Ex 2 Q1.
    x = np.exp(x)
    x /= np.sum(x)

    return x


# lambda to create list of tuples (W, b) from the given parameters
pairs_of_W_and_b = lambda params: [tuple(params[i: i + 2]) for i in range(0, len(params), 2)]


def classifier_output(x, params):
    # create pairs of parameters (W, b)
    pairs = pairs_of_W_and_b(params)

    # go through all the pairs besides the last one
    for (W, b) in pairs[:-1]:
        x = np.tanh(np.dot(x, W) + b)

    # last layer is the softmax
    W, b = pairs[-1]
    probs = softmax(np.dot(x, W) + b)
    return probs


def predict(x, params):
    return np.argmax(classifier_output(x, params))


def get_layer_probs(x, pairs, layer):
    probs = np.array(x)
    for W, b in pairs[:layer]:
        probs = np.tanh(np.dot(probs, W) + b)
    return probs


def loss_and_gradients(x, y, params):
    # loss
    probs = classifier_output(x, params)
    loss = -np.log(probs[y])

    # calculate the gradients
    grads = list()

    pairs = pairs_of_W_and_b(params)
    num_layers = len(pairs)
    t_out = get_layer_probs(x, pairs, num_layers - 1)

    # gradient for the last layer
    W, b = pairs[-1]
    rest_params = pairs[:-1]
    rest_params.reverse()

    dl_dt = -W[:, y] + np.dot(W, probs)
    dl_db = np.copy(probs)
    dl_db[y] -= 1
    dl_dW = np.zeros(W.shape)
    for (i, j) in np.ndindex(dl_dW.shape):
        dl_dW[i, j] = t_out[i] * probs[j] - (j == y) * t_out[i]

    # add to the grads in reverse way (at the end we will be reverse back)
    grads.append(dl_db)
    grads.append(dl_dW)

    curr_dt = dl_dt
    num_layer = num_layers - 2
    for W, b in rest_params:
        layer_in = get_layer_probs(x, pairs, num_layer)
        tanh_grad = 1 - (np.tanh(np.dot(layer_in, W) + b)) ** 2

        dt_db = tanh_grad
        dt_dW = np.dot(layer_in.reshape(len(layer_in), 1), tanh_grad.reshape(1, len(tanh_grad)))
        grads.append(curr_dt * dt_db)
        grads.append(curr_dt * dt_dW)

        dt_dprevt = tanh_grad * W
        curr_dt = np.dot(dt_dprevt, curr_dt)
        num_layer -= 1

    grads.reverse()
    return loss, grads


def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.
    
    Assume a tanh activation function between all the layers.

    return:
    a list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """
    # Xavier Glorot init
    Glorot_init = lambda n, m: np.random.uniform(-np.sqrt(6.0 / (n + m)), np.sqrt(6.0 / (n + m)),
                                                 (n, m) if (n != 1 and m != 1) else n * m)
    params = []
    for i in range(len(dims) - 1):
        in_dim = dims[i]
        out_dim = dims[i + 1]
        params.append(Glorot_init(in_dim, out_dim))
        params.append(Glorot_init(1, out_dim))
    return params


if __name__ == '__main__':
    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    dims = [5, 4, 7, 3]
    params = create_classifier(dims)


    def _loss_and_p_grad(p):
        """
        General function - return loss and the gradients with respect to parameter p
        """
        params_to_send = np.copy(params)
        par_num = 0
        for i in range(len(params)):
            if p.shape == params[i].shape:
                params_to_send[i] = p
                par_num = i

        loss, grads = loss_and_gradients(range(dims[0]), 0, params_to_send)
        return loss, grads[par_num]


    for _ in xrange(10):
        my_params = create_classifier(dims)
        for p in my_params:
            gradient_check(_loss_and_p_grad, p)
