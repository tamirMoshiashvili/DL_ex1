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


def classifier_output(x, params):
    U, W, b, b_tag = params

    tan_h = np.tanh(np.dot(x, W) + b)
    probs = softmax(np.dot(U, tan_h) + b_tag)

    return probs


def predict(x, params):
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
        Compute the loss and the gradients at point x with given parameters.
        y is a scalar indicating the correct label.

        returns:
            loss,[gU, gW, gb, gb_tag]

        loss: scalar
        gU: matrix, gradients of U
        gW: matrix, gradients of W
        gb: vector, gradients of b
        gb_tag: vector, gradients of b_tag
    """
    U, W, b, b_tag = params

    y_hat = classifier_output(x, params)
    loss = -np.log(y_hat[y])

    tan_h = np.tanh(np.dot(x, W) + b)
    probs = softmax(np.dot(U, tan_h) + b_tag)
    layer1_der = 1 - tan_h ** 2

    # gradient of b_tag
    gb_tag = np.copy(probs)
    gb_tag[y] -= 1

    # gradient of U
    gU = np.zeros(U.shape)
    for (i, j) in np.ndindex(gU.shape):
        gU[i, j] = tan_h[i] * probs[j] - (j == y) * tan_h[i]

    # gradient of b - use the chain rule
    dloss_dtanh = -U[:, y] + np.dot(U, probs)
    dtanh_db = layer1_der
    gb = dloss_dtanh * dtanh_db

    # gradient of W - use the chain rule
    gW = np.zeros(W.shape)
    for (i, j) in np.ndindex(gW.shape):
        gW[i, j] = layer1_der[j] * x[i] * dloss_dtanh[j]

    return loss, [gU, gW, gb, gb_tag]


def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.
    """
    # Xavier Glorot init
    Glorot_init = lambda n, m: np.random.uniform(-np.sqrt(6.0 / (n + m)), np.sqrt(6.0 / (n + m)), (n, m) \
        if (n != 1 and m != 1) else n * m)

    U = Glorot_init(hid_dim, out_dim)
    W = Glorot_init(in_dim, hid_dim)
    b = Glorot_init(1, hid_dim)
    b_tag = Glorot_init(1, out_dim)

    params = [U, W, b, b_tag]
    return params


if __name__ == '__main__':
    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    U, W, b, b_tag = create_classifier(3, 6, 5)


    def _loss_and_U_grad(U):
        loss, grads = loss_and_gradients([1, 2, 3], 0, [U, W, b, b_tag])
        return loss, grads[0]


    def _loss_and_W_grad(W):
        global b
        loss, grads = loss_and_gradients([1, 2, 3], 0, [U, W, b, b_tag])
        return loss, grads[1]


    def _loss_and_b_grad(b):
        global W
        loss, grads = loss_and_gradients([1, 2, 3], 0, [U, W, b, b_tag])
        return loss, grads[2]


    def _loss_and_btag_grad(b_tag):
        loss, grads = loss_and_gradients([1, 2, 3], 0, [U, W, b, b_tag])
        return loss, grads[3]


    for _ in xrange(10):
        W = np.random.randn(W.shape[0], W.shape[1])
        b = np.random.randn(b.shape[0])
        U = np.random.randn(U.shape[0], U.shape[1])
        b_tag = np.random.randn(b_tag.shape[0])
        loss, grads = loss_and_gradients([1, 2, 3], 0, [U, W, b, b_tag])

        gradient_check(_loss_and_U_grad, U)
        gradient_check(_loss_and_W_grad, W)
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_btag_grad, b_tag)
