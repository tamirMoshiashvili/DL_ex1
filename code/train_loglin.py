import loglinear as ll
import random
import numpy as np
import utils

STUDENT = {'name': 'Tamir Moshiashvili',
           'ID': '316131259'}


def feats_to_vec(features):
    return np.array(features)


def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        y_prediction = ll.predict(features, params)
        if y_prediction == label:
            good += 1
        else:
            bad += 1
    return good / (good + bad)


def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in xrange(num_iterations):
        cum_loss = 0.0  # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = features  # numpy vector.
            y = label  # a number.
            loss, grads = ll.loss_and_gradients(x, y, params)
            cum_loss += loss

            # SGD update parameters
            W, b = params
            updated_W = W - learning_rate * grads[0]
            updated_b = b - learning_rate * grads[1]
            params = (updated_W, updated_b)

        # notify progress
        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print I, train_loss, train_accuracy, dev_accuracy
    return params


if __name__ == '__main__':
    model = 'unigrams'

    # training parameters
    num_iterations = 30
    learning_rate = 1e-3

    # get params
    vocab_size, num_langs, train_data, dev_data = utils.get_unigrams_params()

    params = ll.create_classifier(vocab_size, num_langs)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)
