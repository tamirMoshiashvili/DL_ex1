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
        y_prdeiction = ll.predict(features, params)
        if y_prdeiction == label:
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

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print I, train_loss, train_accuracy, dev_accuracy
    return params


def bigrams2frequencies(bigrams):
    indexed_vocab = utils.F2I
    features = np.zeros(len(indexed_vocab))
    for bigram in bigrams & indexed_vocab.keys():
        features[indexed_vocab[bigram]] = bigrams.count(bigram)

    # normalized
    return features / len(bigrams)


if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.

    # ...
    train_set = utils.TRAIN
    dev_set = utils.DEV
    indexed_langs = utils.L2I
    indexed_vocab = utils.F2I

    num_langs = len(indexed_langs)
    vocab_size = len(indexed_vocab)
    num_iterations = 50
    learning_rate = 5e-4

    train_data = list()
    for item in train_set:
        lang, bigrams = indexed_langs[item[0]], bigrams2frequencies(item[1])
        train_data.append((lang, bigrams))

    dev_data = list()
    for item in dev_set:
        lang, bigrams = indexed_langs[item[0]], bigrams2frequencies(item[1])
        dev_data.append((lang, bigrams))

    params = ll.create_classifier(vocab_size, num_langs)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)
