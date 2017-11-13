import mlp1
import random
import utils

STUDENT = {'name': 'Tamir Moshiashvili',
           'ID': '316131259'}


def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        y_prediction = mlp1.predict(features, params)
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
            loss, grads = mlp1.loss_and_gradients(x, y, params)
            cum_loss += loss

            # SGD update parameters
            U, W, b, b_tag = params
            updated_U = U - learning_rate * grads[0]
            updated_W = W - learning_rate * grads[1]
            updated_b = b - learning_rate * grads[2]
            updated_btag = b_tag - learning_rate * grads[3]
            params = (updated_U, updated_W, updated_b, updated_btag)

        # notify progress
        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print I, train_loss, train_accuracy, dev_accuracy
    return params


if __name__ == '__main__':
    # load sets from utils
    train_set = utils.TRAIN
    dev_set = utils.DEV
    indexed_langs = utils.L2I
    indexed_vocab = utils.F2I

    # shapes
    num_langs = len(indexed_langs)
    vocab_size = len(indexed_vocab)
    hid_dim = 4

    # training parameters
    num_iterations = 15
    learning_rate = 0.05

    train_data = list()
    for item in train_set:
        lang, bigrams = indexed_langs[item[0]], utils.bigrams_to_frequencies(item[1])
        train_data.append((lang, bigrams))

    dev_data = list()
    for item in dev_set:
        lang, bigrams = indexed_langs[item[0]], utils.bigrams_to_frequencies(item[1])
        dev_data.append((lang, bigrams))

    params = mlp1.create_classifier(vocab_size, hid_dim, num_langs)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)
