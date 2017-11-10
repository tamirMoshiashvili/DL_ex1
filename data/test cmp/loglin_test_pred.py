import code.loglinear as ll
import random
import numpy as np
import code.utils


def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        y_prediction = ll.predict(features, params)
        if y_prediction == label:
            good += 1
        else:
            bad += 1
    return good / (good + bad)


def train_classifier(train_data, num_iterations, learning_rate, params):
    """
        Create and train a classifier, and return the parameters.

        train_data: a list of (label, feature) pairs.
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
        print I, train_loss, train_accuracy
    return params


def bigrams2frequencies(bigrams):
    """
    bigrams: list of bigrams.

    Create numpy-vector that contain the frequency of each bigram in bigrams.
    """
    indexed_vocab = code.utils.F2I
    features = np.zeros(len(indexed_vocab))
    for bigram in set(bigrams) & set(indexed_vocab.keys()):
        features[indexed_vocab[bigram]] = bigrams.count(bigram)

    # normalized
    return 100 * features / float(len(bigrams))


def train():
    """
    train the log linear model with the train and dev sets
    :return: trained parameters
    """

    # load sets from utils
    train_set = code.utils.TRAIN
    dev_set = code.utils.DEV
    indexed_langs = code.utils.L2I
    indexed_vocab = code.utils.F2I

    # shapes
    num_langs = len(indexed_langs)
    vocab_size = len(indexed_vocab)

    # training parameters
    num_iterations = 15
    learning_rate = 0.2

    train_data = list()
    for item in train_set:
        lang, bigrams = indexed_langs[item[0]], bigrams2frequencies(item[1])
        train_data.append((lang, bigrams))

    dev_data = list()
    for item in dev_set:
        lang, bigrams = indexed_langs[item[0]], bigrams2frequencies(item[1])
        dev_data.append((lang, bigrams))

    # get the trained parameters
    params = ll.create_classifier(vocab_size, num_langs)
    trained_params = train_classifier(train_data, num_iterations, learning_rate, params)
    trained_params = train_classifier(dev_data, num_iterations, learning_rate, trained_params)
    return trained_params


def predict_test_with_loglin():
    """
    create list of predictions on test set
    :return: list of predictions
    """
    pred = list()

    # train the model
    test_set = code.utils.TEST
    test_data = list()
    for item in test_set:
        bigrams = bigrams2frequencies(item[1])
        test_data.append(bigrams)

    # make predictions
    I2L = code.utils.I2L
    params = train()
    for x in test_data:
        pred.append(I2L[ll.predict(x, params)])

    return pred

if __name__ == '__main__':
    """
        predict labels for test set with log linear model
    """
    pred = predict_test_with_loglin()  # get predictions

    # write predictions to the file
    my_file = open('test.pred', 'w')
    for item in pred:
        my_file.write(item + '\n')

    my_file.close()
