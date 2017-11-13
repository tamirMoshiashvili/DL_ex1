# This file provides code which you may or may not find helpful.
# Use it if you want, or ignore it.
import random


def read_data(fname):
    data = []
    for line in file(fname):
        label, text = line.strip().lower().split("\t", 1)
        data.append((label, text))
    return data


def text_to_bigrams(text):
    return ["%s%s" % (c1, c2) for c1, c2 in zip(text, text[1:])]


TRAIN = [(l, text_to_bigrams(t)) for l, t in read_data("../data/train")]
DEV = [(l, text_to_bigrams(t)) for l, t in read_data("../data/dev")]
TEST = [(l, text_to_bigrams(t)) for l, t in read_data("../data/test")]

UNI_TRAIN = [(l, list(t)) for l, t in read_data("../data/train")]
UNI_DEV = [(l, list(t)) for l, t in read_data("../data/dev")]
UNI_TEST = [(l, list(t)) for l, t in read_data("../data/test")]

from collections import Counter

fc = Counter()
for l, feats in TRAIN:
    fc.update(feats)

# 600 most common bigrams in the training set.
vocab = set([x for x, c in fc.most_common(600)])

uni_fc = Counter()
for l, feats in UNI_TRAIN:
    uni_fc.update(feats)

# 600 most common unigrams in the training set.
uni_vocab = set([x for x, c in uni_fc.most_common(100)])

# label strings to IDs
L2I = {l: i for i, l in enumerate(list(sorted(set([l for l, t in TRAIN]))))}

# feature strings (bigrams) to IDs
F2I = {f: i for i, f in enumerate(list(sorted(vocab)))}
# feature strings (unigrams) to IDs
UNI_F2I = {f: i for i, f in enumerate(list(sorted(uni_vocab)))}

# IDs to label strings
I2L = {i: l for i, l in enumerate(list(sorted(set([l for l, t in TRAIN]))))}


def bigrams_to_frequencies(bigrams):
    """
    bigrams: list of bigrams.

    Create numpy-vector that contain the frequency of each bigram in bigrams.
    """
    import numpy as np

    features = np.zeros(len(F2I))
    for bigram in set(bigrams) & set(F2I.keys()):
        features[F2I[bigram]] = bigrams.count(bigram)

    # normalized
    return 100 * features / float(len(bigrams))


def unigrams_to_frequencies(unigrams):
    """
    unigrams: list of unirams.

    Create numpy-vector that contain the frequency of each unigram in unigrams.
    """
    import numpy as np

    features = np.zeros(len(UNI_F2I))
    for unigram in set(unigrams) & set(UNI_F2I.keys()):
        features[UNI_F2I[unigram]] = unigrams.count(unigram)

    # normalized
    return 100 * features / float(len(unigrams))


def get_unigrams_params():
    # load sets from utils
    train_set = UNI_TRAIN
    dev_set = UNI_DEV
    indexed_langs = L2I
    indexed_vocab = UNI_F2I

    # shapes
    uni_num_langs = len(indexed_langs)
    uni_vocab_size = len(indexed_vocab)

    uni_train_data = list()
    for item in train_set:
        lang, unigrams = indexed_langs[item[0]], unigrams_to_frequencies(item[1])
        uni_train_data.append((lang, unigrams))

    uni_dev_data = list()
    for item in dev_set:
        lang, unigrams = indexed_langs[item[0]], unigrams_to_frequencies(item[1])
        uni_dev_data.append((lang, unigrams))
    return uni_vocab_size, uni_num_langs, uni_train_data, uni_dev_data


def get_bigrams_params():
    # load sets from utils
    train_set = TRAIN
    dev_set = DEV
    indexed_langs = L2I
    indexed_vocab = F2I

    # shapes
    bi_num_langs = len(indexed_langs)
    bi_vocab_size = len(indexed_vocab)

    bi_train_data = list()
    for item in train_set:
        lang, bigrams = indexed_langs[item[0]], bigrams_to_frequencies(item[1])
        bi_train_data.append((lang, bigrams))

    bi_dev_data = list()
    for item in dev_set:
        lang, bigrams = indexed_langs[item[0]], bigrams_to_frequencies(item[1])
        bi_dev_data.append((lang, bigrams))

    return bi_vocab_size, bi_num_langs, bi_train_data, bi_dev_data
