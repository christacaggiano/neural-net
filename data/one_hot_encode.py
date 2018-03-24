import numpy as np
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from Bio import SeqIO
import textwrap
import progressbar
import pickle
import math
import os.path
import random

num_positives = 137
num_negatives = 3164 * (math.floor(1000 / 17))
sequence_size = 17
number_base_pairs = 4


def one_hot_encode(string):
    d = {"A": 0, "T": 1, "C": 2, "G": 3}
    onehot_encoder = OneHotEncoder(sparse=True, n_values=4)
    integer_encoded = np.array([d[x] for x in string])
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    return onehot_encoder.fit_transform(integer_encoded).toarray()

def generate_labels(size, label_type):
    if label_type:
        return np.full((size, 1), 0.9, dtype=float)
    else:
        return np.full((size, 1), 0, dtype=float)


def subsample(number_of_samples, array):
    x, y, z = array.shape
    subsample_array = np.zeros((number_of_samples, y, z))

    for i in range(number_of_samples):
        index = random.randint(0, x)
        subsample_array[i] = array[index]

    return subsample_array

def load_data():

    global num_negatives
    global num_positives
    global number_base_pairs
    global sequence_size

    if os.path.isfile("positives_array.npy"):
        positives_encoded = np.load("positives_array.npy")
    else:
        positives_encoded = np.zeros((num_positives, sequence_size, number_base_pairs))

        with open("data/rap1-lieb-positives.txt") as pos:
            i = 0
            for line in pos:
                positives_encoded[i] = one_hot_encode(line.strip("\n"))
                i += 1
        np.save("data/positives_array", positives_encoded)

    if os.path.isfile("data/negatives_array.npy"):
        negatives_encoded = np.load("data/negatives_array.npy")

    else:
        negatives_encoded = np.zeros((num_negatives, sequence_size, number_base_pairs))
        fasta_sequences = SeqIO.parse(open("data/yeast-upstream-1k-negative.fa"), 'fasta')
        i = 0
        with progressbar.ProgressBar(max_value=num_negatives) as bar:
            for fasta in fasta_sequences:
                name, sequence = fasta.id, str(fasta.seq)
                # cut off last ones
                for subgroup in textwrap.wrap(sequence, sequence_size)[:-1]:
                    negatives_encoded[i] = one_hot_encode(subgroup)
                    bar.update(i)
                    i += 1
        np.save("data/negatives_array", negatives_encoded)

    return negatives_encoded, positives_encoded


def return_data(num_subsample):

    global num_negatives
    global num_positives

    negatives_encoded, positives_encoded = load_data()
    neg_subsample = subsample(num_subsample, negatives_encoded)

    pos_labels = generate_labels(num_positives, True)
    neg_labels = generate_labels(num_subsample, False)

    input = np.concatenate((neg_subsample, positives_encoded), axis=0)
    output = np.concatenate((neg_labels, pos_labels), axis=0)
    return input.reshape((input.shape[0], 68)), output

