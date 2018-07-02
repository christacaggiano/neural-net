import numpy as np
from sklearn.preprocessing import OneHotEncoder
from Bio import SeqIO
import textwrap
import progressbar
import math
import os.path
import random


# global variables defining the number of training data
num_positives = 137
num_negatives = 3164 * (math.floor(1000 / 17))
sequence_size = 17
number_base_pairs = 4
num_test = 3195


def one_hot_encode(string):
    """
    takes in a string and returns its one hot encoding
    :param string:
    :return: array of one hot encoding. for this task 17, 4 np array
    """
    d = {"A": 0, "T": 1, "C": 2, "G": 3}  # arbitrarily assign the nt values
    onehot_encoder = OneHotEncoder(sparse=True, n_values=4)  # make encoding object from scikit learn
    integer_encoded = np.array([d[x] for x in string])  # integer encode the string given the above dictionary
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)  # make a shape to fit with one hot encoding obj
    return onehot_encoder.fit_transform(integer_encoded).toarray()  # return an array


def generate_labels(size, label_type):
    """
    simple function that assigns a label whether the input is determined a 'true' binding site
    values are arbitrarily set to 0 and 1
    :param size: number of labels
    :param label_type: Boolean True==positive training
    :return: positive numpy array
    """
    if label_type:
        return np.full((size, 1), 1, dtype=float)  # return positive numpy array
    else:
        return np.full((size, 1), 0, dtype=float)  # return negative numpy array


def subsample(number_of_samples, array):
    """
    take a random subsample given an input number of samples and an array
    :param number_of_samples:
    :param array:
    :return: subsetted array
    """
    x, y, z = array.shape
    subsample_array = np.zeros((number_of_samples, y, z))  # initializes new subset array

    for i in range(number_of_samples):  # for the number of samples desired
        index = random.randint(0, x)  # find a random value
        subsample_array[i] = array[index]  # add the array at the position to our new subset

    return subsample_array

def load_data():

    """
    reads in data and generates it only once, since parsing the negative data is time consuming
    :return:
    """

    # global variables set above
    global num_negatives
    global num_positives
    global number_base_pairs
    global sequence_size
    global num_test

    # if the path exists, load preexisting data
    if os.path.isfile("positives_array.npy"):
        positives_encoded = np.load("positives_array.npy")
    else:
        positives_encoded = np.zeros((num_positives, sequence_size, number_base_pairs))

        # otherwise open the positive data set and one hot encode each 17 bp string
        with open("data/rap1-lieb-positives.txt") as pos:
            i = 0
            for line in pos:
                positives_encoded[i] = one_hot_encode(line.strip("\n"))
                i += 1
        np.save("data/positives_array", positives_encoded)  # save for future use

    # if the negative data exists, read it in otherwise
    if os.path.isfile("data/negatives_array.npy"):
        negatives_encoded = np.load("data/negatives_array.npy")

    # using SeqIO, parse the fasta files
    else:
        negatives_encoded = np.zeros((num_negatives, sequence_size, number_base_pairs))
        fasta_sequences = SeqIO.parse(open("data/yeast-upstream-1k-negative.fa"), 'fasta')
        i = 0
        with progressbar.ProgressBar(max_value=num_negatives) as bar:
            for fasta in fasta_sequences:
                name, sequence = fasta.id, str(fasta.seq)  # get the sequence and id
                # cut off last ones
                for subgroup in textwrap.wrap(sequence, sequence_size)[:-1]:  # find all 17-mers
                    negatives_encoded[i] = one_hot_encode(subgroup)  # one hot encode
                    bar.update(i)
                    i += 1
        np.save("data/negatives_array", negatives_encoded)  # save encodings

    # load and encode test array
    if os.path.isfile("test_array.npy"):
        test_array = np.load("data/test_array.np")
    else:
        test_array = np.zeros((num_test, sequence_size, number_base_pairs))

        with open("data/rap1-lieb-test.txt") as test:
            i = 0
            for line in test:
                test_array[i] = one_hot_encode(line.strip("\n"))
                i += 1
        np.save("data/test_array.np", test_array)  # save for future use

    return negatives_encoded, positives_encoded, test_array  # return the positive and negative data to interact with NN, test data for prediction


def return_data(num_subsample):
    """
    returns data for interaction with NN
    :param num_subsample:
    :return:
    """

    global num_negatives
    global num_positives

    negatives_encoded, positives_encoded, test_array = load_data()  # loads data from above
    neg_subsample = subsample(num_subsample, negatives_encoded)  # subsamples negative data

    pos_labels = generate_labels(num_positives, True)  # generates labels
    neg_labels = generate_labels(num_subsample, False)

    input = np.concatenate((neg_subsample, positives_encoded), axis=0)  # puts negative and positive data into one array
    output = np.concatenate((neg_labels, pos_labels), axis=0)

    return input.reshape((input.shape[0], 68)), output, test_array.reshape((test_array.shape[0], 68))  # returns flattened input data and output data for use by NN


def print_predictions(predictions):
    """
    print predictions to a file for assignment
    :param predictions: predictions based on test data
    :return:
    """
    with open("data/rap1-lieb-test.txt", "r") as test, open("predictions.txt", "w") as pred:
        i = 0
        for line in test:
            output = line.strip("\n") + "    " + ' '.join(map(str, predictions[i])) + "\n"
            pred.writelines(output)
            i += 1
