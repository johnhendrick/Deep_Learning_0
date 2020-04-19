import numpy as np
from keras.preprocessing.sequence import pad_sequences
import random
import pandas as pd

first_letters = 'ABCDEF'
second_numbers = '120'
last_letters = 'QWOPZXML'

# returns a string of the following format: [4 letters A-F][1 digit 0-2][3 letters QWOPZXML]


def get_random_string():
    str1 = ''.join(random.choice(first_letters) for i in range(4))
    str2 = random.choice(second_numbers)
    str3 = ''.join(random.choice(last_letters) for i in range(3))
    return str1+str2+str3


# get 25,000 sequences of this format
random_sequences = [get_random_string() for i in range(25000)]
# this will return string according to the following format
# ['CBCA2QOM', 'FBEF0WZW', 'DBFB2ZML', 'BFCB2WXO']
# add some anomalies to our list
random_sequences.extend(
    ['XYDC2DCA', 'TXSX1ABC', 'RNIU4XRE', 'AABDXUEI', 'SDRAC5RF'])
# save this to a dataframe
seqs_ds = pd.DataFrame(random_sequences)


# Build the char index that we will use to encode seqs to numbers
# (this char index was written by Jason Brownlee from Machine Learning Mastery)
char_index = '0abcdefghijklmnopqrstuvwxyz'
char_index += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
char_index += '123456789'
char_index += '().,-/+=&$?@#!*:;_[]|%⸏{}\"\'' + ' ' + '\\'

char_to_int = dict((c, i) for i, c in enumerate(char_index))
int_to_char = dict((i, c) for i, c in enumerate(char_index))

# function that convert a char seqs to numbers seqs
# (it does a little more but lets leave it for now)


def encode_sequence_list(seqs, feat_n=0):
    encoded_seqs = []
    for seq in seqs:
        encoded_seq = [char_to_int[c] for c in seq]
        encoded_seqs.append(encoded_seq)
    if(feat_n > 0):
        encoded_seqs.append(np.zeros(feat_n))
    return pad_sequences(encoded_seqs, padding='post')


def decode_sequence_list(seqs):
    decoded_seqs = []
    for seq in seqs:
        decoded_seq = [int_to_char[i] for i in seq]
        decoded_seqs.append(decoded_seq)
    return decoded_seqs

# Using the char_index, the encode_sequence_list function
# will turn a string like this EBCA0OXO
# to an array like this [29 32 27 27  0 42 42 38]


# encode each string seq to an integer array [[1],[5],[67]], [[45],[76],[7]
encoded_seqs = encode_sequence_list(random_sequences)
# mix everything up
np.random.shuffle(encoded_seqs)
