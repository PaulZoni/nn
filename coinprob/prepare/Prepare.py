#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import pandas
import numpy
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer


def value_to_sequences_form(vocab, sequences):
    new_sequences = sequences
    for i in range(0, len(sequences)):
        seq = sequences[i]
        for j in range(0, len(seq)):
            val = seq[j]
            if val in vocab:
                new_sequences[i][j] = vocab.index(val)
    return new_sequences


path = '/home/pavel/Документы/btc.csv'
df = pandas.read_csv(path,)
array_data = df['date']
array_value = df['TxTfrValMeanUSD']

map_data_and_value = dict()
data = list()
value = list()

for i in range(0, len(array_data)):
    if str(array_value[i]) != 'nan':
        data.append(array_data[i])
        value.append(array_value[i])
print len(data)
print len(value)
value = [int(val) for val in value]

vocab = list()
for i in range(0, len(value)):
    if value[i] not in vocab:
        vocab.append(value[i])
print('Total size: %d' % len(vocab))
vocab_size = len(vocab) - 1

length = 30 + 1
value_sequences = list()
for i in range(length, len(value)):
    seq = value[i - length: i]
    value_sequences.append(seq)

value_sequences = value_to_sequences_form(vocab=vocab, sequences=value_sequences)

value_sequences = numpy.array(value_sequences)
X, Y = value_sequences[:, :-1], value_sequences[:, -1]


Y = to_categorical(y=Y, num_classes=vocab_size + 1)

print X[1]
print (Y[1])
