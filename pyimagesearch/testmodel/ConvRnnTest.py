#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ['THEANO_FLAGS'] = "device=cuda0,force_device=True,floatX=float32,dnn.enabled=False,gcc.cxxflags=-Wno-narrowing,gpuarray.preallocate=0.4,"
# gcc.cxxflags=-Wno-narrowing
import theano
'''-------------------------------------------'''

from keras.callbacks import LearningRateScheduler
from keras.datasets import imdb
from keras.preprocessing import sequence
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
import pandas
from keras.optimizers import SGD
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from pyimagesearch.nn.conv.ConvRnn import RnnForWord
from pyimagesearch.callbacks.StepDecay import step_decay

embedding_vector_length = 100

df = pandas.DataFrame()
df = pandas.read_csv('/home/pavel/PycharmProjects/nn/res/imdb_master.csv', encoding='ISO-8859-1')
df.head(3)
X_train = df.loc[8000: 17000, 'review']
y_train = df.loc[8000: 17000, 'label']

values = {}
true_values = {}

for index, value in X_train.items():
    if len(value.split()) < 100:
        values[index] = value
        true_values[index] = y_train[index]

print(len(values))
print(len(true_values))
X_train = [x for x in values.values()]
y_train = [x for x in true_values.values()]

#X_test = df.loc[25000:, 'review']
#y_test = df.loc[25000:, 'label']


tk = Tokenizer()
total_review = X_train + y_train
print('Total review' + str(len(total_review)))
tk.fit_on_texts(total_review)
print('Total size: ' + str(df.size))

length_max = max([len(s.split()) for s in total_review])
vocabulary_size = len(tk.word_index) + 1

for index in range(len(y_train)):
    if y_train[index] == 'neg':
        y_train[index] = 0
    else:
        y_train[index] = 1
print(str(y_train))
#y_train = y_train[: 2000]
#X_train = X_train[: 2000]
print(len(y_train))
print(len(X_train))

X_train_tokens = tk.texts_to_sequences(X_train)
#X_test_tokens = tk.texts_to_sequences(X_test)

X_train_bad = sequence.pad_sequences(X_train_tokens, maxlen=length_max, padding='post')
#X_test_pad = sequence.pad_sequences(X_test_tokens, maxlen=length_max, padding='post')

figPath = os.path.sep.join(['/home/pavel/PycharmProjects/nn/pyimagesearch/plot/', "{}_.png".format(os.getpid())])
jsonPath = os.path.sep.join(['/home/pavel/PycharmProjects/nn/pyimagesearch/plot/', "{}.json".format(os.getpid())])

optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)
callbacks = [TrainingMonitor(jsonPath=jsonPath, figPath=figPath, val=False), LearningRateScheduler(step_decay)]
model = RnnForWord().build(vocab_size=vocabulary_size, max_review_length=length_max, embedding_vector_length=length_max)
print("[INFO] compiling model...")
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(X_train_bad, y_train, epochs=40, batch_size=16, callbacks=callbacks)

#scores = model.evaluate(X_test_pad, y_test, verbose=0)
#print('Accuracy : %.4f ' % scores[1])

index_list = tk.texts_to_sequences(['it was just a terrible movie',
                                    'I liked the film a lot, I would go to the cinema again'
                                    ])
test = sequence.pad_sequences(index_list, maxlen=length_max)

answer = model.predict(x=test)
print('predict: ' + str(answer))
#print('\n' + str(tk.sequences_to_texts(answer)))


plt.plot(history.history['acc'])
#plt.plot(scores[1])
plt.plot(history.history['loss'])
#plt.plot(scores[0])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['acc', 'loss'], loc='upper left')
plt.show()
plt.savefig('/home/pavel/PycharmProjects/nn/pyimagesearch/plot/RnnForWord_plot.png')