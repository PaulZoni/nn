#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ['THEANO_FLAGS'] = "device=cuda0,force_device=True,floatX=float32,dnn.enabled=False,gcc.cxxflags=-Wno-narrowing,gpuarray.preallocate=0.4"
# gcc.cxxflags=-Wno-narrowing
import theano
'''-------------------------------------------'''
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv.minivggnet import MiniVGGNet
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os


path_to_model = '/home/pavel/PycharmProjects/nn/pyimagesearch/model'
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=False, help="path to weights directory", default=path_to_model)
args = vars(ap.parse_args())

print('[INFO] Loading   CIFAR-10 data...')
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype('float') / 255.0
testX = testX.astype('float') / 255.0
labelBinaries = LabelBinarizer()
trainY = labelBinaries.fit_transform(trainY)
testY = labelBinaries.fit_transform(testY)

print("[INFO] compiling model...")
opt = SGD(lr=0.01, decay=0.01 / 30, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

frame = os.path.sep.join([args['weights'], 'weights-{epoch:03d}-{val_loss:.4f}.hdf5'])
checkpoint = ModelCheckpoint(filepath=frame, monitor='val_loss', mode='min', save_best_only=True, verbose=1)
figPath = os.path.sep.join(['/home/pavel/PycharmProjects/nn/pyimagesearch/plot/', "{}_.png".format(os.getpid())])
jsonPath = os.path.sep.join(['/home/pavel/PycharmProjects/nn/pyimagesearch/plot/', "{}.json".format(os.getpid())])
callbacks = [checkpoint, TrainingMonitor(jsonPath=jsonPath, figPath=figPath, val=False)]
print("[INFO] training network...")
H = model.fit(x=trainX, y=trainY, validation_data=(testX, testY), batch_size=64, epochs=30, callbacks=callbacks, verbose=1)
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 40), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 40), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on CIFAR-10")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
