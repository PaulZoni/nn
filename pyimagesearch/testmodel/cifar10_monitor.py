#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ['THEANO_FLAGS'] = "device=cuda0,force_device=True,floatX=float32,dnn.enabled=False,gcc.cxxflags=-Wno-narrowing,gpuarray.preallocate=0.4"
# gcc.cxxflags=-Wno-narrowing
import theano
'''-------------------------------------------'''

import matplotlib
matplotlib.use("Agg")
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import LearningRateScheduler
from pyimagesearch.nn.conv.minivggnet import MiniVGGNet
from pyimagesearch.callbacks.StepDecay import step_decay
from keras.optimizers import SGD
from keras.datasets import cifar10
import argparse
import os


ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=False, help="path to the output directory",
                default='/home/pavel/PycharmProjects/nn/pyimagesearch/plot/')
args = vars(ap.parse_args())
print("[INFO process ID: {}".format(os.getpid()))

print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

print("[INFO] compiling model...")
opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

figPath = os.path.sep.join([args["output"], "{}_.png".format(os.getpid())])
jsonPath = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])

callbacks = [TrainingMonitor(jsonPath=jsonPath, figPath=figPath), LearningRateScheduler(step_decay)]

model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=40, callbacks=callbacks, verbose=1)
