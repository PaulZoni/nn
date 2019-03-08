import numpy as np

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from sklearn import datasets
import matplotlib.pyplot as plt
import argparse
from pyimagesearch import load_MNIST, load_target_MNIST


path = '/home/pavel/PycharmProjects/nn/pyimagesearch/mnist-original.mat'

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=False, help="path to the output loss/accuracy plot",
                default="/home/pavel/Документы")
args = vars(ap.parse_args())

print("[INFO] loading MNIST (full) dataset...")
dataset = load_MNIST(path)

#data = dataset.data.astype("float") / 255.0
data = dataset.astype('float') / 255.0

(trainX, testX, trainY, testY) = train_test_split(data, load_target_MNIST(path), test_size=0.25)


lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)


model = Sequential()
model.add(Dense(256, input_shape=(784,), activation="sigmoid"))
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(10, activation="softmax"))

print("[INFO] training network...")
sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=128)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
                            target_names=[str(x) for x in lb.classes_]))


plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])