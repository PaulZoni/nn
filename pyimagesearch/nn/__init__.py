from tkinter import *
import tkinter as tk
import numpy as np
import cv2
from PIL import Image, ImageDraw
import PIL
from pyimagesearch import load_MNIST, load_target_MNIST
from pyimagesearch.nn.neuralnetwork import NeuralNetwork
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


canvas_width = 250
canvas_height = 250
center = canvas_height // 2
white = (225, 225, 225)
green = (0, 128, 0)


def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    cv.create_oval(x1, y1, x2, y2, fill='White', width=30,)
    draw.line([x1, y1, x2, y2], fill='White', width=30)


def save_image():
    filename = "image.png"
    image1.save(filename)
    image = cv2.imread(filename, 0)
    p = SimplePreprocessor(28, 28)
    data = p.preprocess(image)
    #train_network(np.concatenate(data), data)
    test_network(np.concatenate(data),)


def train_network():
    print("[INFO] loading MNIST (sample) dataset...")
    path = '/home/pavel/PycharmProjects/nn/pyimagesearch/mnist-original.mat'
    dataset = load_MNIST(path)
    data = dataset.astype('float') / 255.0
    (trainX, testX, trainY, testY) = train_test_split(data, load_target_MNIST(path), test_size=0.25)

    print("[INFO] samples: {}, dim: {}".format(data.shape[0], data.shape[1]))

    trainY = LabelBinarizer().fit_transform(trainY)
    testY = LabelBinarizer().fit_transform(testY)

    print("[INFO] training network...")
    nn = NeuralNetwork([784, 256, 128, 10])
    print("[INFO] {}".format(nn))
    nn.fit(trainX, trainY, epochs=100, displayUpdate=1)

    print("[INFO] evaluating network...")
    predictions = nn.predict(testX)
    predictions = predictions.argmax(axis=1)

    print(classification_report(testY.argmax(axis=1), predictions))


def test_network(dataN,):
    dataN = dataN.astype("float")
    dataN = (dataN - dataN.min()) / (dataN.max() - dataN.min())
    nn = NeuralNetwork([784, 256, 128, 10])
    print("[INFO] evaluating network...")
    predictions = nn.predict_with_save(dataN)
    predictions = predictions.argmax(axis=1)
    print(predictions)


root = Tk()
cv = Canvas(root, width=canvas_width, height=canvas_height, bg='White')
cv.pack()
image1 = PIL.Image.new("RGB", (canvas_width, canvas_height), white)
draw = ImageDraw.Draw(image1)

cv.pack(expand=YES, fill=BOTH)
cv.bind("<B1-Motion>", paint)

button = Button(text="save", command=save_image)
button.pack()
root.mainloop()











