from facerecognition.evaluate.FaceCropper import FaceCropper
from pyimagesearch.datasets.simpledatasetloader import SimpleDatasetLoader
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from keras.models import load_model
import argparse
import cv2
import numpy as np
from imutils import paths


path_face = '/home/pavel/Документы/datasets/test/IMG_20190320_215904.jpg'
path_i = "/home/pavel/Документы/datasets/test/IMG_20190320_213945.jpg"
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=False, help="path to input dataset", default='/home/pavel/Документы/datasets/test')
ap.add_argument("-m", "--model", required=False, help="path to pre-trained model", default="/home/pavel/PycharmProjects/nn/facerecognition/savemodel/face_weights.hdf5")
args = vars(ap.parse_args())

classLabels = ['much_i', 'persons']
print("[INFO] sampling images...")

imagePaths = np.array(list(paths.list_images(args["dataset"])))
idxs = np.random.randint(0, len(imagePaths), size=(10,))
imagePaths = imagePaths[idxs]

sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths)
data = data.astype("float") / 255.0

print("[INFO] loading pre-trained network...")
model = load_model(args["model"])

preds = model.predict(data, batch_size=32).argmax(axis=1)

for (i, imagePath) in enumerate(imagePaths):

    image = cv2.imread(imagePath)
    cv2.putText(image, "Label: {}".format(classLabels[preds[i]]), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    image_resize = cv2.resize(image, (1200, 1000))
    cv2.imshow("Image", image_resize)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

