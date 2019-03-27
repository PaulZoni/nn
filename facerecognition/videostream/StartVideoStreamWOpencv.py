import cv2
import numpy as np
import imutils
import argparse
from keras.models import load_model
from facerecognition.evaluate.FaceCropper import FaceCropper

person = '/home/pavel/Документы/VID_20190320_215159.mp4'
i = "/home/pavel/Документы/VID_20190317_115037.mp4"

cap = cv2.VideoCapture(i)

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=False, help="path to pre-trained model", default="/home/pavel/PycharmProjects/nn/facerecognition/savemodel/face_weights.hdf5")
args = vars(ap.parse_args())

classLabels = ['much_i', 'persons']
print("[INFO] sampling images...")

model = load_model(args["model"])
currentFrame = 0
data = None

while(True):
    ret, frame = cap.read()

    detecter = FaceCropper()
    data = detecter.generate(frame=frame, show_result=True)
    #data = cv2.resize(frame, (32, 32), interpolation=cv2.INTER_AREA)

    print("[INFO] loading pre-trained network...")
    print(data.shape)
    if data is not 0:
        data = data.astype("float") / 255.0
        preds = model.predict(data, batch_size=32).argmax(axis=1)
        print(preds)
        cv2.putText(frame, "Label: {}".format(classLabels[preds[0]]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    frame = cv2.resize(frame, (960, 540))
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # To stop duplicate images
    currentFrame += 1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
