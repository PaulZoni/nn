import cv2
import numpy as np


class FaceCropper(object):
    CASCADE_PATH = "/home/pavel/PycharmProjects/nn/facerecognition/evaluate/haarcascade_frontalface_default.xml"
    frontal_face_extended = "/home/pavel/PycharmProjects/nn/facerecognition/evaluate/haarcascade_frontalcatface_extended.xml"

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(self.CASCADE_PATH)

    def generate(self, image_path=None, show_result=None, size=32, inter=cv2.INTER_AREA, frame=None):

        img = None

        if frame is None:
            img = cv2.imread(image_path)
        else:
            img = frame

        if img is None and frame is None:
            print("Can't open image file")
            return 0

        print(len(img))
        faces = self.face_cascade.detectMultiScale(img, 1.1, 3, minSize=(100, 100),)

        if faces is None:
            print('Failed to detect face')
            return 0

        if show_result:
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            image_resize = cv2.resize(img, (960, 540))
            cv2.imshow('img', image_resize)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        facecnt = len(faces)
        print("Detected faces: %d" % facecnt)
        if facecnt is 0:
            return 0
        i = 0
        height, width = img.shape[:2]

        last_images = []
        for (x, y, w, h) in faces:
            r = max(w, h) / 2
            centerx = x + w / 2
            centery = y + h / 2
            nx = int(centerx - r)
            ny = int(centery - r)
            nr = int(r * 2)

            faceimg = img[ny:ny + nr, nx:nx + nr]
            lastimg = cv2.resize(faceimg, (size, size), interpolation=inter)
            i += 1
            last_images.append(lastimg)
        return np.array(last_images)
