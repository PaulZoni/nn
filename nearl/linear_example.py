import numpy as np
import cv2
from matplotlib import pyplot as plt

labels = ["dog", "cat", "panda"]
np.random.seed(1)

W = np.random.randn(3, 3072)
b = np.random.randn(3)

orig = cv2.imread("/home/pavel/Документы/datasets/animals/dog/17dog_img.jpg")
image = cv2.resize(orig, (32, 32)).flatten()

scores = W.dot(image) + b

for (label, score) in zip(labels, scores):
    print("[INFO] {}: {:.2f}".format(label, score))

cv2.putText(orig, "Label: {}".format(labels[int(np.argmax(scores))]),
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


plt.imshow(orig)
plt.show()


#cv2.imshow("Image", orig)
#cv2.waitKey(0)
#cv2.destroyAllWindows()