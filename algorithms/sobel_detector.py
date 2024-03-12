import numpy as np
from scipy import ndimage
import cv2
import matplotlib.pyplot as plt

UPLOAD_DIR = "static/uploads/"
GENERATED_DIR = "static/gen/"

class SobelDetector:
    def __init__(self, filename) -> None:
        self.filename = filename
        self.__detect()

    def __detect(self):
        image = cv2.imread(self.filename, cv2.IMREAD_GRAYSCALE)
        image = image / 255

        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        ky = np.rot90(kx)

        Gx = ndimage.convolve(image, kx)
        Gy = ndimage.convolve(image, ky)

        edged_img = np.sqrt(np.square(Gx) + np.square(Gy))
        plt.imsave(GENERATED_DIR + "sobel.jpg", edged_img, cmap="gray")
