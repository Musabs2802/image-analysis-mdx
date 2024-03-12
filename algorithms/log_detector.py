import numpy as np
from scipy import ndimage
import cv2
import matplotlib.pyplot as plt

UPLOAD_DIR = "static/uploads/"
GENERATED_DIR = "static/gen/"

class LOGDetector:
    def __init__(self, filename) -> None:
        self.filename = filename
        self.sigma = 2
        self.size = None
        self.__detect()

    def __detect(self):
        image = cv2.imread(self.filename, cv2.IMREAD_GRAYSCALE)
        image = image / 255

        if self.size is None:
            self.size = int(6 * self.sigma + 1) if self.sigma >= 1 else 7

        if self.size % 2 == 0:
            self.size += 1

        x, y = np.meshgrid(np.arange(-self.size//2+1, self.size//2+1), np.arange(-self.size//2+1, self.size//2+1))
        kernel = -(1/(np.pi * self.sigma**4)) * (1 - ((x**2 + y**2) / (2 * self.sigma**2))) * np.exp(-(x**2 + y**2) / (2 * self.sigma**2))
        kernel = kernel / np.sum(np.abs(kernel))

        edged_img = ndimage.convolve(image, kernel)
        plt.imsave(GENERATED_DIR + "laplacian.jpg", edged_img, cmap="gray")