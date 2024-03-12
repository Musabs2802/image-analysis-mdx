import numpy as np
import cv2
import matplotlib.pyplot as plt

UPLOAD_DIR = "static/uploads/"
GENERATED_DIR = "static/gen/"


class FreiChenDetector:
    def __init__(self, filename) -> None:
        self.filename = filename
        self.__detect()

    def __detect(self):
        image = cv2.imread(self.filename, cv2.IMREAD_GRAYSCALE)

        masks = [
            np.array(
                [[1, np.sqrt(2), 1], [0, 0, 0], [-1, -np.sqrt(2), -1]]
            ),
            np.array(
                [[1, 0, -1], [np.sqrt(2), 0, -np.sqrt(2)], [1, 0, -1]]
            ),
            np.array(
                [[np.sqrt(2), 1, 0], [1, 0, -1], [0, -1, -np.sqrt(2)]]
            ),
            np.array(
                [[0, 1, np.sqrt(2)], [-1, 0, 1], [-np.sqrt(2), -1, 0]]
            ),
        ]

        edged_img = np.zeros_like(image)

        for mask in masks:
            edged_img += cv2.filter2D(image, -1, mask)
        
        edged_img = cv2.normalize(edged_img, None, 0, 255, cv2.NORM_MINMAX)

        plt.imsave(GENERATED_DIR + "frei_chen.jpg", edged_img, cmap="gray")
