import numpy as np
import cv2
import matplotlib.pyplot as plt

UPLOAD_DIR = "static/uploads/"
GENERATED_DIR = "static/gen/"


class KMeansSegmentation:
    def __init__(self, filename, k) -> None:
        self.filename = filename
        self.k = k
        self.__analyze()

    def __analyze(self):
        source = cv2.imread(self.filename)
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

        source_2d = source.reshape((-1, 3))
        source_2d = np.float32(source_2d)

        _, labels, centers = cv2.kmeans(
            source_2d,
            self.k,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85),
            None,
            10,
            cv2.KMEANS_RANDOM_CENTERS,
        )

        centers = np.uint8(centers)
        
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape((source.shape))

        plt.imsave(GENERATED_DIR + "kmeans.jpg", segmented_image, cmap="gray")
