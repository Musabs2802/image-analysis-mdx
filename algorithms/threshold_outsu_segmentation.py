import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray

UPLOAD_DIR = "static/uploads/"
GENERATED_DIR = "static/gen/"


class ThresholdOtsuSegmentation:
    def __init__(self, filename) -> None:
        self.filename = filename
        self.__analyze()

    def __analyze(self):
        original_image = cv2.imread(self.filename)
        image_gscale = rgb2gray(original_image)

        threshold_otsu_calculated = threshold_otsu(image_gscale)
        segmented_image = image_gscale > threshold_otsu_calculated

        plt.imsave(GENERATED_DIR + "threshold_otsu.jpg", segmented_image)
