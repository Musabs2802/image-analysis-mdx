import numpy as np
from scipy import ndimage
import cv2
import matplotlib.pyplot as plt

UPLOAD_DIR = "static/uploads/"
GENERATED_DIR = "static/gen/"


class CannyDetector:
    def __init__(self, filename) -> None:
        self.filename = filename
        self.__detect()

    def __apply_gaussian_blur(self, image, kernel_size=(5, 5), sigma=1.4):
        # Calculate kernel weights based on Gaussian distribution
        kernel_y, kernel_x = kernel_size
        kernel = np.zeros((kernel_y, kernel_x), dtype=np.float64)
        center_x = (kernel_x - 1) / 2
        center_y = (kernel_y - 1) / 2
        for i in range(kernel_y):
            for j in range(kernel_x):
                x = i - center_x
                y = j - center_y
                dist = np.sqrt(x**2 + y**2)
                kernel[i, j] = np.exp(-(dist**2) / (2.0 * sigma**2))

        # Normalize the kernel weights to sum to 1
        kernel /= np.sum(kernel)

        # Perform convolution using the custom kernel
        blurred_image = cv2.filter2D(image, -1, kernel)

        return blurred_image

    def __apply_sobel_filter(self, image):
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        ky = np.rot90(kx)

        fimg = np.float32(image)
        Gx = ndimage.convolve(fimg, kx)
        Gy = ndimage.convolve(fimg, ky)

        mag = np.sqrt(np.square(Gx) + np.square(Gy))
        ang = np.rad2deg(np.arctan2(Gy, Gx))

        return mag, ang

    def __apply_non_maximum_suppression(
        self, image, mag, ang, weak_threshold=None, strong_threshold=None
    ):
        mag_max = np.max(mag)
        if not weak_threshold:
            weak_threshold = mag_max * 0.1
        if not strong_threshold:
            strong_threshold = mag_max * 0.5

        height, width = image.shape

        for i_x in range(width):
            for i_y in range(height):

                grad_ang = ang[i_y, i_x]
                grad_ang = abs(grad_ang - 180) if abs(grad_ang) > 180 else abs(grad_ang)

                # selecting the neighbours of the target pixel
                # according to the gradient direction
                # In the x axis direction
                if grad_ang <= 22.5:
                    neigh_1_x, neigh_1_y = i_x - 1, i_y
                    neigh_2_x, neigh_2_y = i_x + 1, i_y

                # top right (diagonal-1) direction
                elif grad_ang > 22.5 and grad_ang <= (22.5 + 45):
                    neigh_1_x, neigh_1_y = i_x - 1, i_y - 1
                    neigh_2_x, neigh_2_y = i_x + 1, i_y + 1

                # In y-axis direction
                elif grad_ang > (22.5 + 45) and grad_ang <= (22.5 + 90):
                    neigh_1_x, neigh_1_y = i_x, i_y - 1
                    neigh_2_x, neigh_2_y = i_x, i_y + 1

                # top left (diagonal-2) direction
                elif grad_ang > (22.5 + 90) and grad_ang <= (22.5 + 135):
                    neigh_1_x, neigh_1_y = i_x - 1, i_y + 1
                    neigh_2_x, neigh_2_y = i_x + 1, i_y - 1

                # Now it restarts the cycle
                elif grad_ang > (22.5 + 135) and grad_ang <= (22.5 + 180):
                    neigh_1_x, neigh_1_y = i_x - 1, i_y
                    neigh_2_x, neigh_2_y = i_x + 1, i_y

                # Non-maximum suppression step
                if width > neigh_1_x >= 0 and height > neigh_1_y >= 0:
                    if mag[i_y, i_x] < mag[neigh_1_y, neigh_1_x]:
                        mag[i_y, i_x] = 0
                        continue

                if width > neigh_2_x >= 0 and height > neigh_2_y >= 0:
                    if mag[i_y, i_x] < mag[neigh_2_y, neigh_2_x]:
                        mag[i_y, i_x] = 0

        return mag, weak_threshold, strong_threshold

    def __apply_double_thresholding(self, img, mag, weak_threshold, strong_threshold):
        height, width = img.shape

        ids = np.zeros_like(img)

        for i_x in range(width):
            for i_y in range(height):

                grad_mag = mag[i_y, i_x]

                if grad_mag < weak_threshold:
                    mag[i_y, i_x] = 0
                elif strong_threshold > grad_mag >= weak_threshold:
                    ids[i_y, i_x] = 1
                else:
                    ids[i_y, i_x] = 2

        return mag

    def __apply_edge_tracking_by_hysteresis(
        self, magnitude, low_threshold=30, high_threshold=100
    ):
        rows, cols = magnitude.shape
        edge_map = np.zeros((rows, cols), dtype=np.uint8)

        strong_edge_m, strong_edge_n = np.where(magnitude >= high_threshold)
        weak_edge_m, weak_edge_n = np.where(
            (magnitude >= low_threshold) & (magnitude < high_threshold)
        )

        # mark strong edges as white (255)
        edge_map[strong_edge_m, strong_edge_n] = 255

        # mark weak edges as white if they are connected to strong edges
        for i, j in zip(weak_edge_m, weak_edge_n):
            if (edge_map[i - 1 : i + 2, j - 1 : j + 2] == 255).any():
                edge_map[i, j] = 255

        return edge_map

    def __detect(self):
        image = cv2.imread(self.filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blurr_image = self.__apply_gaussian_blur(image)
        mag, ang = self.__apply_sobel_filter(blurr_image)
        mag_image, weak_th, strong_th = self.__apply_non_maximum_suppression(
            image, mag, ang
        )
        mag_image = self.__apply_double_thresholding(image, mag_image, weak_th, strong_th)
        edged_image = self.__apply_edge_tracking_by_hysteresis(mag_image)

        plt.imsave(GENERATED_DIR + "canny.jpg", edged_image, cmap="gray")
