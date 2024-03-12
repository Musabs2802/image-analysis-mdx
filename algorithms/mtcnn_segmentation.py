from mtcnn.mtcnn import MTCNN
import cv2
import matplotlib.pyplot as plt

UPLOAD_DIR = "static/uploads/"
GENERATED_DIR = "static/gen/"


class MTCNNSegmentation:
    def __init__(self, filename) -> None:
        self.filename = filename
        self.__analyze()

    def __analyze(self):
        image = cv2.imread(self.filename)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        model = MTCNN()
        output = model.detect_faces(image)

        for i in range(0, len(output)):
            box = output[i]["box"]
            right_eye = output[i]["keypoints"]["right_eye"]
            left_eye = output[i]["keypoints"]["left_eye"]
            nose = output[i]["keypoints"]["nose"]
            confidence = output[i]["confidence"]
            if confidence > 0.6:
                cv2.rectangle(
                    image,
                    (box[0], box[1]),
                    (box[0] + box[2], box[1] + box[3]),
                    (0, 255, 0),
                    4,
                )
                cv2.circle(image, nose, 3, (255, 0, 0), 4)
                cv2.circle(image, right_eye, 3, (255, 0, 0), 4)
                cv2.circle(image, left_eye, 3, (255, 0, 0), 4)

        plt.imsave(GENERATED_DIR + "mtcnn.jpg", image)
