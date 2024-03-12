from flask import Flask, flash, redirect, request, render_template
import os
from algorithms.sobel_detector import SobelDetector
from algorithms.prewitt_detector import PrewittDetector
from algorithms.roberts_detector import RobertsDetector
from algorithms.frei_chen_detector import FreiChenDetector
from algorithms.scharr_detector import ScharrDetector
from algorithms.canny_detector import CannyDetector
from algorithms.log_detector import LOGDetector
from algorithms.k_means_segmentation import KMeansSegmentation
from algorithms.threshold_outsu_segmentation import ThresholdOtsuSegmentation
from algorithms.mtcnn_segmentation import MTCNNSegmentation
from algorithms.svm_classification import SVMClassifier

UPLOAD_DIR = "static/uploads/"
GENERATED_DIR = "static/gen/"
ALLOWED_FILE_EXTENSIONS = [
    "jpg",
    "png",
]

app = Flask(__name__)
app.secret_key = "m00979361"
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["GENERATED_FOLDER"] = GENERATED_DIR
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


@app.route("/")
def root():
    return render_template("main.html")


def allowed_files(filename):
    return filename.rsplit(".", 1)[1].lower() in ALLOWED_FILE_EXTENSIONS


@app.route("/", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        flash("Upload Image first !!!")
        return redirect(request.url)

    file = request.files["file"]
    if file.filename == "":
        flash("No image selected !!!")
        return redirect(request.url)
    if file and allowed_files(file.filename):
        file.save(os.path.join(app.config["UPLOAD_FOLDER"], file.filename))
        flash("Image uploaded successfully !")

        uploaded_image = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)

        SobelDetector(uploaded_image)
        PrewittDetector(uploaded_image)
        RobertsDetector(uploaded_image)
        FreiChenDetector(uploaded_image)
        ScharrDetector(uploaded_image)
        CannyDetector(uploaded_image)
        LOGDetector(uploaded_image)
        KMeansSegmentation(uploaded_image, k=3)
        ThresholdOtsuSegmentation(uploaded_image)
        MTCNNSegmentation(uploaded_image)
        svm_classifier = SVMClassifier(uploaded_image)
        svm_class, svm_score = svm_classifier.detect()

        return render_template(
            "main.html",
            sobel_img=os.path.join(app.config["GENERATED_FOLDER"], "sobel.jpg"),
            prewitt_img=os.path.join(app.config["GENERATED_FOLDER"], "prewitt.jpg"),
            roberts_img=os.path.join(app.config["GENERATED_FOLDER"], "roberts.jpg"),
            frei_chen_img=os.path.join(app.config["GENERATED_FOLDER"], "frei_chen.jpg"),
            scharr_img=os.path.join(app.config["GENERATED_FOLDER"], "scharr.jpg"),
            canny_img=os.path.join(app.config["GENERATED_FOLDER"], "canny.jpg"),
            log_image=os.path.join(app.config["GENERATED_FOLDER"], "laplacian.jpg"),
            k_means_img=os.path.join(app.config["GENERATED_FOLDER"], "kmeans.jpg"),
            thresh_otsu_img=os.path.join(
                app.config["GENERATED_FOLDER"], "threshold_otsu.jpg"
            ),
            mtcnn_img=os.path.join(app.config["GENERATED_FOLDER"], "mtcnn.jpg"),
            svm_img=os.path.join(app.config["GENERATED_FOLDER"], "svm.jpg"),
            svm_class=svm_class,
            svm_score=svm_score,
        )
    else:
        flash("Only .jpg and .png image types are allowed")
        return redirect(request.url)


if __name__ == "__main__":
    app.run(debug=True)
