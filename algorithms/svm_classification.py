import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

UPLOAD_DIR = "static/uploads/"
GENERATED_DIR = "static/gen/"


class SVMClassifier:
    def __init__(self, filename) -> None:
        self.filename = filename
        self.model_file = r"models\human-non-human.sav"
        self.categories = ["human", "non-human"]

    def __train_model(self):
        flat_data_arr = []  # input array
        target_arr = []  # output array
        datadir = "data"
        # path which contains all the categories of images
        for category in self.categories:
            print(f"loading... category : {category}")
            path = os.path.join(datadir, category)
            for img in os.listdir(path):
                img_array = imread(os.path.join(path, img))
                img_resized = resize(img_array, (150, 150, 3))
                flat_data_arr.append(img_resized.flatten())
                target_arr.append(self.categories.index(category))
            print(f"loaded category:{category} successfully")
        flat_data = np.array(flat_data_arr)
        target = np.array(target_arr)

        df = pd.DataFrame(flat_data)
        df["Target"] = target

        x = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.20, random_state=77, stratify=y
        )

        param_grid = {
            "C": [0.1, 1, 10, 100],
            "gamma": [0.0001, 0.001, 0.1, 1],
            "kernel": ["rbf", "poly"],
        }

        svc = svm.SVC(probability=True)

        model = GridSearchCV(svc, param_grid, scoring="accuracy", verbose=10)
        model.fit(x_train, y_train)

        # Pickling the model
        pickle.dump(model, open(self.model_file, "wb"))

        y_pred = model.predict(x_test)

        # Calculating the accuracy of the model
        accuracy = accuracy_score(y_pred, y_test)
        print(f"The model is {accuracy*100}% accurate")

    def detect(self):
        with open(self.model_file, "rb") as file:
            model = pickle.load(file)

        image = cv2.imread(self.filename)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        resized_img = resize(image, (150, 150, 3))
        l = [resized_img.flatten()]

        probability = model.predict_proba(l)
        for ind, val in enumerate(self.categories):
            print(f"{val} = {probability[0][ind]*100}%")

        plt.imsave(GENERATED_DIR + "svm.jpg", image)
        return self.categories[model.predict(l)[0]], probability[0][model.predict(l)[0]]
