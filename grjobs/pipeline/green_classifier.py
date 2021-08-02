# File: pipeline/green_classifier.py

"""Module for GreenClassifier class.

  Typical usage example:

  python grjobs/pipeline/green_classifier.py

"""
# ---------------------------------------------------------------------------------
import numpy as np
from collections import Counter
import os
import json
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

from grjobs.utils.text_cleaning_utils import clean_text
from grjobs import get_yaml_config, Path, PROJECT_DIR
from grjobs.getters.keywords import get_expanded_green_words
from grjobs.pipeline.create_labelled_data import green_count

# ---------------------------------------------------------------------------------
# Load config file
grjobs_config = get_yaml_config(Path(str(PROJECT_DIR) + "/grjobs/config/base.yaml"))


# get model ouputs path
pretrained_model_path = str(PROJECT_DIR) + grjobs_config["MODEL_OUTPUT_PATH"]


class GreenClassifier:
    """
    A green classifier class to train/save/load/predict whether
    a job is green or not.

    Attributes:
        split_random_seed: int(default = 42)

    Methods:
        split_data(labelled_data): splits the data
        into test/train sets
        fit(X_train, y_train): fit the vectoriser and classifier
        to the split data
        transform(X_test): predict classes from vectorised text
        evaluate(y_test, y_pred): print classification report
        and confusion matrix based on pipeline
        save_model(file_name): save model to yaml file
        load_model(file_name): load model saved to yaml file
    """

    def __init__(self, split_random_seed=42):
        self.split_random_seed = split_random_seed

    def preprocess_text(self, job_ads):

        for ad in job_ads:
            ad["clean_description"] = clean_text(
                ad["job_title_raw"] + " " + ad["description"]
            )

        return job_ads

    def preprocess_green_count(self, job_ads):

        expanded_green_words = get_expanded_green_words()

        for ad in job_ads:
            ad["green_count"] = green_count(
                ad["clean_description"], expanded_green_words
            )

        return [ad["green_count"] for ad in job_ads]

    def split_data(self, job_ads, test_size=0.15, verbose=False):

        X = [{k: v for k, v in job_ad.items() if k != "label"} for job_ad in job_ads]
        y = [t["label"] for t in job_ads]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.split_random_seed
        )

        if verbose:
            print(f"Size of training data: {len(y_train)}")
            print(f"Size of test data: {len(y_test)}")
            print(f"Counter of training data classes: {Counter(y_train)}")
            print(f"Counter of test data classes: {Counter(y_test)}")

        return X_train, X_test, y_train, y_test

    def fit(self, X_train, y_train):

        X_train = self.preprocess_text(X_train)
        green_counts = np.array(self.preprocess_green_count(X_train))
        self.vectoriser = TfidfVectorizer(
            min_df=grjobs_config["min_df"], max_df=grjobs_config["max_df"]
        )

        X_vec = self.vectoriser.fit_transform(
            [x["clean_description"] for x in X_train]
        ).toarray()
        X_green_vec = np.hstack((X_vec, green_counts[:, None]))

        # Fit classifier
        self.classifier = Pipeline(
            [
                ("sampling", SMOTE(random_state=self.split_random_seed)),
                (
                    "classifier",
                    XGBClassifier(
                        max_depth=grjobs_config["max_depth"],
                        min_child_weight=grjobs_config["min_child_weight"],
                    ),
                ),
            ]
        )

        self.classifier.fit(X_green_vec, y_train)

    def transform(self, X):

        X = self.preprocess_text(X)
        green_counts = np.array(self.preprocess_green_count(X))
        X_vec = self.vectoriser.transform([t["clean_description"] for t in X]).toarray()
        X_green_vec = np.hstack((X_vec, green_counts[:, None]))
        y_pred = self.classifier.predict(X_green_vec)

        return y_pred

    def predict(self, X):
        return self.transform(X)

    def evaluate(self, y, y_pred, verbose=True):
        class_rep = classification_report(y, y_pred, output_dict=True)
        if verbose:
            print(classification_report(y, y_pred))
            print(confusion_matrix(y, y_pred))
        return class_rep

    def save_model(self, file_name):

        model_path = pretrained_model_path + file_name + '.pkl'

        with open(model_path, "wb") as f:
            pickle.dump(self, f)

def load_model(file_name):

    model_path = pretrained_model_path + file_name + '.pkl'

    with open(model_path, "rb") as f:
        green_model = pickle.load(f)

    return green_model