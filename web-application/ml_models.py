import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline


def supportVector_model():
    svm_pipe = Pipeline([("model", SVC(
        C=10, decision_function_shape="ovr", gamma=0.1, kernel="rbf", probability=True))])
    return svm_pipe


def kNeighbors_model():
    kn_pipe = Pipeline([
        ("model", KNeighborsClassifier(n_neighbors=3, weights='distance', metric='euclidean'))])
    return kn_pipe


def logisticRegression_model():
    log_pipe = Pipeline([("model", LogisticRegression(
        C=0.01, penalty='l2', solver='liblinear'))])
    return log_pipe


def randomForest_model():
    ran_pipe = Pipeline([("model", RandomForestClassifier(
        n_estimators=500, min_samples_split=2, min_samples_leaf=1, max_features="auto", max_depth=6, bootstrap=False))])
    return ran_pipe


def neuralNetwork_model():
    nn_model = Sequential([keras.layers.Flatten(input_shape=(5,)),
                           keras.layers.Dense(5, activation=tf.nn.relu),
                           keras.layers.Dense(5, activation=tf.nn.relu),
                           keras.layers.Dense(1, activation=tf.nn.sigmoid),
                           ])
    nn_model.compile(optimizer="adam", loss="binary_crossentropy",
                     metrics=["accuracy"])
    return nn_model


def decisionTree_model():
    dt_pipe = Pipeline([("model", DecisionTreeClassifier(
        criterion="entropy", max_depth=7, max_features="log2", min_samples_split=6))])
    return dt_pipe


def voting_model():
    clf1 = kNeighbors_model()
    clf2 = supportVector_model()
    clf3 = decisionTree_model()
    estimators = [("kn", clf1),
                  ("svm", clf2), ("dt", clf3)]
    vot_model = VotingClassifier(estimators=estimators, voting="soft")
    return vot_model


def stacking_model():
    clf1 = kNeighbors_model()
    clf2 = supportVector_model()
    clf3 = decisionTree_model()

    estimators = [("kn", clf1),
                  ("svm", clf2), ("dt", clf3)]

    stack_model = StackingClassifier(
        estimators=estimators, final_estimator=clf2)
    return stack_model
