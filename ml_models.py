import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn import tree
from sklearn import metrics


def supportVector_model(X, y):
    svm_pipe = Pipeline([("scale", StandardScaler()), ("model", SVC(
        C=10, decision_function_shape="ovr", gamma=0.1, kernel="rbf", probability=True))])
    svm_pipe.fit(X, y)
    return svm_pipe


def kNeighbors_model(X, y):
    kn_pipe = Pipeline([
        ("model", KNeighborsClassifier(n_neighbors=5))])
    kn_pipe.fit(X, y)
    return kn_pipe


def logisticRegression_model(X, y):
    log_pipe = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(
        C=0.01, penalty='l2', solver='liblinear'))])
    log_pipe.fit(X, y)
    return log_pipe


def randomForest_model(X, y):
    ran_pipe = Pipeline([("scale", StandardScaler()), ("model", RandomForestClassifier(
        n_estimators=600, min_samples_split=10, min_samples_leaf=4, max_features="sqrt", max_depth=110, bootstrap=True))])
    ran_pipe.fit(X, y)
    return ran_pipe


def neuralNetwork_model(X, y):
    nn_model = Sequential([keras.layers.Flatten(input_shape=(10,)),
                           keras.layers.Dense(10, activation=tf.nn.relu),
                           keras.layers.Dense(10, activation=tf.nn.relu),
                           keras.layers.Dense(1, activation=tf.nn.sigmoid),
                           ])
    nn_model.compile(optimizer="adam", loss="binary_crossentropy",
                     metrics=["accuracy"])
    nn_model.fit(X, y, epochs=25, batch_size=1, verbose=0)
    return nn_model


def decisionTree_model(X, y):
    dt_pipe = Pipeline([("scale", StandardScaler()), ("model", DecisionTreeClassifier(
        criterion="entropy", max_depth=9, max_features="sqrt", min_samples_split=4))])
    dt_pipe.fit(X, y)
    return dt_pipe

def ensamble_model(X, y):
