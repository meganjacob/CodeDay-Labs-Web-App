import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from PIL import Image
from matplotlib import pyplot as plt

import ml_models

# Load image
img = Image.open("neurons.jpg")
st.image(img, use_column_width=True)

# Title and description
st.title("Parkinson's Disease Web App")
st.write("***")
st.write("This app builds various Machiene Learning models and predicts the **Parkinsons's disease**.")

# Load the dataset
df_data = pd.read_csv("parkinsons.csv")
df_data = df_data.loc[:, ["status", "name", "MDVP:Fo(Hz)", "RPDE",
                          "DFA", "PPE", "spread2"]]

# Features and Target
X = df_data.drop(columns=["name", "status"]).values
y = df_data["status"]
status = {0: "Parkinson's Negative", 1: "Parkinson's Positive"}

X = StandardScaler().fit_transform(X)

# Display the dataset
with st.beta_expander("Data Frame Preview"):
    st.dataframe(df_data.loc[40:50, :].set_index("name"))

# Parameter descriptions
st.subheader("**Parameter Descriptions**")
st.markdown("""
    - **MDVP:Fo(Hz):** Average vocal fundamental frequency. Averaged lowest frequencies in waveform period.
    - **RPDE:** Method that determines the repetitivity of a sound.
    - **DFA:** Signal fractal scaling component. Fractal analysis of graph trajectories.
    - **PPE:** Dysphonia measure that ignores healthy variations and outside noise.
    - **Spread2** Deviating change over time from standard frequency.
    - **Status:** Healthy person (0), Parkinson's positive (1).
    """)
st.write("***")

st.subheader("**Mean Values of Each Column**")
st.dataframe(df_data.groupby("status").mean())


# Sidebar Organization
sidebar_col1, sidebar_col2, sidebar_col3 = st.sidebar.beta_columns(3)

# Top of sidebar
with sidebar_col1:
    st.sidebar.header('User Input Parameters')

# Get user inputs


def user_input_features():
    model = st.sidebar.selectbox("Machiene Learning Model", [
        "SupportVectorClassifier", "LogisticRegressionClassifier", "DecisionTreeClassifier", "RandomForestClassifier", "KNeighborsClassifier", "NeuralNetwork", "VotingClassifier"])
    st.sidebar.write("Features")
    MDVP_Fo_Hz = st.sidebar.number_input(
        'MDVP:Fo(Hz)', min_value=80.0, max_value=280.0, value=154.2286, format="%.4f", step=0.01)
    RPDE = st.sidebar.number_input(
        'RPDE', min_value=0.0, max_value=0.8, value=0.4985, format="%.4f", step=0.01)
    DFA = st.sidebar.number_input(
        'DFA', min_value=0.0, max_value=1.0, value=0.7180, format="%.4f", step=0.01)
    PPE = st.sidebar.number_input(
        'PPE', min_value=0.0, max_value=0.6, value=0.2065, format="%.4f", step=0.01)
    spread2 = st.sidebar.number_input(
        'spread2', min_value=0.0, max_value=0.6, value=0.2265, format="%.4f", step=0.01)

    data = {'Model': model,
            'MDVP:Fo(Hz)': MDVP_Fo_Hz,
            'RPDE': RPDE,
            'DFA': DFA,
            'PPE': PPE,
            'spread2': spread2}
    features = pd.DataFrame(data, index=[0])
    return features


# Middle of sidebar
with sidebar_col2:
    df = user_input_features()

# Bottom of sidebar
with sidebar_col3:
    changes = st.sidebar.button("Generate Prediction")

# Only when user confirms the changes
if changes:
    model = df.loc[0, "Model"]
    df.drop(columns=["Model"], axis=1, inplace=True)
    df = StandardScaler().fit_transform(df)
    df = df.reshape(1, -1)

    if model == "SupportVectorClassifier":
        svm_model = ml_models.supportVector_model()
        svm_model.fit(X, y)
        predict = svm_model.predict(df)
        proba = svm_model.predict_proba(df)
    elif model == "LogisticRegressionClassifier":
        log_model = ml_models.logisticRegression_model()
        log_model.fit(X, y)
        predict = log_model.predict(df)
        proba = log_model.predict_proba(df)
    elif model == "RandomForestClassifier":
        ran_model = ml_models.randomForest_model()
        ran_model.fit(X, y)
        predict = ran_model.predict(df)
        proba = ran_model.predict_proba(df)
    elif model == "DecisionTreeClassifier":
        dt_model = ml_models.decisionTree_model()
        dt_model.fit(X, y)
        predict = dt_model.predict(df)
        proba = dt_model.predict_proba(df)
    elif model == "NeuralNetwork":
        nn_model = ml_models.neuralNetwork_model()
        nn_model.fit(X, y, epochs=50, batch_size=1, verbose=0)
        predict = nn_model.predict(df)[0]
    elif model == "KNeighborsClassifier":
        kn_model = ml_models.kNeighbors_model()
        kn_model.fit(X, y)
        predict = kn_model.predict(df)
        proba = kn_model.predict_proba(df)
    elif model == "VotingClassifier":
        vot_model = ml_models.voting_model()
        vot_model.fit(X, y)
        predict = vot_model.predict(df)
        proba = vot_model.predict_proba(df)

    st.write(f"**Output:** {status[round(predict[0])]}")
    st.write(
        f"{model}'s confidence on the prediction is {round(proba[0][round(predict[0])], 4) * 100}%")
    st.write(proba)
