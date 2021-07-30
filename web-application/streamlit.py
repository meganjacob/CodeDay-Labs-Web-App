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
df_data.set_index("name", inplace=True)
df_data.drop(["MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "Jitter:DDP", "Shimmer:APQ3", "MDVP:APQ",
              "Shimmer:DDA", "NHR", "HNR", "DFA", "spread1", "PPE"], axis=1, inplace=True)

# Features and Target
X = df_data.drop(columns=["status"], axis=1)
y = df_data["status"]
status = {0: "Parkinson's Negative", 1: "Parkinson's Positive"}

X = StandardScaler().fit_transform(X)

# Display the dataset
with st.beta_expander("Data Frame Preview"):
    st.dataframe(df_data.head(10))

# Parameter descriptions
st.subheader("**Parameter Descriptions**")
st.markdown("""
    - **MDVP:Shimmer:** General difference between cycle amplitudes.
    - **MDVP:Shimmer(dB):** Local shimmer in decibels.
    - **MDVP:Fo(Hz):** Average vocal fundamental frequency. Averaged lowest frequencies in waveform period.
    - **MDVP:Fhi(Hz):** Maximum vocal fundamental frequency. Highest floor frequency in waveform period.
    - **Shimmer:APQ5:** Quotient of amplitude disturbance within five periods.
    - **MDVP:PPQ:** Average absolute difference between a period and its adjacent ones, divided by average period.
    - **MDVP:RAP:** Relative amplitude differences between cycles.
    - **RPDE:** Method that determines the repetitivity of a sound.
    - **Spread2** Deviating change over time from standard frequency.
    - **D2:** Dimensionality measurement to characterize the attractor.
    - **Status:** Healthy person (0), Parkinson's positive (1).
    """)
st.write("***")

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
    MDVP_Shimmer = st.sidebar.number_input(
        'MDVP:Shimmer', max_value=0.15, value=0.0297, format="%.4f",)
    MDVP_Shimmer_dB = st.sidebar.number_input(
        'MDVP:Shimmer(dB)', 0.0, max_value=2.0, value=0.2822, format="%.4f")
    MDVP_Fo_Hz = st.sidebar.number_input(
        'MDVP:Fo(Hz)', min_value=80.0, max_value=280.0, value=154.2286, format="%.4f")
    MDVP_Fhi_Hz = st.sidebar.number_input(
        'MDVP:Fhi(Hz)', min_value=100.0, max_value=600.0, value=197.1049, format="%.4f")
    Shimmer_APQ5 = st.sidebar.number_input(
        'Shimmer:APQ5', max_value=0.1, value=0.0178, format="%.4f")
    MDVP_PPQ = st.sidebar.number_input(
        'MDVP:PPQ', max_value=0.03, value=0.0034, format="%.4f")
    MDVP_RAP = st.sidebar.number_input(
        'MDVP:RAP', max_value=0.03, value=0.0033, format="%.4f")
    RPDE = st.sidebar.number_input(
        'RPDE', max_value=0.8, value=0.4985, format="%.4f")
    spread2 = st.sidebar.number_input(
        'spread2', max_value=0.6, value=0.2265, format="%.4f")
    D2 = st.sidebar.number_input(
        'D2', min_value=1.0, max_value=4.0, value=2.3818, format="%.4f")

    data = {'Model': model,
            'MDVP:Shimmer': MDVP_Shimmer,
            'MDVP:Shimmer(dB)': MDVP_Shimmer_dB,
            'MDVP:Fo(Hz)': MDVP_Fo_Hz,
            'MDVP:Fhi(Hz)': MDVP_Fhi_Hz,
            'Shimmer:APQ5': Shimmer_APQ5,
            'MDVP:PPQ': MDVP_PPQ,
            'MDVP:RAP': MDVP_RAP,
            'RPDE': RPDE,
            'spread2': spread2,
            'D2': D2}
    features = pd.DataFrame(data, index=[0])
    return features


# Middle of sidebar
with sidebar_col2:
    df = user_input_features()

# Bottom of sidebar
with sidebar_col3:
    changes = st.sidebar.button("Confirm Changes")

# Only when user confirms the changes
if changes:
    model = df.loc[0, "Model"]
    df.drop(columns=["Model"], axis=1, inplace=True)
    df = df.values.reshape(1, -1)
    df = StandardScaler().fit_transform(df)

    if model == "SupportVectorClassifier":
        svm_model = ml_models.supportVector_model()
        svm_model.fit(X, y)
        predict = svm_model.predict(df)
    elif model == "LogisticRegressionClassifier":
        log_model = ml_models.logisticRegression_model()
        log_model.fit(X, y)
        predict = log_model.predict(df)
    elif model == "RandomForestClassifier":
        ran_model = ml_models.randomForest_model()
        ran_model.fit(X, y)
        predict = ran_model.predict(df)
    elif model == "DecisionTreeClassifier":
        dt_model = ml_models.decisionTree_model()
        dt_model.fit(X, y)
        predict = dt_model.predict(df)
    elif model == "NeuralNetwork":
        nn_model = ml_models.neuralNetwork_model()
        nn_model.fit(X, y, epochs=50, batch_size=1, verbose=0)
        predict = nn_model.predict(df)
    elif model == "KNeighborsClassifier":
        kn_model = ml_models.kNeighbors_model()
        kn_model.fit(X, y)
        predict = kn_model.predict(df)
    elif model == "VotingClassifier":
        vot_model = ml_models.voting_model()
        vot_model.fit(X, y)
        predict = vot_model.predict(df)

    st.write(f"**Output:** {status[predict[0]]}")
