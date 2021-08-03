import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
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
st.write("This app builds various Machine Learning models and predicts the **Parkinsons's disease**.")

# Load the dataset
df_data = pd.read_csv("parkinsons.csv")
df_data = df_data.loc[:, ["status", "name", "MDVP:Fo(Hz)", "RPDE",
                          "DFA", "PPE", "spread2"]]

# Features and Target
X = df_data.drop(columns=["name", "status"]).values
y = df_data["status"]
status = {0: "Parkinson's Negative", 1: "Parkinson's Positive"}

scale = StandardScaler()
scale.fit(X)
X = scale.transform(X)

# Display the dataset
with st.beta_expander("Data Frame Preview"):
    st.dataframe(df_data.loc[36:46, :].set_index("name"))
    st.write("See the [csv](https://github.com/meganjacob/CodeDay-Labs-Web-App/blob/main/web-application/parkinsons.csv) input file")

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

# Display the heatmap
with st.beta_expander("Intercorrelation Heatmap"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax = sns.heatmap(df_data.set_index("name").corr(),
                     fmt='.2f', annot=True, cmap="YlGnBu")
    st.pyplot(fig)

st.write("***")
st.subheader("**Mean Values of Each Column**")
st.dataframe(df_data.groupby("status").mean())
st.write("***")

# Sidebar Organization
sidebar_col1, sidebar_col2, sidebar_col3 = st.sidebar.beta_columns(3)

# Top of sidebar
with sidebar_col1:
    st.sidebar.header('User Input Parameters')

# Get user inputs


def user_input_features():
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

    data = {'MDVP:Fo(Hz)': MDVP_Fo_Hz,
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

    df = scale.transform(df)

    svc_model = ml_models.supportVector_model()
    ran_model = ml_models.randomForest_model()
    stacking_model = ml_models.stacking_model()
    voting_model = ml_models.voting_model()
    models = [svc_model, ran_model, stacking_model, voting_model]
    model_names = ["SupportVectorClassifier", "RandomForestClassifier",
                   "StackingClassifier", "VotingClassifier"]

    model_probabilities = {}
    model_predictions = {}
    for index, model in enumerate(models):
        model.fit(X, y)
        predict = model.predict(df)
        proba = model.predict_proba(df)
        model_probabilities[model_names[index]] = round(
            proba[0][round(predict[0])], 4) * 100
        model_predictions[model_names[index]] = status[round(predict[0])]

    best_model = max(model_probabilities, key=model_probabilities.get)

    st.write("**Predictions**")
    st.write(pd.DataFrame(model_predictions, index=["Prediction"]))
    with st.beta_expander("Confidence Chart"):
        st.dataframe(pd.DataFrame(model_probabilities, index=["Confidence"]))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax = plt.bar(model_names,
                     model_probabilities.values(), color="y", label="str")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    best_model = max(model_probabilities, key=model_probabilities.get)
    st.write(f"**Output:** {model_predictions[best_model]}")
    st.write(
        f"{best_model} has the highest confidence on the prediction with {model_probabilities[best_model]}%")

st.write("#")

st.subheader("**About This App**")
st.write("This project is a part of **CodeDay Program**. Led by [Megan Jacob](https://github.com/meganjacob), and built by [Yusa Kaya](https://github.com/mrbonabane), [Josh Tagle](https://github.com/JWizard05), [Ananya Unnikrishnan](https://github.com/s-aunnikrishnan).")
st.write(
    "See the project in the [Github](https://github.com/meganjacob/CodeDay-Labs-Web-App)")
