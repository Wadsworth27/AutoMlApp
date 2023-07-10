import streamlit as st
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import ydata_profiling
import numpy as np
import pickle
from stml import STMLRunner
from pycaret import regression, classification

df = None
stml = STMLRunner()
with st.sidebar:
    st.title("AutoML For Everyone")
    app_choice = st.radio(
        "App Select", ["Build a new model", "Predict using pretrained model"]
    )
    if app_choice == "Build a new model":
        choice = st.radio(
            "Navigation", ["Upload", "Profiling", "ML", "Predict", "Download"]
        )
        st.session_state["app_state"] = "build"
    if app_choice == "Predict using pretrained model":
        choice = st.radio(
            "Navigation", ["Upload Model and Data", "Profile Data", "Predict Data"]
        )
        st.session_state["app_state"] = "existing"
    st.info("This application allows you to build an automated ML Pipeline.")
if app_choice == "Build a new model":
    if choice == "Upload":
        stml.run_create_upload_tab()
    if choice == "Profiling":
        stml.run_create_profiling()
    if choice == "ML":
        stml.run_model_builder()
    if choice == "Predict":
        stml.run_prediction()

    if choice == "Download":
        stml.run_create_download()

if app_choice == "Predict using pretrained model":
    if choice == "Upload Model and Data":
        model_type = st.radio("Select Model Type:", ["Regression", "Classification"])
        model_file = st.file_uploader("Load your model as a .pkl here:", type="pkl")
        if model_file:
            if model_type == "Regression":
                try:
                    st.session_state["regression_model"] = regression.load_model(
                        model_file.name[:-4]
                    )
                    st.session_state["regression_model"]
                except Exception as e:
                    st.write(e)
                    st.info("Error Loading model, please ensure it is a .pkl file")
            if "regression_model" in st.session_state:
                if "classification_model" in st.session_state:
                    del st.session_state["classification_model"]
                data_file = st.file_uploader("Upload your Dataset here", type="csv")
                st.session_state["pretained_model_dataframe"] = pd.read_csv(data_file)
                if "pretained_model_dataframe" in st.session_state:
                    st.write("All set to profile and predict! Proceed to next steps.")
                    st.dataframe(
                        regression.predict_model(
                            st.session_state["regression_model"],
                            st.session_state["data_file"],
                        )
                    )
            if model_type == "Classification":
                try:
                    st.session_state[
                        "classification_model"
                    ] = classification.load_model(model_file.name[:-4])
                    st.session_state["classification_model"]
                except Exception as e:
                    st.write(e)
                    st.info("Error Loading model, please ensure it is a .pkl file")
            if "classification_model" in st.session_state:
                if "regression_model" in st.session_state:
                    del st.session_state["regression_model"]
                data_file = st.file_uploader("Upload your Dataset here", type="csv")
                st.session_state["data_file"] = pd.read_csv(data_file)
                if "data_file" in st.session_state:
                    st.dataframe(
                        classification.predict_model(
                            st.session_state["regression_model"],
                            st.session_state["data_file"],
                        )
                    )
    if choice == "Profile Data":
        if "pretained_model_dataframe" in st.session_state:
            df = st.session_state["pretained_model_dataframe"]
            profile = df.profile_report()
            st_profile_report(profile)
    if choice == "Predict Data":
        if "pretained_model_dataframe" in st.session_state:
            if "regression_model" in st.session_state:
                model = st.session_state["regression_model"]
                data = st.session_state["pretained_model_dataframe"]
                if st.button(
                    "Click here to make predictions using your pretrained model!"
                ):
                    if "regression_model" in st.session_state:
                        prediction_df = regression.predict_model(model, data)
                        st.dataframe(prediction_df)
                        st.info("Click below to dowload results")
                        st.download_button(
                            "Download",
                            prediction_df.to_csv(index=False),
                            "results.csv",
                        )
            elif "classification_model" in st.session_state:
                pass
            else:
                st.info("Please upload a model before proceeding to this step.")
        else:
            st.info("Please upload a dataset before proceeding to this step.")
