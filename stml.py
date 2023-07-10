import streamlit as st
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import ydata_profiling
from pycaret import regression, classification
from STMLUtility import STMLUtility


class STMLRunner:
    def __init__(self):
        self.app_state = "create"
        self.stmlutil = STMLUtility()

    def run_create_upload_tab(self):
        st.title("Upload your Data for Modeling")
        col1, col2 = st.columns([3, 1])

        with col1:
            file = st.file_uploader("Upload your Dataset here.")
            if file:
                df = pd.read_csv(file)
                st.session_state["create_data"] = df
            st.write("Or use a sample dataset below:")
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                st.write("Regression")
                if st.button("Use Redwine Quality Dataset"):
                    st.session_state["create_data"] = pd.read_csv("winequality-red.csv")
                    st.session_state["target"] = "quality"
                    st.session_state["regression_sample_used"] = True
            with sub_col2:
                st.write("Classification")
                if st.button("Use Iris Data Set"):
                    st.session_state["create_data"] = pd.read_csv("iris.csv")
                    st.session_state["target"] = "Species"
                    st.session_state["classification_sample_used"] = True

            if "create_data" in st.session_state:
                df = st.session_state["create_data"]
                st.dataframe(df)
                st.write(
                    "Select target column for dataset if applicable, select only numeric types for regression."
                )
                if "sample_used" in st.session_state:
                    st.info(
                        "Because you used a sample data set, the recommended target has been set for you."
                    )

                try:
                    st.session_state["target"] = st.selectbox(
                        label="Select your target variable.",
                        options=df.columns.tolist(),
                        index=self.stmlutil.find_target_index(
                            st.session_state["target"], df.columns.tolist()
                        ),
                    )
                except:
                    st.session_state["target"] = st.selectbox(
                        label="Select your target variable.",
                        options=df.columns.tolist(),
                    )
                st.dataframe(df.dtypes)
                with col2:
                    st.write("Utilities")
                    selected_columns = st.multiselect(
                        "Select Columns to include.",
                        options=df.columns.tolist(),
                        default=df.columns.tolist(),
                    )
                    if st.button("Reduce to selected columns."):
                        for column in df.columns.tolist():
                            if column not in selected_columns:
                                df.drop(column, axis=1, inplace=True)
                        st.session_state["create_data"] = df
                        st.experimental_rerun()

    def run_create_profiling(self):
        st.title("Data Profiling")
        if "create_data" in st.session_state:
            df = st.session_state["create_data"]
            profile = df.profile_report()
            st_profile_report(profile)
        else:
            st.error("You must upload data first in order to run profiling")

    def run_model_builder(self):
        st.title("ML Model Build Tool")
        model_build_load = st.radio(
            "Create a new ML Model.",
            ["Create Regression Model", "Create Classification Model"],
        )
        if "regression_sample_used" in st.session_state:
            st.info(
                "Because you used the Wine Sample data, you should select regression here."
            )
        if "classification_sample_used" in st.session_state:
            st.info(
                "Because you used the Iris Sample data, you should select clasification here"
            )
        if st.button("What's the difference?"):
            st.info(
                "Regression helps predict a continuous quantity(like a house price), classification predicts discrete class labels(like a type of flower)"
            )
        if model_build_load == "Create Regression Model":
            if "create_data" in st.session_state:
                df = st.session_state["create_data"].copy()
                if st.button("Setup and run Experiments"):
                    st.info("These are the ML experiment settings")
                    regression.setup(
                        df, target=st.session_state["target"], verbose=False
                    )
                    setup_df = regression.pull()
                    st.dataframe(setup_df)
                    st.info("Experiments running, this may take several minutes.")
                    best_model = regression.compare_models()
                    st.session_state["best_model"] = best_model
                    st.session_state["model_type"] = "regression"
                    compare_df = regression.pull()
                    st.dataframe(compare_df)
                    st.info("This is the best model")
                    st.write(best_model)
                    st.write(
                        "Click here to see sample predictions on the loaded data. You can predict on additional data or download the model in the next steps."
                    )
                    st.write("Sample predictions:")
                    st.dataframe(regression.predict_model(best_model, df).head())
                    try:
                        regression.plot_model(
                            best_model, plot="pipeline", display_format="streamlit"
                        )
                    except:
                        pass
                    try:
                        regression.plot_model(
                            best_model,
                            plot="residuals_interactive",
                            display_format="streamlit",
                        )
                    except:
                        pass
                    try:
                        regression.plot_model(
                            best_model, plot="feature", display_format="streamlit"
                        )
                    except:
                        pass
        if model_build_load == "Create Classification Model":
            if "create_data" in st.session_state:
                df = st.session_state["create_data"]
                if st.button("Setup and run Experiments"):
                    st.info("These are the ML experiment settings")
                    classification.setup(
                        df, target=st.session_state["target"], verbose=False
                    )
                    setup_df = classification.pull()
                    st.dataframe(setup_df)
                    st.info("Experiments running, this may take several minutes.")
                    best_model = classification.compare_models()
                    st.session_state["best_model"] = best_model
                    st.session_state["model_type"] = "classification"
                    compare_df = classification.pull()
                    st.info("This is the best model")
                    st.dataframe(compare_df)
                    st.session_state["best_model"] = best_model
                    st.write(
                        "Click here to see sample predictions on the loaded data. You can predict on additional data or download the model in the next steps."
                    )
                    if st.button("Sample predictions"):
                        st.dataframe(classification.predict_model(best_model, df))
                    classification.plot_model(
                        best_model, plot="learning", display_format="streamlit"
                    )
                    try:
                        classification.plot_model(
                            best_model, plot="pipeline", display_format="streamlit"
                        )
                    except:
                        pass
                    try:
                        classification.plot_model(
                            best_model,
                            plot="confusion_matrix",
                            display_format="streamlit",
                        )
                    except:
                        pass
                    try:
                        classification.plot_model(
                            best_model, plot="feature", display_format="streamlit"
                        )
                    except:
                        pass

    def run_prediction(self):
        predict_choice = st.radio(
            "Use data from previous steps or upload a new dataset",
            ["Previous Step Data", "New Data"],
        )
        if predict_choice == "Previous Step Data":
            st.title("Predict")
            if "best_model" in st.session_state:
                df = st.session_state["create_data"]
                st.info("Displaying first 5 rows")
                st.dataframe(st.session_state["create_data"].head())
                if st.button("Predict"):
                    best_model = st.session_state["best_model"]
                    st.write(
                        f"Predicting for Target Variable {st.session_state['target']}"
                    )
                    if st.session_state["model_type"] == "regression":
                        regression.setup(data=df, target=st.session_state["target"])
                        st.session_state["prediction_df"] = regression.predict_model(
                            best_model, df
                        )
                    if st.session_state["model_type"] == "classification":
                        classification.setup(data=df, target=st.session_state["target"])
                        st.session_state[
                            "prediction_df"
                        ] = classification.predict_model(best_model, df)

                    st.dataframe(st.session_state["prediction_df"])

            else:
                st.info("You need to train a model before maing predictions")

        if predict_choice == "New Data":
            new_data = st.file_uploader(
                "Upload new data for model here, make sure the format is consistent with the trained model!"
            )
            st.session_state["new_data"] = pd.read_csv(new_data)
            st.dataframe(st.session_state["new_data"])
            if st.button("Predict"):
                best_model = st.session_state["best_model"]
                st.session_state["prediction_df"] = classification.predict_model(
                    best_model, st.session_state["new_data"]
                )
                st.dataframe(st.session_state["prediction_df"])

        if "prediction_df" in st.session_state:
            prediction_df = st.session_state["prediction_df"]
            st.info("Click below to dowload results")
            st.download_button(
                "Download", prediction_df.to_csv(index=False), "results.csv"
            )

    def run_create_download(self):
        st.title("Download Your Model")
        if "model_type" in st.session_state:
            if st.session_state["model_type"] == "regression":
                regression.save_model(st.session_state["best_model"], "top_model")
            if st.session_state["model_type"] == "classification":
                classification.save_model(st.session_state["best_model"], "top_model")
            with open("top_model.pkl", "rb") as f:
                st.download_button("Download Model", f, file_name="top_model.pkl")
        else:
            st.info("You must create a model in order to dowload it.")
