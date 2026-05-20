import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error

import streamlit as st


def run_prediction(df):

    numeric_cols = df.select_dtypes(
        include='number'
    ).columns

    if len(numeric_cols) < 2:

        st.error(
            "Need at least 2 numeric columns"
        )

        return

    # Select columns
    target_col = st.selectbox(
        "🎯 Select Target Column",
        numeric_cols
    )

    feature_col = st.selectbox(
        "📈 Select Feature Column",
        numeric_cols
    )

    if target_col != feature_col:

        # Prepare data
        X = df[[feature_col]]

        y = df[target_col]

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(

            X,
            y,

            test_size=0.2,

            random_state=42
        )

        # Model
        model = LinearRegression()

        model.fit(
            X_train,
            y_train
        )

        # Predictions
        predictions = model.predict(
            X_test
        )

        # Error
        error = mean_absolute_error(
            y_test,
            predictions
        )

        st.success(
            f"Model trained successfully! MAE: {error:.2f}"
        )

        # Predict custom value
        custom_input = st.number_input(
            f"Enter {feature_col}"
        )

        future_prediction = model.predict(
            [[custom_input]]
        )

        st.subheader(
            "🔮 Prediction Result"
        )

        st.write(

            f"Predicted {target_col}: "

            f"{future_prediction[0]:.2f}"
        )