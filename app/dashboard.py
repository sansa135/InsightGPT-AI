import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import yfinance as yf
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    r2_score
)

from sklearn.ensemble import RandomForestRegressor

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="InsightGPT AI",
    page_icon="🚀",
    layout="wide"
)

# ==============================
# LOAD ENV
# ==============================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ==============================
# GEMINI SETUP
# ==============================

try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
except:
    model = None

# ==============================
# SIDEBAR
# ==============================

st.sidebar.title("📊 Navigation")

menu = st.sidebar.radio(
    "Select Module",
    [
        "Dashboard",
        "Analytics",
        "AI Chat",
        "Predictions",
        "Stock Market",
        "Reports"
    ]
)

theme = st.sidebar.selectbox(
    "Theme",
    ["Light", "Dark"]
)

# ==============================
# THEME
# ==============================

if theme == "Dark":

    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0E1117;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# ==============================
# DASHBOARD
# ==============================

if menu == "Dashboard":

    st.title("🚀 InsightGPT AI")
    st.subheader("Enterprise GenAI Analytics Platform")

    uploaded_file = st.file_uploader(
        "Upload Dataset",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:

        # ======================
        # READ FILE
        # ======================

        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        else:
            df = pd.read_excel(uploaded_file)

        st.success("Dataset Uploaded Successfully ✅")

        # ======================
        # OVERVIEW
        # ======================

        st.header("📌 Dataset Overview")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Rows",
            df.shape[0]
        )

        col2.metric(
            "Columns",
            df.shape[1]
        )

        col3.metric(
            "Missing Values",
            df.isnull().sum().sum()
        )

        st.dataframe(df.head())

        # ======================
        # COLUMN INFO
        # ======================

        st.header("📋 Column Information")

        info_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str),
            "Missing Values": df.isnull().sum().values
        })

        st.dataframe(info_df)

        # ======================
        # VISUALIZATION
        # ======================

        st.header("📊 Interactive Visualization")

        numeric_cols = df.select_dtypes(include=np.number).columns

        if len(numeric_cols) > 0:

            x_axis = st.selectbox(
                "Select X Axis",
                numeric_cols
            )

            y_axis = st.selectbox(
                "Select Y Axis",
                numeric_cols,
                index=min(1, len(numeric_cols)-1)
            )

            chart_type = st.selectbox(
                "Chart Type",
                [
                    "Scatter",
                    "Line",
                    "Bar",
                    "Histogram"
                ]
            )

            if chart_type == "Scatter":

                fig = px.scatter(
                    df,
                    x=x_axis,
                    y=y_axis,
                    title=f"{x_axis} vs {y_axis}"
                )

            elif chart_type == "Line":

                fig = px.line(
                    df,
                    x=x_axis,
                    y=y_axis,
                    title=f"{x_axis} vs {y_axis}"
                )

            elif chart_type == "Bar":

                fig = px.bar(
                    df,
                    x=x_axis,
                    y=y_axis,
                    title=f"{x_axis} vs {y_axis}"
                )

            else:

                fig = px.histogram(
                    df,
                    x=x_axis,
                    title=f"{x_axis} Distribution"
                )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

# ==============================
# ANALYTICS MODULE
# ==============================

elif menu == "Analytics":

    st.title("📈 Advanced Analytics")

    uploaded_file = st.file_uploader(
        "Upload Dataset",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:

        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("Statistical Summary")

        st.dataframe(df.describe())

        st.subheader("Correlation Matrix")

        numeric_df = df.select_dtypes(include=np.number)

        if not numeric_df.empty:

            corr = numeric_df.corr()

            fig = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                title="Correlation Heatmap"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.subheader("Missing Values")

        missing = df.isnull().sum()

        missing_df = pd.DataFrame({
            "Column": missing.index,
            "Missing Values": missing.values
        })

        st.dataframe(missing_df)

# ==============================
# AI CHAT MODULE
# ==============================

elif menu == "AI Chat":

    st.title("🤖 AI Data Assistant")

    prompt = st.text_area(
        "Ask AI anything about Analytics / ML / Data"
    )

    if st.button("Generate AI Response"):

        if model is not None:

            try:

                response = model.generate_content(prompt)

                st.success("AI Response Generated ✅")

                st.write(response.text)

            except Exception as e:

                st.error(f"Error: {e}")

        else:

            st.error("Gemini API Key Missing")

# ==============================
# PREDICTION MODULE
# ==============================

elif menu == "Predictions":

    st.title("🔮 ML Prediction Engine")

    uploaded_file = st.file_uploader(
        "Upload Dataset",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.dataframe(df.head())

        numeric_cols = list(
            df.select_dtypes(include=np.number).columns
        )

        target = st.selectbox(
            "Select Target Column",
            numeric_cols
        )

        features = st.multiselect(
            "Select Feature Columns",
            numeric_cols
        )

        model_type = st.selectbox(
            "Choose ML Model",
            [
                "Linear Regression",
                "Random Forest"
            ]
        )

        if st.button("Train Model"):

            if len(features) > 0:

                X = df[features]
                y = df[target]

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=42
                )

                if model_type == "Linear Regression":

                    model_ml = LinearRegression()

                else:

                    model_ml = RandomForestRegressor()

                model_ml.fit(
                    X_train,
                    y_train
                )

                preds = model_ml.predict(X_test)

                mae = mean_absolute_error(
                    y_test,
                    preds
                )

                r2 = r2_score(
                    y_test,
                    preds
                )

                st.success("Model Trained Successfully ✅")

                col1, col2 = st.columns(2)

                col1.metric(
                    "MAE",
                    round(mae, 2)
                )

                col2.metric(
                    "R2 Score",
                    round(r2, 2)
                )

                pred_df = pd.DataFrame({
                    "Actual": y_test,
                    "Predicted": preds
                })

                st.dataframe(pred_df.head())

                fig = px.scatter(
                    pred_df,
                    x="Actual",
                    y="Predicted",
                    title="Actual vs Predicted"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

# ==============================
# STOCK MARKET MODULE
# ==============================

elif menu == "Stock Market":

    st.title("📈 AI Stock Market Analytics")

    stock = st.text_input(
        "Enter Stock Symbol",
        "AAPL"
    )

    if st.button("Analyze Stock"):

        try:

            data = yf.download(
                stock,
                period="1y"
            )

            st.success("Stock Data Loaded ✅")

            st.dataframe(data.tail())

            fig = px.line(
                data,
                x=data.index,
                y="Close",
                title=f"{stock} Stock Price"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            if model is not None:

                prompt = f"""
                Analyze this stock:
                {stock}

                Latest Price:
                {data['Close'].iloc[-1]}
                """

                response = model.generate_content(prompt)

                st.subheader("🤖 AI Market Insight")

                st.write(response.text)

        except Exception as e:

            st.error(f"Error: {e}")

# ==============================
# REPORTS MODULE
# ==============================

elif menu == "Reports":

    st.title("📄 AI Reports")

    st.info(
        "PDF Report Generation Coming Soon 🔥"
    )

    st.markdown("""
    ### Features Included
    - AI Analytics
    - ML Prediction
    - Stock Market AI
    - Interactive Charts
    - Data Visualization
    - Enterprise Dashboard
    - Gemini AI Assistant
    """)