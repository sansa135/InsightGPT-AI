import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    r2_score
)

from sqlalchemy import create_engine
import google.generativeai as genai
from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv("AIzaSyDLrIaJof9l0Z2EjARrT4IQA9AY8IrOUVo")

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(

    page_title="InsightGPT AI",

    page_icon="🚀",

    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""

<style>

body {

    background-color: #0E1117;
}

.main {

    background-color: #0E1117;
}

h1, h2, h3, h4 {

    color: white;
}

[data-testid="stSidebar"] {

    background-color: #111827;
}

.stButton>button {

    background-color: #4F46E5;

    color: white;

    border-radius: 10px;

    border: none;

    padding: 10px 20px;
}

.stMetric {

    background-color: #1F2937;

    padding: 15px;

    border-radius: 15px;
}

</style>

""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🚀 InsightGPT AI")

menu = st.sidebar.radio(

    "Select Module",

    [

        "Dashboard",

        "Analytics",

        "Business Insights",

        "AI Chat",

        "Predictions",

        "Reports",

        "SQL Analytics"
    ]
)

# ==========================================
# TITLE
# ==========================================

st.title("🚀 InsightGPT AI")

st.subheader(
    "Enterprise AI Analytics Platform"
)

# ==========================================
# FILE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(

    "📂 Upload Dataset",

    type=["csv", "xlsx"]
)

# ==========================================
# LOAD DATA
# ==========================================

df = None

if uploaded_file is not None:

    try:

        if uploaded_file.name.endswith(".csv"):

            df = pd.read_csv(
                uploaded_file
            )

        else:

            df = pd.read_excel(
                uploaded_file
            )

    except Exception as e:

        st.error(
            f"Error loading dataset: {e}"
        )

# ==========================================
# DASHBOARD MODULE
# ==========================================

if menu == "Dashboard":

    st.header("📊 Dashboard")

    if df is not None:

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Rows",
                df.shape[0]
            )

        with col2:

            st.metric(
                "Columns",
                df.shape[1]
            )

        with col3:

            st.metric(
                "Missing Values",
                df.isnull().sum().sum()
            )

        with col4:

            st.metric(
                "Duplicate Rows",
                df.duplicated().sum()
            )

        st.subheader(
            "📄 Dataset Preview"
        )

        st.dataframe(
            df.head()
        )

        st.subheader(
            "📌 Dataset Information"
        )

        info_df = pd.DataFrame({

            "Column": df.columns,

            "Data Type": df.dtypes.astype(str),

            "Missing Values": df.isnull().sum().values
        })

        st.dataframe(info_df)

    else:

        st.info(
            "Upload dataset first."
        )

# ==========================================
# ANALYTICS MODULE
# ==========================================

elif menu == "Analytics":

    st.header("📈 Advanced Analytics")

    if df is not None:

        numeric_cols = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if len(numeric_cols) > 0:

            chart_type = st.selectbox(

                "Select Chart",

                [

                    "Histogram",

                    "Scatter Plot",

                    "Box Plot",

                    "Line Chart"
                ]
            )

            selected_col = st.selectbox(

                "Select Column",

                numeric_cols
            )

            if chart_type == "Histogram":

                fig = px.histogram(
                    df,
                    x=selected_col
                )

            elif chart_type == "Scatter Plot":

                y_col = st.selectbox(
                    "Select Y Axis",
                    numeric_cols
                )

                fig = px.scatter(
                    df,
                    x=selected_col,
                    y=y_col
                )

            elif chart_type == "Box Plot":

                fig = px.box(
                    df,
                    y=selected_col
                )

            else:

                fig = px.line(
                    df,
                    y=selected_col
                )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.subheader(
                "🔥 Correlation Heatmap"
            )

            corr = df[numeric_cols].corr()

            heatmap = px.imshow(

                corr,

                text_auto=True,

                aspect="auto"
            )

            st.plotly_chart(
                heatmap,
                use_container_width=True
            )

        else:

            st.warning(
                "No numeric columns found."
            )

    else:

        st.info(
            "Upload dataset first."
        )

# ==========================================
# BUSINESS INSIGHTS MODULE
# ==========================================

elif menu == "Business Insights":

    st.header("📊 Business Insights")

    if df is not None:

        numeric_cols = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        total_rows = df.shape[0]

        total_columns = df.shape[1]

        missing_values = df.isnull().sum().sum()

        duplicate_rows = df.duplicated().sum()

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Total Rows",
                total_rows
            )

        with col2:

            st.metric(
                "Total Columns",
                total_columns
            )

        with col3:

            st.metric(
                "Missing Values",
                missing_values
            )

        with col4:

            st.metric(
                "Duplicate Rows",
                duplicate_rows
            )

        st.subheader(
            "📈 Statistical Summary"
        )

        st.dataframe(
            df.describe()
        )

        st.subheader(
            "🔥 Smart Insights"
        )

        for col in numeric_cols:

            st.success(

                f"""

                📌 {col}

                Mean: {df[col].mean():.2f}

                Max: {df[col].max():.2f}

                Min: {df[col].min():.2f}

                Std Dev: {df[col].std():.2f}

                """
            )

        selected_col = st.selectbox(

            "Select Column for Distribution",

            numeric_cols
        )

        fig = px.histogram(

            df,

            x=selected_col,

            marginal="box"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        csv = df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(

            label="⬇ Download Dataset",

            data=csv,

            file_name="clean_dataset.csv",

            mime="text/csv"
        )

    else:

        st.info(
            "Upload dataset first."
        )

# ==========================================
# AI CHAT MODULE
# ==========================================

elif menu == "AI Chat":

    st.header("🤖 Gemini AI Assistant")

    user_question = st.text_area(
        "Ask AI anything"
    )

    if st.button("Generate AI Response"):

        if user_question:

            try:

                genai.configure(
                    api_key=api_key
                )

                model = genai.GenerativeModel(
                    "gemini-1.5-flash"
                )

                response = model.generate_content(
                    user_question
                )

                st.success(
                    "AI Response"
                )

                st.write(
                    response.text
                )

            except Exception as e:

                st.error(
                    f"Error: {e}"
                )

        else:

            st.warning(
                "Please enter a question."
            )
# ==========================================
# PREDICTION MODULE
# ==========================================

elif menu == "Predictions":

    st.header("📉 Machine Learning Predictions")

    if df is not None:

        numeric_cols = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if len(numeric_cols) >= 2:

            target_col = st.selectbox(

                "🎯 Select Target Column",

                numeric_cols
            )

            feature_cols = [

                col for col in numeric_cols

                if col != target_col
            ]

            X = df[feature_cols]

            y = df[target_col]

            X_train, X_test, y_train, y_test = train_test_split(

                X,
                y,

                test_size=0.2,

                random_state=42
            )

            model = LinearRegression()

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_test
            )

            mae = mean_absolute_error(
                y_test,
                predictions
            )

            r2 = r2_score(
                y_test,
                predictions
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "MAE",
                    round(mae, 2)
                )

            with col2:

                st.metric(
                    "R² Score",
                    round(r2, 2)
                )

            fig = go.Figure()

            fig.add_trace(

                go.Scatter(

                    y=y_test,

                    mode='lines',

                    name='Actual'
                )
            )

            fig.add_trace(

                go.Scatter(

                    y=predictions,

                    mode='lines',

                    name='Predicted'
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning(
                "Need minimum 2 numeric columns."
            )

    else:

        st.info(
            "Upload dataset first."
        )

# ==========================================
# REPORTS MODULE
# ==========================================

elif menu == "Reports":

    st.header("📑 Reports")

    if df is not None:

        st.subheader(
            "Dataset Summary"
        )

        st.write(
            df.describe()
        )

        st.download_button(

            label="⬇ Download CSV Report",

            data=df.to_csv(index=False),

            file_name="processed_dataset.csv",

            mime="text/csv"
        )

        st.success(
            "Report Generated Successfully!"
        )

    else:

        st.info(
            "Upload dataset first."
        )

# ==========================================
# SQL ANALYTICS MODULE
# ==========================================

elif menu == "SQL Analytics":

    st.header("🗄 SQL Analytics")

    host = st.text_input("Host")

    user = st.text_input("Username")

    password = st.text_input(

        "Password",

        type="password"
    )

    database = st.text_input("Database")

    query = st.text_area(
        "Enter SQL Query"
    )

    if st.button(
            "Connect Database"
    ):

        try:

            engine = create_engine(

                f"mysql+pymysql://{user}:{password}@{host}/{database}"
            )

            st.success(
                "Database Connected Successfully!"
            )

            if query:

                result = pd.read_sql(
                    query,
                    engine
                )

                st.subheader(
                    "📊 Query Result"
                )

                st.dataframe(
                    result
                )

        except Exception as e:

            st.error(
                f"Connection Error: {e}"
            )