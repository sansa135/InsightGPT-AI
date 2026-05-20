import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from transformers import pipeline

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import (
    getSampleStyleSheet
)

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(

    page_title="InsightGPT AI",

    page_icon="🚀",

    layout="wide"
)

# =====================================
# SESSION STATE
# =====================================

if "authenticated" not in st.session_state:

    st.session_state["authenticated"] = False

# =====================================
# LOGIN FUNCTION
# =====================================

def login():

    st.title("🔐 Login Page")

    username = st.text_input(
        "Username"
    )

    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button("Login"):

        if (
                username == "admin"
                and
                password == "admin123"
        ):

            st.session_state[
                "authenticated"
            ] = True

            st.success(
                "Login Successful!"
            )

            st.rerun()

        else:

            st.error(
                "Invalid Credentials"
            )

# =====================================
# LOGIN CHECK
# =====================================

if not st.session_state["authenticated"]:

    login()

    st.stop()

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("📊 Navigation")

# Logout
if st.sidebar.button("🚪 Logout"):

    st.session_state[
        "authenticated"
    ] = False

    st.rerun()

# =====================================
# THEME
# =====================================

theme = st.sidebar.selectbox(

    "🎨 Select Theme",

    [

        "Light",

        "Dark"
    ]
)

# =====================================
# DARK MODE
# =====================================

if theme == "Dark":

    st.markdown(

        """

        <style>

        .stApp {

            background-color: #0E1117;

            color: white;
        }

        section[data-testid="stSidebar"] {

            background-color: #161B22;
        }

        h1, h2, h3, h4, h5, h6 {

            color: white !important;
        }

        p, label, div {

            color: white !important;
        }

        </style>

        """,

        unsafe_allow_html=True
    )

# =====================================
# MENU
# =====================================

menu = st.sidebar.radio(

    "Select Module",

    [

        "Dashboard",

        "Analytics",

        "AI Chat",

        "Predictions",

        "Reports"
    ]
)

# =====================================
# TITLE
# =====================================

st.title("🚀 InsightGPT AI")

st.subheader(
    "Enterprise GenAI Analytics Platform"
)

# =====================================
# FILE UPLOAD
# =====================================

uploaded_file = st.file_uploader(

    "📂 Upload Dataset",

    type=[

        "csv",

        "xlsx"
    ]
)

# =====================================
# DATA CLEANING
# =====================================

def clean_data(df):

    df.drop_duplicates(
        inplace=True
    )

    numeric_cols = df.select_dtypes(
        include='number'
    ).columns

    for col in numeric_cols:

        df[col].fillna(

            df[col].mean(),

            inplace=True
        )

    object_cols = df.select_dtypes(
        include='object'
    ).columns

    for col in object_cols:

        df[col].fillna(

            "Unknown",

            inplace=True
        )

    return df

# =====================================
# LOAD DATA
# =====================================

df = None

if uploaded_file:

    try:

        if uploaded_file.name.endswith(".csv"):

            df = pd.read_csv(
                uploaded_file
            )

        else:

            df = pd.read_excel(
                uploaded_file
            )

        df = clean_data(df)

    except Exception as e:

        st.error(
            f"Error loading file: {e}"
        )

# =====================================
# AI MODEL
# =====================================

@st.cache_resource
def load_model():

    model = pipeline(

        "text-generation",

        model="distilgpt2"
    )

    return model

chatbot = load_model()

# =====================================
# PDF REPORT
# =====================================

def generate_pdf(df):

    pdf_file = "AI_Report.pdf"

    doc = SimpleDocTemplate(
        pdf_file
    )

    styles = getSampleStyleSheet()

    elements = []

    title = Paragraph(

        "InsightGPT AI Report",

        styles['Title']
    )

    elements.append(title)

    elements.append(
        Spacer(1, 20)
    )

    rows = df.shape[0]

    cols = df.shape[1]

    info = Paragraph(

        f"""
        Total Rows: {rows}
        <br/>
        Total Columns: {cols}
        """,

        styles['BodyText']
    )

    elements.append(info)

    elements.append(
        Spacer(1, 20)
    )

    summary = df.describe().to_string()

    summary_text = Paragraph(

        f"""
        Statistical Summary:
        <br/><br/>
        {summary}
        """,

        styles['BodyText']
    )

    elements.append(summary_text)

    doc.build(elements)

    return pdf_file

# =====================================
# DASHBOARD MODULE
# =====================================

if menu == "Dashboard":

    st.header("📊 Dashboard")

    if df is not None:

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

        st.subheader(
            "📄 Dataset Preview"
        )

        st.dataframe(
            df.head()
        )

    else:

        st.info(
            "Upload dataset first."
        )

# =====================================
# ANALYTICS MODULE
# =====================================

elif menu == "Analytics":

    st.header("📈 Analytics")

    if df is not None:

        numeric_cols = df.select_dtypes(
            include='number'
        ).columns

        if len(numeric_cols) > 0:

            selected_col = st.selectbox(

                "Select Column",

                numeric_cols
            )

            chart_type = st.selectbox(

                "Select Chart",

                [

                    "Histogram",

                    "Box Plot",

                    "Line Chart"
                ]
            )

            if chart_type == "Histogram":

                fig = px.histogram(

                    df,

                    x=selected_col
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
                "🔥 Correlation Matrix"
            )

            corr = df[
                numeric_cols
            ].corr()

            heatmap = px.imshow(

                corr,

                text_auto=True
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

        st.warning(
            "Upload dataset first."
        )

# =====================================
# AI CHAT MODULE
# =====================================

elif menu == "AI Chat":

    st.header(
        "🤖 AI Analytics Assistant"
    )

    question = st.text_input(
        "Ask business questions"
    )

    if question:

        with st.spinner(
                "Thinking..."
        ):

            response = chatbot(

                question,

                max_length=120,

                num_return_sequences=1
            )

            answer = response[0][
                "generated_text"
            ]

            st.success(
                "AI Response"
            )

            st.write(answer)

# =====================================
# PREDICTION MODULE
# =====================================

elif menu == "Predictions":

    st.header(
        "📈 AI Predictions"
    )

    if df is not None:

        numeric_cols = df.select_dtypes(
            include='number'
        ).columns

        if len(numeric_cols) >= 2:

            target_col = st.selectbox(

                "🎯 Target Column",

                numeric_cols
            )

            feature_col = st.selectbox(

                "📊 Feature Column",

                numeric_cols
            )

            if target_col != feature_col:

                X = df[[feature_col]]

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

                error = mean_absolute_error(

                    y_test,

                    predictions
                )

                st.success(

                    f"Model trained successfully! MAE: {error:.2f}"
                )

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

        else:

            st.warning(
                "Need at least 2 numeric columns."
            )

    else:

        st.warning(
            "Upload dataset first."
        )

# =====================================
# REPORT MODULE
# =====================================

elif menu == "Reports":

    st.header(
        "📑 AI Reports"
    )

    if df is not None:

        if st.button(
                "📄 Generate PDF Report"
        ):

            pdf_path = generate_pdf(df)

            with open(
                    pdf_path,
                    "rb"
            ) as file:

                st.download_button(

                    label="⬇ Download Report",

                    data=file,

                    file_name="AI_Report.pdf",

                    mime="application/pdf"
                )

    else:

        st.warning(
            "Upload dataset first."
        )