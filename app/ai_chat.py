import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# API Key
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Title
st.subheader("🤖 AI Analytics Assistant")

# User input
question = st.text_input(
    "Ask anything about your data"
)

if question:

    with st.spinner("Thinking..."):

        response = client.chat.completions.create(
            model="gpt-4o-mini",

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert "
                        "data analyst and "
                        "business intelligence AI."
                    )
                },

                {
                    "role": "user",
                    "content": question
                }
            ]
        )

        answer = response.choices[0].message.content

        st.success("AI Response")

        st.write(answer)