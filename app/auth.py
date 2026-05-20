import streamlit as st


def login():

    st.title("🔐 Login")

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
                "Login successful!"
            )

            st.rerun()

        else:

            st.error(
                "Invalid credentials"
            )


def logout():

    if st.sidebar.button(
            "🚪 Logout"
    ):

        st.session_state[
            "authenticated"
        ] = False

        st.rerun()
