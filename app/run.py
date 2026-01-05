import streamlit as st

from predict_rf import predict_difficulty
from predict_score import predict_score


# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="AutoJudge",
    layout="centered"
)

# -------------------------------------------------
# Custom CSS (Ocean Green Theme + Readable Text)
# -------------------------------------------------
st.markdown("""
<style>
body, .stApp {
    background-color: #e6f7f4;
    color: #0f2f2a;
}

div[data-testid="stVerticalBlock"] {
    border: 1px solid #bfe6dd;
    padding: 18px;
    border-radius: 10px;
    background-color: #ffffff;
    color: #0f2f2a;
}

h1, h2, h3, h4, h5, h6 {
    color: #0f2f2a !important;
}

label, p, span, div {
    color: #0f2f2a !important;
}

textarea {
    background-color: #ffffff !important;
    color: #0f2f2a !important;
    border-radius: 6px;
}

button[kind="primary"] {
    background-color: #1aa37a !important;
    color: white !important;
    border-radius: 8px;
    height: 3em;
    font-size: 16px;
}

button[kind="primary"]:hover {
    background-color: #148f69 !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Title & Overview
# -------------------------------------------------
st.title("AutoJudge")

st.markdown("""
**AutoJudge** estimates the difficulty of a programming problem using only its text.

### How to use:
1. Paste the **problem description** (required)
2. Paste **input** and **output descriptions** (recommended)
3. Click **Predict**

The system will return:
- A difficulty **class** (Easy / Medium / Hard)
- A numerical **difficulty score**
""")

# -------------------------------------------------
# Input Section
# -------------------------------------------------
st.markdown("### Problem Details")

description = st.text_area(
    "Problem Description *",
    height=220
)

input_description = st.text_area(
    "Input Description (optional)",
    height=120
)

output_description = st.text_area(
    "Output Description (optional)",
    height=120
)

# -------------------------------------------------
# Prediction state
# -------------------------------------------------
if "confirm_run" not in st.session_state:
    st.session_state.confirm_run = False

# -------------------------------------------------
# Predict button
# -------------------------------------------------
if st.button("Predict", type="primary"):

    st.session_state.confirm_run = False

    if not description.strip():
        st.error("Problem description is required to make a prediction.")

    elif not input_description.strip() or not output_description.strip():
        st.warning(
            "Input or Output description is missing.\n\n"
            "Do you want to continue anyway?"
        )
        st.session_state.confirm_run = True

    else:
        st.session_state.confirm_run = "run"

# -------------------------------------------------
# Confirmation button
# -------------------------------------------------
if st.session_state.confirm_run is True:
    if st.button("Confirm & Predict Anyway", type="primary"):
        st.session_state.confirm_run = "run"


# -------------------------------------------------
# Run prediction
# -------------------------------------------------
if st.session_state.confirm_run == "run":

    with st.spinner("Predicting difficulty..."):
        full_text = " ".join([
    description.strip(),
    input_description.strip(),
    output_description.strip()
])


        predicted_class = predict_difficulty(full_text)
        predicted_score = predict_score(full_text)

    st.markdown("### Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Difficulty Class",
            value=predicted_class.capitalize()
        )

    with col2:
        st.metric(
            label="Difficulty Score",
            value=f"{predicted_score:.2f}"
        )
