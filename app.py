import streamlit as st
import torch
import pandas as pd
import plotly.express as px

from transformers import (
    BertTokenizer,
    BertForSequenceClassification
)

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="BERT Sentiment Analyzer",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        to right,
        #141e30,
        #243b55
    );
    color: white;
}

h1, h2, h3, h4 {
    color: white;
}

textarea {
    background-color: white !important;
    color: black !important;
    border-radius: 12px !important;
    font-size: 18px !important;
}

.stButton > button {
    width: 100%;
    height: 3em;
    border-radius: 12px;
    background-color: #4CAF50;
    color: white;
    font-size: 18px;
    border: none;
}

.result-card {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    margin-top: 20px;
}

.positive {
    background-color: rgba(0,255,0,0.2);
    border: 2px solid green;
}

.negative {
    background-color: rgba(255,0,0,0.2);
    border: 2px solid red;
}

.neutral {
    background-color: rgba(255,255,0,0.2);
    border: 2px solid orange;
}

.sidebar .sidebar-content {
    background-color: #111827;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
MODEL_PATH = "bert_sentiment_model"

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)

model = BertForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.eval()

# -----------------------------
# LABELS
# -----------------------------
labels = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:

    st.title("Navigation")

    st.markdown("---")

    st.subheader("About Model")

    st.write("""
    This application uses a fine-tuned BERT model
    for sentiment analysis.

    Model:
    - BERT Base Uncased
    - Transformer Architecture
    - Fine-tuned on sentiment dataset
    """)

    st.markdown("---")

    st.subheader("Example Inputs")

    if st.button("Positive Example"):
        st.session_state.example_text = (
            "This product is absolutely amazing."
        )

    if st.button("Negative Example"):
        st.session_state.example_text = (
            "Worst experience ever."
        )

    if st.button("Neutral Example"):
        st.session_state.example_text = (
            "The event was okay overall."
        )

# -----------------------------
# TITLE
# -----------------------------
st.title("BERT Sentiment Analyzer")

# -----------------------------
# SESSION STATE
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "example_text" not in st.session_state:
    st.session_state.example_text = ""

# -----------------------------
# INPUT BOX
# -----------------------------
text = st.text_area(
    "Enter text for sentiment analysis",
    value=st.session_state.example_text,
    height=250
)

# -----------------------------
# ANALYZE BUTTON
# -----------------------------
if st.button("Analyze Sentiment"):

    if text.strip() == "":

        st.warning("Please enter text.")

    else:

        with st.spinner("Analyzing sentiment..."):

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )

            with torch.no_grad():

                outputs = model(**inputs)

                probs = torch.softmax(
                    outputs.logits,
                    dim=1
                )[0]

                prediction = torch.argmax(probs).item()

            sentiment = labels[prediction]

            confidence = probs[prediction].item() * 100

            prob_values = probs.tolist()

        # -----------------------------
        # RESULT CARD
        # -----------------------------
        if sentiment == "positive":

            st.markdown(f"""
            <div class="result-card positive">
                POSITIVE
            </div>
            """, unsafe_allow_html=True)

        elif sentiment == "negative":

            st.markdown(f"""
            <div class="result-card negative">
                NEGATIVE
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown(f"""
            <div class="result-card neutral">
                NEUTRAL
            </div>
            """, unsafe_allow_html=True)

        # -----------------------------
        # CONFIDENCE BAR
        # -----------------------------
        st.subheader("Confidence Score")

        st.progress(int(confidence))

        st.write(f"{confidence:.2f}%")

        # -----------------------------
        # PIE CHART
        # -----------------------------
        st.subheader("Prediction Distribution")

        chart_df = pd.DataFrame({
            "Sentiment": [
                "Negative",
                "Neutral",
                "Positive"
            ],
            "Probability": prob_values
        })

        fig = px.pie(
            chart_df,
            names="Sentiment",
            values="Probability",
            hole=0.4
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # -----------------------------
        # SAVE HISTORY
        # -----------------------------
        st.session_state.history.append({
            "Text": text[:100],
            "Sentiment": sentiment,
            "Confidence": f"{confidence:.2f}%"
        })

# -----------------------------
# HISTORY SECTION
# -----------------------------
if len(st.session_state.history) > 0:

    st.subheader("Prediction History")

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True
    )