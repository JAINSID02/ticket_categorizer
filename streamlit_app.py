"""
Mini live demo for the Auto Email / Ticket Categorizer.

Run with:
    pip install streamlit
    streamlit run streamlit_app.py
"""

import pandas as pd
import streamlit as st

from ticket_categorizer import (
    DATA_PATH,
    REVIEW_THRESHOLD,
    load_data,
    train_final_model,
    predict_ticket,
)

st.set_page_config(page_title="Ticket Categorizer", page_icon="🎫", layout="centered")


@st.cache_resource
def get_model():
    df = load_data(DATA_PATH)
    # We pick Multinomial Naive Bayes here because ticket_categorizer.py's
    # own evaluation (run `python ticket_categorizer.py`) selects it as the
    # stronger performer on this dataset -- see README for the comparison.
    vectorizer, model = train_final_model(df, "Multinomial Naive Bayes")
    return vectorizer, model


st.title("🎫 Auto Ticket Categorizer")
st.caption("TF-IDF + Naive Bayes triage layer — Billing · Technical · HR · General")

vectorizer, model = get_model()

with st.form("ticket_form"):
    subject = st.text_input("Subject", placeholder="e.g. App is completely broken")
    body = st.text_area(
        "Body",
        placeholder="e.g. The app crashes every time I try to open the dashboard.",
        height=120,
    )
    submitted = st.form_submit_button("Categorize ticket")

if submitted:
    if not subject.strip() and not body.strip():
        st.warning("Please enter a subject or body for the ticket.")
    else:
        result = predict_ticket(subject, body, vectorizer, model)

        if result["routed_to"] == "NEEDS_HUMAN_REVIEW":
            st.error(
                f"⚠️ Low confidence ({result['confidence']:.0%}) — "
                f"flagged for **manual review** instead of auto-assigning."
            )
        else:
            st.success(
                f"✅ Routed to **{result['routed_to']}** "
                f"({result['confidence']:.0%} confidence)"
            )

        col1, col2 = st.columns(2)
        col1.metric("Predicted category", result["predicted_category"])
        col2.metric(
            "Priority",
            result["priority"],
            delta="needs attention" if result["priority"] == "URGENT" else None,
            delta_color="inverse",
        )

        st.subheader("Confidence breakdown")
        scores_df = pd.DataFrame(
            {"category": list(result["all_scores"].keys()),
             "confidence": list(result["all_scores"].values())}
        ).sort_values("confidence", ascending=False)
        st.bar_chart(scores_df.set_index("category"))

        st.caption(
            f"Tickets below {REVIEW_THRESHOLD:.0%} confidence are routed to a "
            "manual-review queue instead of being auto-assigned."
        )

st.divider()
st.caption(
    "This demo trains on a small labeled sample dataset (data/tickets.csv). "
    "In production, this would sit behind the live ticket queue and re-train "
    "periodically as new labeled tickets come in."
)