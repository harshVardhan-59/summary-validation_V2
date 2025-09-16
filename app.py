import streamlit as st
from summary_validator import SummaryValidator

# Page setup
st.set_page_config(page_title="Summary Validator", layout="wide")
st.title("📊 Summary Validator Dashboard")

# Run validation
validator = SummaryValidator()
result = validator.run()

# --- Ground Truth & Data Source ---
st.markdown("## 📝 Summary Info")
col1, col2 = st.columns(2)
with col1:
    st.metric("Data Source", result["Data Source"])
with col2:
    st.metric("Ground Truth", result["Ground Truth Summary"])

# --- Forward Validation ---
st.markdown("## 🔎 Forward Validation")
for key, value in result["Forward Validation"].items():
    with st.expander(f"➡️ {key}"):
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (int, float)):
                    st.progress(min(max(sub_value, 0.0), 1.0))  # normalize 0–1 scores
                    st.write(f"{sub_key}: **{sub_value}**")
                else:
                    st.write(f"{sub_key}: {sub_value}")
        else:
            st.write(value)

# --- Answerability in a nice table ---
if "Answerability" in result["Forward Validation"]:
    st.markdown("### ❓ Answerability Results")
    answerability = result["Forward Validation"]["Answerability"]
    st.table([
        {
            "Question": q["Question"],
            "Answer": q["Answer"],
            "Score": q["Score"],
        }
        for q in answerability["Questions"]
    ])

# --- Backward Validation ---
st.markdown("## 🔄 Backward Validation")
for key, value in result["Backward Validation"].items():
    with st.expander(f"➡️ {key}"):
        for sub_key, sub_value in value.items():
            st.write(f"{sub_key}: {sub_value}")
