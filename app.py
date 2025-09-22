import streamlit as st
import yaml
import asyncio
from summary_validator import SummaryValidator
from DatabaseConn import DatabaseConn

# --- Fix asyncio loop issues (MacOS + Streamlit + torch) ---
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# --- Load config ---
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

api_url = config.get("api_url")
session_id = config.get("session_id", "summary_validation")

# --- Initialize validator (DB + config + API) ---
db_conn = DatabaseConn()
validator = SummaryValidator(db_conn=db_conn, config_path="config.yaml", api_url=api_url)

# --- Streamlit UI ---
st.set_page_config(page_title="Summary Validator", layout="wide")
st.title("📊 Summary Validator Dashboard")

# User prompt input
# User intent input
user_intent = st.text_input("Enter the user intent (natural language query):")
user_prompt = st.text_area("Enter the SQL/summary prompt to validate:", height=100)

if st.button("Run Validation"):
    if not user_intent.strip():
        st.warning("⚠️ Please enter a user intent before running validation.")
    elif not user_prompt.strip():
        st.warning("⚠️ Please enter a prompt before running validation.")
    else:
        try:
            result = validator.run(user_prompt=user_prompt, user_intent=user_intent)


            # --- Forward Validation ---
            st.markdown("## 🔎 Forward Validation")
            for key, value in result["Forward Validation"].items():
                with st.expander(f"➡️ {key}"):
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, (int, float)):
                                st.progress(min(max(sub_value, 0.0), 1.0))
                                st.write(f"{sub_key}: **{sub_value}**")
                            else:
                                st.write(f"{sub_key}: {sub_value}")
                    else:
                        st.write(value)

            # --- Answerability ---
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
        except Exception as e:
            st.error(f"Validation failed: {str(e)}")
