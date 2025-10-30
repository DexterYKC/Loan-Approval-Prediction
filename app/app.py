import json, joblib, pandas as pd
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Loan Approval", page_icon="", layout="centered")

MODEL_PATH = Path("model.pkl")
SCHEMA_PATH = Path("features.json")
DATA_PATH = Path("../data/loan.csv")

assert MODEL_PATH.exists() and SCHEMA_PATH.exists(), "Lance python train.py"
model = joblib.load(MODEL_PATH)
with open(SCHEMA_PATH,"r") as f:
    meta = json.load(f)

num_cols = meta.get("num", [])
cat_cols = meta.get("cat", [])
all_cols = num_cols + cat_cols

defaults = {}
if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
    for c in num_cols:
        defaults[c] = float(pd.to_numeric(df[c], errors="coerce").median()) if c in df.columns else 0.0
    for c in cat_cols:
        if c in df.columns and df[c].dropna().shape[0] > 0:
            defaults[c] = str(df[c].dropna().astype(str).mode().iloc[0])
        else:
            defaults[c] = "Unknown"
else:
    for c in num_cols: defaults[c] = 0.0
    for c in cat_cols: defaults[c] = "Unknown"

st.markdown(
    """
    <style>
    .result-card {border:1px solid #e6e6e6; padding:1rem; border-radius:12px;}
    .ok {background:#ecfdf5; border:1px solid #34d399;}
    .ko {background:#fef2f2; border:1px solid #f87171;}
    .muted{color:#666;}
    </style>
    """,
    unsafe_allow_html=True,
)

title_col, meta_col = st.columns([0.7,0.3])
with title_col:
    st.title("Loan Approval Prediction")

expected_fields = set(["age","gender","occupation","education_level","marital_status","income","credit_score"])
smart_mode = expected_fields.issubset(set(all_cols))

tab1, tab2 = st.tabs(["formulaire", "Typing"])

if smart_mode:
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Age", min_value=18, max_value=80, value=int(defaults.get("age",30)))
            income = st.number_input("Income", min_value=0.0, value=float(defaults.get("income", 30000.0)), step=100.0)
            credit_score = st.slider("Credit score", min_value=300, max_value=900, value=int(defaults.get("credit_score", 650)))
        with c2:
            gender = st.selectbox("Gender", options=["Male","Female","Other"], index=0 if defaults.get("gender","Male") not in ["Female","Other"] else ["Male","Female","Other"].index(defaults.get("gender","Male")))
            occupation = st.text_input("Occupation", value=str(defaults.get("occupation","Employee")))
            education_level = st.selectbox("Education level", options=["High School","Bachelor","Master","PhD","Other"], index=0)
            marital_status = st.selectbox("Marital status", options=["Single","Married","Divorced","Widowed"], index=0)

        user_row = {
            "age": age,
            "gender": gender,
            "occupation": occupation,
            "education_level": education_level,
            "marital_status": marital_status,
            "income": income,
            "credit_score": credit_score
        }
        for c in all_cols:
            if c not in user_row:
                user_row[c] = defaults.get(c, "Unknown" if c in cat_cols else 0.0)

        if st.button("Predict", type="primary"):
            X = pd.DataFrame([user_row])
            proba = float(model.predict_proba(X)[0,1])
            pred = int(proba >= 0.5)

            st.progress(min(max(proba,0.0),1.0))
            st.markdown(
                f"""
                <div class="result-card {'ok' if pred==1 else 'ko'}">
                  <h3 style="margin:0; color: black;">{"✅ Approved" if pred==1 else "❌ Rejected"}</h3>
                  <p class="muted" style="margin:.25rem 0 0 0;">Probability of approval: <b>{proba:.2f}</b></p>
                </div>
                """,
                unsafe_allow_html=True
            )
else:
    st.info("Le dataset ne correspond pas utilise  « Typing ».")

with tab2:
    st.caption("")
    cols = st.columns(2) if len(all_cols) > 1 else [st]
    generic = {}
    for i, c in enumerate(all_cols):
        with cols[i % 2]:
            if c in num_cols:
                generic[c] = st.number_input(c, value=float(defaults.get(c,0.0)))
            else:
                generic[c] = st.text_input(c, value=str(defaults.get(c,"Unknown")))
    if st.button("Predict"):
        X = pd.DataFrame([generic])
        proba = float(model.predict_proba(X)[0,1])
        pred = int(proba >= 0.5)
        st.markdown(
            f"""
            <div class="result-card {'ok' if pred==1 else 'ko'}">
              <h3 style="margin:0;">{"✅ Approved" if pred==1 else "❌ Rejected"}</h3>
              <p class="muted" style="margin:.25rem 0 0 0;">Probability of approval: <b>{proba:.2f}</b></p>
            </div>
            """,
            unsafe_allow_html=True
        )

st.divider()

