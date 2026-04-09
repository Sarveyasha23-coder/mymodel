import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Bank Churn Predictor", page_icon="🏦", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'DM Serif Display', serif; }
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
    border-radius: 16px; padding: 2.5rem 2rem 2rem;
    margin-bottom: 2rem; color: white; text-align: center;
}
.hero h1 { color: #e8d5b7; font-size: 2.2rem; margin-bottom: 0.3rem; }
.hero p  { color: #a8b8c8; font-size: 1rem; margin: 0; }
.result-stay {
    background: linear-gradient(135deg, #0d6e3f, #1a9e5c);
    color: white; border-radius: 12px; padding: 1.5rem;
    text-align: center; font-size: 1.3rem; font-weight: 600; margin-top: 1.5rem;
}
.result-churn {
    background: linear-gradient(135deg, #c0392b, #e74c3c);
    color: white; border-radius: 12px; padding: 1.5rem;
    text-align: center; font-size: 1.3rem; font-weight: 600; margin-top: 1.5rem;
}
.stButton>button {
    background: #0f3460; color: white; border: none;
    border-radius: 8px; padding: 0.65rem 2.5rem;
    font-size: 1rem; font-weight: 600; width: 100%;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artefacts():
    model   = joblib.load("mlp_model.pkl")
    scaler  = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
    return model, scaler, columns

try:
    model, scaler, feature_cols = load_artefacts()
    artefacts_ok = True
except Exception as e:
    artefacts_ok = False
    load_error   = str(e)

st.markdown("""
<div class="hero">
  <h1>🏦 Bank Churn Predictor</h1>
  <p>Enter customer details below to predict whether they are likely to leave the bank.</p>
</div>
""", unsafe_allow_html=True)

if not artefacts_ok:
    st.error(f"Could not load model files. Make sure mlp_model.pkl, scaler.pkl, and columns.pkl are in the repo.\n\n{load_error}")
    st.stop()

st.subheader("Customer Information")
col1, col2 = st.columns(2)

with col1:
    credit_score     = st.slider("Credit Score",          300, 850, 650)
    age              = st.slider("Age",                    18,  92,  38)
    tenure           = st.slider("Tenure (years)",          0,  10,   5)
    num_products     = st.selectbox("Number of Products", [1, 2, 3, 4])
    has_cr_card      = st.selectbox("Has Credit Card",    ["Yes", "No"])

with col2:
    balance          = st.number_input("Account Balance (EUR)",   0.0, 300000.0, 60000.0, 500.0)
    estimated_salary = st.number_input("Estimated Salary (EUR)",  0.0, 250000.0, 70000.0, 500.0)
    is_active_member = st.selectbox("Active Member",              ["Yes", "No"])
    geography        = st.selectbox("Geography",                  ["France", "Germany", "Spain"])
    gender           = st.selectbox("Gender",                     ["Male", "Female"])

if st.button("Predict Churn"):
    raw = {
        "CreditScore":       credit_score,
        "Age":               age,
        "Tenure":            tenure,
        "Balance":           np.log1p(balance),
        "NumOfProducts":     num_products,
        "HasCrCard":         1 if has_cr_card      == "Yes" else 0,
        "IsActiveMember":    1 if is_active_member == "Yes" else 0,
        "EstimatedSalary":   np.log1p(estimated_salary),
        "Geography_Germany": 1 if geography == "Germany" else 0,
        "Geography_Spain":   1 if geography == "Spain"   else 0,
        "Gender_Male":       1 if gender    == "Male"    else 0,
    }
    input_vec    = np.array([[raw.get(c, 0) for c in feature_cols]])
    input_scaled = scaler.transform(input_vec)
    prob_churn   = float(model.predict_proba(input_scaled)[0][1])
    prob_stay    = 1 - prob_churn

    st.markdown("---")
    st.subheader("Prediction Result")
    c1, c2 = st.columns(2)
    c1.metric("Churn Probability",     f"{prob_churn*100:.1f}%")
    c2.metric("Retention Probability", f"{prob_stay*100:.1f}%")
    st.progress(prob_churn, text="Churn likelihood")

    if prob_churn >= 0.5:
        st.markdown(f'<div class="result-churn">⚠️ HIGH CHURN RISK — Likely to leave.<br><small>Confidence: {prob_churn*100:.1f}%</small></div>', unsafe_allow_html=True)
        st.warning("Recommended: Offer a retention deal, dedicated support, or loyalty rewards.")
    else:
        st.markdown(f'<div class="result-stay">✅ LOW CHURN RISK — Likely to stay.<br><small>Confidence: {prob_stay*100:.1f}%</small></div>', unsafe_allow_html=True)
        st.success("Recommended: Continue current engagement strategy.")

st.markdown("---")
st.caption("Model: Sklearn MLP Classifier · Dataset: Bank Customer Churn (Kaggle)")
