import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from io import StringIO
from datetime import datetime

# -------------------------
# Load trained model
# -------------------------
with open("model (5).pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Patient Health Risk Prediction", layout="wide")

# -------------------------
# Constants / Helpers
# -------------------------
# Core features used during training
FEATURES = [
    "Pregnancies","Glucose","BloodPressure","SkinThickness",
    "Insulin","BMI","DiabetesPedigreeFunction","Age"
]

def expected_feature_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder/drop columns to match model's expected features.
    Tries model.feature_names_in_ first; falls back to FEATURES.
    """
    expected = getattr(model, "feature_names_in_", None)
    if expected is None:
        expected = FEATURES

    missing = [c for c in expected if c not in df.columns]
    extra = [c for c in df.columns if c not in expected]

    if missing:
        st.error(
            "Missing required columns: " + ", ".join(missing) +
            "\nPlease upload a CSV with exactly these columns: " + ", ".join(expected)
        )
        st.stop()

    if extra:
        st.info("Dropping extra columns not used by the model: " + ", ".join(extra))
        df = df.drop(columns=extra)

    return df[expected]

def clean_impossible_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """Replace zeros in physiologically impossible columns with each column median (same logic as training)."""
    cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in cols_with_zero:
        if col in df.columns:
            med = df[col].replace(0, np.nan).median()
            df[col] = df[col].replace(0, med)
    return df

# Normal ranges (approx) to flag inputs — purely informational
NORMALS = {
    "Glucose": (70, 140),
    "BloodPressure": (60, 120),
    "BMI": (18.5, 24.9),
    "SkinThickness": (10, 50),  # dataset-specific proxy
    "Insulin": (16, 166),       # very rough ref, varies by lab/context
    "Age": (18, 100),
    "Pregnancies": (0, 10),
    "DiabetesPedigreeFunction": (0.0, 1.0)
}

def flag_value(name, val):
    lo, hi = NORMALS.get(name, (None, None))
    if lo is None:
        return None
    if val < lo:
        return f"Low ({val}) – expected ≥ {lo}"
    if val > hi:
        return f"High ({val}) – expected ≤ {hi}"
    return f"OK ({val})"

def set_patient(values: dict):
    """Programmatically set sidebar slider values via session_state."""
    for k, v in values.items():
        st.session_state[k] = v

def random_patient():
    """Generate a plausible random patient profile."""
    rng = np.random.default_rng()
    return {
        "Pregnancies": int(rng.integers(0, 12)),
        "Glucose": int(rng.integers(70, 200)),
        "BloodPressure": int(rng.integers(60, 122)),
        "SkinThickness": int(rng.integers(10, 60)),
        "Insulin": int(rng.integers(20, 300)),
        "BMI": float(np.round(rng.uniform(18.0, 40.0), 1)),
        "DiabetesPedigreeFunction": float(np.round(rng.uniform(0.1, 2.0), 2)),
        "Age": int(rng.integers(18, 85)),
    }

# -------------------------
# Sidebar Navigation
# -------------------------
st.sidebar.title("📌 Navigation")
app_mode = st.sidebar.radio("Go to", [
    "Single Patient Analysis", 
    "Batch Testing", 
    
    "Patient Trends",
    "Compare Patients",
    "Reports & Summaries",
    "About"
])

# -------------------------
# Sidebar Patient Input
# -------------------------
st.sidebar.header("Enter Patient Details")
# Use keys so we can programmatically update values
Pregnancies = st.sidebar.slider("Pregnancies", 0, 20, 1, key="Pregnancies")
Glucose = st.sidebar.slider("Glucose Level", 0, 200, 100, key="Glucose")
BloodPressure = st.sidebar.slider("Blood Pressure", 0, 122, 70, key="BloodPressure")
SkinThickness = st.sidebar.slider("Skin Thickness", 0, 100, 20, key="SkinThickness")
Insulin = st.sidebar.slider("Insulin Level", 0, 900, 80, key="Insulin")
BMI = st.sidebar.slider("BMI", 0.0, 70.0, 25.0, key="BMI")
DiabetesPedigreeFunction = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5, step=0.01, key="DiabetesPedigreeFunction")
Age = st.sidebar.slider("Age", 18, 100, 30, key="Age")

def current_patient_df() -> pd.DataFrame:
    data = {
        "Pregnancies": st.session_state.Pregnancies,
        "Glucose": st.session_state.Glucose,
        "BloodPressure": st.session_state.BloodPressure,
        "SkinThickness": st.session_state.SkinThickness,
        "Insulin": st.session_state.Insulin,
        "BMI": st.session_state.BMI,
        "DiabetesPedigreeFunction": st.session_state.DiabetesPedigreeFunction,
        "Age": st.session_state.Age
    }
    df = pd.DataFrame(data, index=[0])
    return df

# Sidebar quick actions
col_a, col_b = st.sidebar.columns(2)
if col_a.button("🎲 Random"):
    set_patient(random_patient())
if col_b.button("♻️ Reset"):
    set_patient({
        "Pregnancies": 1, "Glucose": 100, "BloodPressure": 70, "SkinThickness": 20,
        "Insulin": 80, "BMI": 25.0, "DiabetesPedigreeFunction": 0.5, "Age": 30
    })

# -------------------------
# Single Patient Analysis
# -------------------------
if app_mode == "Single Patient Analysis":
    st.title("🩺 Patient Health Risk Prediction")
    st.write("Predict the likelihood of **Diabetes** based on patient health metrics.")

    df = current_patient_df()
    df = clean_impossible_zeros(df)      # keep in sync with training
    df_ordered = expected_feature_order(df)

    st.subheader("📋 Patient Data")
    st.write(df_ordered)

    # Quick input flags
    st.subheader("🧪 Input Quality Checks")
    cols = st.columns(4)
    checks = []
    for i, feat in enumerate(FEATURES):
        msg = flag_value(feat, float(df_ordered.iloc[0][feat]))
        checks.append((feat, msg))
    for i, (feat, msg) in enumerate(checks):
        with cols[i % 4]:
            if msg.startswith("OK"):
                st.success(f"{feat}: {msg}")
            elif "Low" in msg:
                st.warning(f"{feat}: {msg}")
            else:
                st.error(f"{feat}: {msg}")

    # Prediction at default 0.5 threshold
    proba = model.predict_proba(df_ordered)[0][1]
    pred = int(proba >= 0.5)

    st.subheader("🔍 Prediction Result (threshold 0.5)")
    st.write("**Diabetes Risk:**", "✅ Yes" if pred == 1 else "❌ No")

    st.subheader("📊 Risk Probability")
    st.write(f"Chance of Diabetes: **{proba*100:.2f}%**")
    fig, ax = plt.subplots()
    ax.bar(["No Diabetes", "Diabetes"], [1-proba, proba])
    ax.set_ylabel("Probability")
    st.pyplot(fig)

    # Medical Advice (basic rule-based)
    st.subheader("💡 Medical Suggestion")
    if pred == 1 or proba >= 0.6:
        st.error("⚠️ High diabetes risk. Recommend clinical follow-up, HbA1c test, and lifestyle modification.")
    else:
        st.success("🟢 Low risk based on current inputs. Keep monitoring routinely.")

    # Download Report
    st.subheader("📥 Download Patient Report")
    report = df_ordered.copy()
    report["PredictedLabel"] = "Diabetes" if pred == 1 else "No Diabetes"
    report["RiskProbability(%)"] = round(proba*100, 2)
    report["GeneratedAt"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.download_button(
        "Download Report (CSV)",
        report.to_csv(index=False),
        file_name="patient_report.csv",
        mime="text/csv"
    )

# -------------------------
# Batch Testing
# -------------------------
elif app_mode == "Batch Testing":
    st.title("📂 Batch Testing for Multiple Patients")
    uploaded_file = st.file_uploader("Upload Patient Data CSV", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Drop target if present & clean zeros
        if "Outcome" in data.columns:
            data = data.drop(columns=["Outcome"])
        data = clean_impossible_zeros(data)
        data = expected_feature_order(data)

        st.subheader("📋 Uploaded Data (first 5)")
        st.write(data.head())

        probabilities = model.predict_proba(data)[:, 1]
        threshold = st.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.01)
        predictions = (probabilities >= threshold).astype(int)

        results = data.copy()
        results["Prob_Diabetes"] = np.round(probabilities, 4)
        results["Prediction"] = np.where(predictions == 1, "Diabetes", "No Diabetes")

        st.subheader("🔍 Prediction Results")
        st.write(results)

        # Summary chart
        st.subheader("📊 Summary")
        counts = pd.Series(results["Prediction"]).value_counts()
        fig, ax = plt.subplots()
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        # Download
        st.download_button(
            "Download Full Results (CSV)",
            results.to_csv(index=False),
            file_name="batch_results.csv",
            mime="text/csv"
        )
    else:
        st.info("Upload a CSV with columns: " + ", ".join(FEATURES))


# -------------------------
# Patient Trends
# -------------------------
elif app_mode == "Patient Trends":
    st.title("📈 Patient Health Trends")
    uploaded_trend = st.file_uploader("Upload Patient Historical Data (CSV)", type=["csv"])

    if uploaded_trend:
        trend_data = pd.read_csv(uploaded_trend)
        st.write("📋 Patient Historical Data", trend_data.head())

        # Select metric (skip first column assuming it's time/index)
        metric = st.selectbox("Select Metric to Visualize", trend_data.columns[1:])

        # Detect first column (time / record index)
        x_col = trend_data.columns[0]

        import altair as alt

        # Build bar chart
        chart = alt.Chart(trend_data).mark_bar().encode(
            x=alt.X(x_col, sort=None, title=x_col),   # X-axis = first column (time/index)
            y=alt.Y(metric, title=metric),            # Y-axis = selected metric
            color=alt.Color(metric, scale=alt.Scale(scheme='tealblues')), # Colored bars
            tooltip=[x_col, metric]                   # Hover tooltip
        ).properties(
            width=700,
            height=400,
            title=f"{metric} Trends"
        )

        st.altair_chart(chart, use_container_width=True)

        # ✅ Custom message after chart
        st.success(f"✅ Visualization complete! The chart above shows trends in **{metric}** over time. Use it to spot patterns and take informed health decisions.")


# -------------------------
# Compare Patients
# -------------------------
elif app_mode == "Compare Patients":
    st.title("⚖️ Compare Patients")
    uploaded_compare = st.file_uploader("Upload 2 Patients Data (CSV with 2 rows)", type=["csv"])

    if uploaded_compare:
        comp_data = pd.read_csv(uploaded_compare)

        # ✅ Check that exactly 2 patients are uploaded
        if comp_data.shape[0] == 2:
            st.write("👥 Patients Data", comp_data)

            # ✅ Only keep features used in training
            feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure',
                            'SkinThickness', 'Insulin', 'BMI',
                            'DiabetesPedigreeFunction', 'Age']
            comp_features = comp_data[feature_cols]

            # ✅ Make predictions
            comp_pred = model.predict(comp_features)
            comp_prob = model.predict_proba(comp_features)[:, 1]

            # ✅ Add results back to original dataframe
            comp_data["Prediction"] = ["Diabetes" if p == 1 else "No Diabetes" for p in comp_pred]
            comp_data["Risk Probability (%)"] = (comp_prob * 100).round(2)

            st.subheader("🔍 Comparison Results")
            st.write(comp_data)

            # 🔹 Optional: Highlight higher risk patient
            higher_risk_idx = comp_data["Risk Probability (%)"].idxmax()
            st.success(f"📊 Patient {higher_risk_idx+1} has the higher diabetes risk.")
        else:
            st.error("Please upload exactly 2 patient records.")


# Reports & Summaries
# -------------------------
elif app_mode == "Reports & Summaries":
    st.title("🧾 Patient Reports & Summaries")
    uploaded_summary = st.file_uploader("Upload Patient Data CSV", type=["csv"])

    if uploaded_summary:
        summary_data = pd.read_csv(uploaded_summary)

        # ✅ Only keep training features
        feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure',
                        'SkinThickness', 'Insulin', 'BMI',
                        'DiabetesPedigreeFunction', 'Age']
        summary_features = summary_data[feature_cols]

        # ✅ Predictions & probabilities
        predictions = model.predict(summary_features)
        probabilities = model.predict_proba(summary_features)[:, 1]

        # ✅ Append results
        summary_data["Prediction"] = ["Diabetes" if p == 1 else "No Diabetes" for p in predictions]
        summary_data["Risk Probability (%)"] = (probabilities * 100).round(2)

        st.subheader("📋 Summary of Patients")
        st.write(summary_data)

        # ✅ Highlight high-risk patients
        high_risk = summary_data[summary_data["Risk Probability (%)"] > 70]
        if not high_risk.empty:
            st.subheader("⚠️ High Risk Patients")
            st.write(high_risk)
        else:
            st.info("✅ No patients with risk probability above 70%.")

        # ✅ Download button
        st.download_button(
            "Download Summary Report",
            summary_data.to_csv(index=False),
            "summary_report.csv",
            "text/csv"
        )

# -------------------------

# -------------------------
# About Page
# -------------------------
elif app_mode == "About":
    st.title("ℹ️ About This App")
    st.markdown("""
    ### 🩺 Patient Health Risk Prediction Dashboard
    **What’s inside:**
    - Single patient scoring with input quality checks
    - Batch testing with automatic column fixes
    - Adjustable decision threshold
    - Presets and CSV template download

    **Note:** This tool provides risk estimation for educational use. It is **not** a medical device.
    """)

