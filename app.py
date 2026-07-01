"""
Streamlit app for the Telco Customer Churn model trained in
customer_churn_analysis.ipynb (RandomForestClassifier + SMOTE).

Run locally:
    streamlit run app.py

Deploy:
    Push this file, requirements.txt, and customer_churn_model.pkl to a GitHub repo,
    then deploy on https://share.streamlit.io (Streamlit Community Cloud).
"""

import pickle

import pandas as pd
import streamlit as st

MODEL_PATH = "customer_churn_model.pkl"

# Binary columns were encoded with sklearn's LabelEncoder, which sorts
# categories alphabetically -> "No"/"Female" = 0, "Yes"/"Male" = 1
BINARY_MAP_YES_NO = {"No": 0, "Yes": 1}
GENDER_MAP = {"Female": 0, "Male": 1}


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    return model_data["model"], model_data["features_name"]


def build_feature_row(raw: dict, feature_names: list) -> pd.DataFrame:
    """Recreate the same preprocessing used in training for a single record."""
    df = pd.DataFrame([raw])

    # Label-encoded binary columns
    df["gender"] = df["gender"].map(GENDER_MAP)
    for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        df[col] = df[col].map(BINARY_MAP_YES_NO)

    # One-hot encode the same columns as training (drop_first=True there;
    # here we just generate dummies and reindex to the training columns,
    # which achieves the same end result and is robust to which category
    # happens to be present in this single row).
    one_hot_cols = [
        "InternetService",
        "Contract",
        "PaymentMethod",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    df = pd.get_dummies(df, columns=one_hot_cols)

    # Align to the exact columns/order the model was trained on
    df = df.reindex(columns=feature_names, fill_value=0)
    df = df.astype(int)
    return df


def main():
    st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")
    st.title("📉 Customer Churn Predictor")
    st.write(
        "Enter a customer's details below to predict whether they are likely to churn, "
        "using the RandomForest model trained on the Telco Customer Churn dataset."
    )

    try:
        model, feature_names = load_model()
    except FileNotFoundError:
        st.error(
            f"Could not find `{MODEL_PATH}`. Run `train_model.py` first "
            "(with the Telco Customer Churn CSV in the same folder), then "
            "place the resulting `customer_churn_model.pkl` next to this app."
        )
        st.stop()

    with st.form("churn_form"):
        st.subheader("Demographics")
        col1, col2, col3 = st.columns(3)
        gender = col1.selectbox("Gender", ["Female", "Male"])
        senior_citizen = col2.selectbox("Senior Citizen", ["No", "Yes"])
        partner = col3.selectbox("Has Partner", ["No", "Yes"])
        dependents = col1.selectbox("Has Dependents", ["No", "Yes"])
        tenure = col2.number_input("Tenure (months)", min_value=0, max_value=100, value=12)

        st.subheader("Account Info")
        col4, col5, col6 = st.columns(3)
        contract = col4.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = col5.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = col6.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly_charges = col4.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=0.5)
        total_charges = col5.number_input("Total Charges ($)", min_value=0.0, value=840.0, step=1.0)

        st.subheader("Services")
        col7, col8, col9 = st.columns(3)
        phone_service = col7.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = col8.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = col9.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = col7.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = col8.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = col9.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = col7.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = col8.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = col9.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        submitted = st.form_submit_button("Predict Churn")

    if submitted:
        raw = {
            "gender": gender,
            "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        x = build_feature_row(raw, feature_names)
        prediction = model.predict(x)[0]
        probability = model.predict_proba(x)[0]

        st.divider()
        if prediction == 1:
            st.error(f"⚠️ This customer is **likely to churn** (probability: {probability[1]:.1%})")
        else:
            st.success(f"✅ This customer is **likely to stay** (probability of staying: {probability[0]:.1%})")

        st.progress(float(probability[1]))
        st.caption(f"Churn probability: {probability[1]:.1%} | Retention probability: {probability[0]:.1%}")

        with st.expander("See feature vector sent to the model"):
            st.dataframe(x.T.rename(columns={0: "value"}))


if __name__ == "__main__":
    main()