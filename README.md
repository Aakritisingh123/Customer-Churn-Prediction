# 📉 Customer Churn Prediction

A Machine Learning web application built with **Python, Scikit-learn, and Streamlit** that predicts whether a telecom customer is likely to churn based on customer demographics, account information, and service details.

🌐 **Live Demo:**  
https://aakritisingh123-customer-churn-prediction-app-dpbuqb.streamlit.app/

---

## 📌 Project Overview

Customer churn is one of the biggest challenges for telecom companies. This project uses a **Random Forest Classifier** trained on the **Telco Customer Churn Dataset** to predict whether a customer is likely to leave the service.

The application provides an easy-to-use web interface where users can enter customer information and receive an instant prediction.

---

## 🚀 Features

- Predict customer churn in real time
- Interactive web interface built using Streamlit
- Machine Learning model trained using Random Forest
- User-friendly input form
- Fast and lightweight deployment
- Publicly accessible web application

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Joblib

---

## 📂 Dataset

This project uses the **Telco Customer Churn Dataset**, which contains customer information such as:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Contract Type
- Internet Service
- Monthly Charges
- Total Charges
- Paperless Billing
- Payment Method
- Churn (Target Variable)

---

## 🤖 Machine Learning Model

**Algorithm Used:**

- Random Forest Classifier

### Workflow

1. Load Dataset
2. Data Cleaning
3. Handle Missing Values
4. Encode Categorical Variables
5. Feature Selection
6. Train Random Forest Model
7. Save Model using Joblib
8. Deploy using Streamlit

---

## 📁 Project Structure

```
Customer-Churn-Prediction/
│
├── app.py
├── customer_churn_model.pkl
├── requirements.txt
├── README.md
├── Telco-Customer-Churn.csv
└── notebooks/
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/your-username/customer-churn-prediction.git
```

Move into the project folder

```bash
cd customer-churn-prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 🌐 Live Application

👉 https://aakritisingh123-customer-churn-prediction-app-dpbuqb.streamlit.app/

---


## 🔮 Future Improvements

- Add prediction probability
- Feature importance visualization
- Explainable AI using SHAP
- Dark/Light mode
- Batch predictions using CSV upload
- Better UI/UX


---

## 👩‍💻 Author

**Aakriti Singh**

GitHub: https://github.com/aakritisingh123

LinkedIn: https://www.linkedin.com/in/aakriti-singh-38b7052a0/

---

## ⭐ If you like this project

Please consider giving it a ⭐ on GitHub!
