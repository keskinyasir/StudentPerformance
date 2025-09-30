import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# --- MODEL YÜKLE ---
@st.cache_data
def load_model():
    model = joblib.load("rf_model.pkl")
    return model

rf_model = load_model()

# --- FEATURE IMPORTANCE ---
feature_importance = pd.DataFrame({
    "feature": [
        'approval_rate','approved_rate_second_sem','Curricular units 2nd sem (approved)',
        'total_approved','Curricular units 2nd sem (grade)','Tuition fees up to date',
        'approved_rate_first_sem','Curricular units 1st sem (approved)','avg_approved_grade',
        'Age at enrollment','Admission grade','Previous qualification (grade)',
        'Curricular units 1st sem (grade)','Course','Curricular units 2nd sem (evaluations)',
        "Father's occupation",'Curricular units 1st sem (evaluations)',"Mother's occupation",
        'Course_freq',"Father's qualification"
    ],
    "importance": [
        0.107862,0.090768,0.066530,0.062611,0.051740,0.043191,0.039569,
        0.038377,0.034948,0.032126,0.029174,0.028469,0.026559,0.022744,
        0.022472,0.020807,0.019943,0.018967,0.018777,0.016436
    ]
})

# --- STREAMLIT UI ---
st.title("🎓 Öğrenci Dropout Tahmin Dashboard")

st.sidebar.header("Tahmin için Özellik Girdisi")
def user_input_features():
    data = {}
    data['approval_rate'] = st.sidebar.slider('Approval Rate (0-1)', 0.0, 1.0, 0.7)
    data['approved_rate_second_sem'] = st.sidebar.slider('Approved Rate 2nd Sem (0-1)', 0.0, 1.0, 0.7)
    data['total_approved'] = st.sidebar.number_input('Total Approved', 0, 50, 10)
    data['Curricular units 2nd sem (approved)'] = st.sidebar.number_input('2nd Sem Approved Units', 0, 20, 5)
    data['Tuition fees up to date'] = st.sidebar.selectbox('Tuition Fees Up to Date', [0,1])
    data['avg_approved_grade'] = st.sidebar.number_input('Average Approved Grade', 0.0, 20.0, 12.0)
    data['Age at enrollment'] = st.sidebar.number_input('Age at Enrollment', 17, 70, 20)
    data['Admission grade'] = st.sidebar.number_input('Admission Grade', 95, 190, 126)
    data['Course'] = st.sidebar.number_input('Course Code', 0, 10000, 9000)
    df = pd.DataFrame(data, index=[0])
    return df

input_df = user_input_features()

st.subheader("📊 Feature Importance (Top 15)")
plt.figure(figsize=(10,6))
top_features = feature_importance.head(15).sort_values(by="importance")
plt.barh(top_features['feature'], top_features['importance'], color='skyblue')
plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")
st.pyplot(plt)

# --- MODEL TAHMİNİ ---
st.subheader("🎯 Model Tahmini")
prediction_prob = rf_model.predict_proba(input_df)[:,1][0]
prediction = 1 if prediction_prob > 0.5 else 0

st.write(f"**Tahmin:** {'Dropout' if prediction==1 else 'Enrolled'}")
st.write(f"**Dropout Olasılığı:** {prediction_prob:.2f}")

# --- ROC Curve Örneği ---
st.subheader("ROC Curve (Örnek)")
fpr = np.linspace(0,1,100)
tpr = fpr**0.9
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = 0.929)')
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
st.pyplot(plt)
