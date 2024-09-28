import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载模型
model = joblib.load('XGBoost.pkl')

# 定义自变量名称
feature_names = [
    "Gender", "Self perceived health status", "Arthritis", "Kidney disease", 
    "Digestive disease", "Life satisfaction", "Disability", "Falldown", 
    "Pain", "ADL score", "Sleep time"
]

# Streamlit用户界面
st.title("COPD Depression Risk Predictor")

# Gender: 分类选择
gender = st.selectbox("Gender (0=Female, 1=Male):", options=[0, 1], format_func=lambda x: 'Female (0)' if x == 0 else 'Male (1)')

# Self perceived health status: 分类选择
health_status = st.selectbox("Self perceived health status (0=Poor, 1=Average, 2=Good):", options=[0, 1, 2], format_func=lambda x: {0: 'Poor (0)', 1: 'Average (1)', 2: 'Good (2)'}[x])

# Arthritis: 分类选择
arthritis = st.selectbox("Arthritis (0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# Kidney disease: 分类选择
kidney_disease = st.selectbox("Kidney disease (0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# Digestive disease: 分类选择
digestive_disease = st.selectbox("Digestive disease (0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# Life satisfaction: 分类选择
life_satisfaction = st.selectbox("Life satisfaction (0=Not satisfied, 1=Average, 2=Satisfied):", options=[0, 1, 2], format_func=lambda x: {0: 'Not satisfied (0)', 1: 'Average (1)', 2: 'Satisfied (2)'}[x])

# Disability: 分类选择
disability = st.selectbox("Disability (0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# Falldown: 分类选择
falldown = st.selectbox("Falldown (0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# Pain: 分类选择
pain = st.selectbox("Pain (0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# ADL score: 数值输入，范围修改为 0-6
adl_score = st.number_input("ADL score:", min_value=0.0, max_value=6.0, value=3.0)

# Sleep time: 数值输入，范围为 0-24 小时
sleep_time = st.number_input("Sleep time (hours):", min_value=0.0, max_value=24.0, value=7.0)

# 处理输入并进行预测
feature_values = [gender, health_status, arthritis, kidney_disease, digestive_disease, life_satisfaction, disability, falldown, pain, adl_score, sleep_time]
features = np.array([feature_values])

if st.button("Predict"):
    # 预测类别和概率
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 显示预测结果
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # 根据预测结果生成建议
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        advice = (
            f"According to our model, you have a higher risk of health issues. "
            f"The model predicts that your probability is {probability:.1f}%. "
            "It is recommended to consult with your healthcare provider for further evaluation."
        )
    else:
        advice = (
            f"According to our model, you have a lower risk of health issues. "
            f"The model predicts that your probability is {probability:.1f}%. "
            "Maintaining a healthy lifestyle is still important."
        )

    st.write(advice)

    # 计算SHAP值并显示force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
