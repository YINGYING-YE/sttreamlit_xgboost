import streamlit as st
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import pickle
import shap

from pathlib import Path
import pandas as pd
import numpy as np

st.set_page_config(    
    page_title="Prediction Model for EBV infection-associated phagocytic lymphohistiocytosis",
    page_icon="⭕",
    layout="wide"
)

app_dir = Path(__file__).parent

with open(app_dir / 'model.pkl', 'rb') as f:
    model = pickle.load(f)

st.markdown('''
    <h1 style="font-size: 20px; text-align: center; color: black; background: transparent; border-radius: .5rem; margin-bottom: 1rem;">
    Prediction Model for EBV infection-associated phagocytic lymphohistiocytosis
    </h1>''', unsafe_allow_html=True)

col = ["WBC", "PLT", "FER", "Ddimer", "TG", "CD3_CD4", "CLA", "fever days"]

with st.form("input"):
    c = st.columns(4)
    WBC = c[0].number_input("White Blood Cell Count(109/L)", value=6.37, step=0.01)
    PLT = c[1].number_input("platelet(109/L)", value=70, step=1)
    FER = c[2].number_input("ferritin(pmol/L)", value=1785.10, step=0.01)
    DDIMER = c[3].number_input("D-dimer(ug/L)", value=4150, step=1)
    TG = c[0].number_input("Triglyceride(mmol/L)", value=3.39, step=0.01)
    CD3_CD4 = c[1].number_input("CD3 positive CD4 positive T-cell percentage(%)", value=13.60, step=0.01)
    CLA = c[2].number_input("Cervical lymphadenopathy", value=1, step=1)
    FEVER_DAYS = c[3].number_input("Fever days(d)", value=14, step=1)
    st.markdown("---")
    c = st.columns(5)
    btn = c[2].form_submit_button("Predict", use_container_width=True)

if btn:
    data = pd.DataFrame([[WBC, PLT, FER, DDIMER, TG, CD3_CD4, CLA, FEVER_DAYS]], columns=col)
    res = model.predict(data)
    if res[0]==1:
        res = "hemophagocytic lymphohistiocytosis"
    else:
        res = "infectious mononucleosis"
    st.markdown(f'''
        <div style="text-align: center;"><span style="background: #FF4B4B; color: white; padding: 0.3rem; padding-left: 1rem; padding-right: 1rem;">预测结果</span></div>
        <div style="font-size: 20px; text-align: center; border-radius: .5rem; margin-bottom: 1rem; padding: 1rem; border: 1px solid red;">
        Predict result: <span style="color: red;">{res}</span>
        </div>''', unsafe_allow_html=True)
    explainer = shap.TreeExplainer(model) 
    shap_values = explainer.shap_values(data)
    shap_plot = shap.plots.force(explainer.expected_value, shap_values[0,:], data.iloc[0, :], show=False, matplotlib=True)
    c = st.columns([1, 5, 1])
    c[1].pyplot(plt.gcf(), use_container_width=True)
else:
    res = "Not input values to predict!"
    st.markdown(f'''
        <div style="text-align: center;"><span style="background: #FF4B4B; color: white; padding: 0.3rem; padding-left: 1rem; padding-right: 1rem;">预测结果</span></div>
        <div style="font-size: 20px; text-align: center; color: red; border-radius: .5rem; margin-bottom: 1rem; padding: 1rem; border: 1px solid red;">
        {res}
        </div>''', unsafe_allow_html=True)



