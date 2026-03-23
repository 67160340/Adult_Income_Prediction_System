
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

st.set_page_config(page_title="Income Predictor Pro", page_icon="💰", layout="centered")

@st.cache_resource
def load_assets():
    model = joblib.load('income_xgb_pipeline.pkl')
    with open('feature_names.json', 'r') as f:
        features = json.load(f)
    return model, features

model, model_features = load_assets()

st.title("💰 Income Prediction System")

with st.form("main_form"):
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("อายุ", 17, 90, 30)
        edu = st.slider("ระดับการศึกษา (ปี)", 1, 16, 10)
        gain = st.number_input("Capital Gain", 0)
        sex = st.selectbox("เพศ", ["Male", "Female"])
    with c2:
        hours = st.number_input("ชั่วโมงทำงาน/สัปดาห์", 1, 99, 40)
        work = st.selectbox("กลุ่มอาชีพ", ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Federal-gov'])
        marit = st.selectbox("สถานะการสมรส", ['Never-married', 'Married-civ-spouse', 'Divorced', 'Widowed'])
        occ = st.selectbox("สายงาน", ['Prof-specialty', 'Exec-managerial', 'Adm-clerical', 'Sales', 'Craft-repair'])
    
    btn = st.form_submit_button("วิเคราะห์ผลลัพธ์")

if btn:
    in_df = pd.DataFrame([{'age':age, 'education.num':edu, 'hours.per.week':hours, 'capital.gain':gain, 'workclass':work, 'marital.status':marit, 'sex':sex, 'occupation':occ}])
    in_enc = pd.get_dummies(in_df)
    final_in = pd.DataFrame(0, index=[0], columns=model_features)
    for c in in_enc.columns:
        if c in final_in.columns: final_in[c] = in_enc[c].iloc[0]
    
    prob = model.predict_proba(final_in)[0][1]
    st.divider()
    if prob > 0.5:
        st.success(f"### ผลการวิเคราะห์: รายได้ > $50K (โอกาส {prob*100:.1f}%)")
    else:
        st.warning(f"### ผลการวิเคราะห์: รายได้ <= $50K (โอกาส {(1-prob)*100:.1f}%)")
