import streamlit as st
import pandas as pd
import joblib

# 1. โหลดโมเดล (ใช้ Cache เพื่อความรวดเร็ว)
@st.cache_resource 
def load_model():
    return joblib.load('income_model.pkl')

model = load_model()

# 2. ส่วนหัวของเว็บ
st.title("💰 โปรแกรมทำนายระดับรายได้")
st.markdown("กรอกข้อมูลส่วนตัวด้านล่าง เพื่อทำนายว่ารายได้ของคุณจะ **มากกว่า 50,000 เหรียญต่อปี** หรือไม่?")
st.divider()

# 3. จัด Layout เป็น 2 คอลัมน์ (ซ้าย-ขวา)
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("อายุ (Age)", min_value=17, max_value=90, value=30)
    education_num = st.number_input("ระดับการศึกษา (1-16)", min_value=1, max_value=16, value=10)
    capital_gain = st.number_input("กำไรจากการลงทุน (Capital Gain)", min_value=0, value=0)

with col2:
    hours_per_week = st.number_input("ชั่วโมงทำงานต่อสัปดาห์", min_value=1, max_value=99, value=40)
    is_married = st.selectbox("สถานะครอบครัว", ["โสด / อื่นๆ", "แต่งงาน (Married-civ-spouse)"])
    is_exec = st.selectbox("อาชีพ", ["พนักงานทั่วไป / อื่นๆ", "ผู้บริหาร (Exec-managerial)"])

st.divider()

# แปลงค่าจาก Dropdown ให้เป็นตัวเลข 0 หรือ 1 ตามที่โมเดลเข้าใจ
married_val = 1 if is_married == "แต่งงาน (Married-civ-spouse)" else 0
exec_val = 1 if is_exec == "ผู้บริหาร (Exec-managerial)" else 0

# 4. ปุ่มกดทำนายผลแบบเต็มความกว้าง
if st.button("🚀 ทำนายผลรายได้", use_container_width=True):
    
    # ดึงค่าทั้งหมดมาสร้างเป็นตาราง 1 แถว (ชื่อคอลัมน์ต้องเป๊ะ)
    input_data = pd.DataFrame([[
        age, education_num, capital_gain, hours_per_week, married_val, exec_val
    ]], columns=[
        'age', 'education.num', 'capital.gain', 'hours.per.week', 
        'marital.status_Married-civ-spouse', 'occupation_Exec-managerial'
    ])
    
    # ให้โมเดลทำนาย
    prediction = model.predict(input_data)[0]
    
    # แสดงผลลัพธ์
    if prediction == 1:
        st.success("🎉 **ผลการทำนาย:** รายได้ของคุณมีแนวโน้ม **มากกว่า $50K ต่อปี (>50K)**")
        st.balloons()
    else:
        st.warning("💵 **ผลการทำนาย:** รายได้ของคุณมีแนวโน้ม **น้อยกว่าหรือเท่ากับ $50K ต่อปี (<=50K)**")
