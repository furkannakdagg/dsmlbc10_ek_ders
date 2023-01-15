import pandas as pd
import joblib
import streamlit as st

# streamlit run dsmlbc10-databee.py

# .ipynb => DESTEKLEMEZ!
st.title("Titanic Prediction ⛴")
# st.markdown("Yazı")
# st.text("Yazı tipi 2")
# st.code("import numpy as np")
# st.success("Kazanan")
# st.error("Kaybeden")

# String ve numerik değer alma
# name = st.text_input("Adınızı Girin")
# age = st.number_input("Yaşınızı Girin", min_value=1, max_value=90, step=1, value=30)
# if name:
#     st.markdown(f"Adınız {name}, yaşınız {age}")
#
#
# img = st.checkbox("Resim görmek ister misiniz?")
# if img:
#     from PIL import Image
#     miuul = Image.open("images/miuul.png")
#     st.image(miuul, width=250)
#
# # gender = st.selectbox("Cinsiyet", ["Erkek", "Kadın"])
# option = ["Erkek", "Kadın"]
# gender = st.selectbox("Cinsiyet", option)
#
# st.sidebar.selectbox("Cinsiyet", ["Male", "Female"])







# SAYFAYI BÖLME
#col1, col2 = st.columns(2)

#with col1:
    # 'pclass', 'sex', 'age', 'parch', "embarked"
pclass = st.number_input("pclass", min_value=1, max_value=3, step=1)
gender = ["Male", "Female"]
sex = st.selectbox("sex", gender)
age = st.number_input("age", min_value=10, max_value=99, step=1)
parch = st.number_input("parch", min_value=0, max_value=6, step=1)
emb = ['S', "C", "Q"]
embarked = st.selectbox("embarked", emb)

pred_df = pd.DataFrame({"pclass": pclass,
                        "sex": sex, "age": age,
                        "parch": parch, "embarked": embarked}, index=[0])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in ["sex", "embarked"]:
    pred_df[col] = le.fit_transform(pred_df[col])
surv = st.checkbox("Tahmini Gör")


#with col2:
    # MODELİ ALMA
model = joblib.load("model.pkl")

if surv:
    # TAHMİN
    st.markdown(f"Survived Tahmini: **{model.predict(pred_df)[0]}**")



#multi = st.multiselect("multiselect_deneme", ["erkek", "kadın"])
#st.markdown(multi)

