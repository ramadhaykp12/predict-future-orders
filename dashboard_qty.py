import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import pickle as pkl
import streamlit as st
with open('prophet_model.pkl', 'rb') as file:
        model = pkl.load(file)

st.write('Input future date in days')
days_input = st.number_input('How many future days ?', min_value=0)
future_dates = model.make_future_dataframe(periods=days_input, freq='D')

if st.button('Prediksi'):
        if days_input:
            prediction = model.predict(future_dates)
            st.table(prediction)
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.lineplot(data=prediction, x='ds', y='yhat')
            st.write('Visualisasi Hasil predikisi: ')
            st.pyplot(fig)
        else:
            st.write('Input gagal')
