import streamlit as st
import pandas as pd
import requests 
import matplotlib.pyplot as plt

API_URL = "http://127.0.0.:5000"

st.set_page_config(
    layout ='wide',
    page_title="Linear Regression"
)

st.title("Salary Prediction using Linear Regression")

response = requests.get(f"{API_URL}/data")
if response.status_code == 200:
    data = pd.DataFrame(response.json())
else:
    st.error("Error fetching dataset!")