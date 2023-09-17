import streamlit as st
import pickle
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
import datetime as dt
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import base64


#image = Image.open('ock.jpg')
#st.image(image,width=700,use_column_width='never')
model1= pickle.load(open('knn.pkl', 'rb'))
model2=pickle.load(open('lg.pkl', 'rb'))
scaler_lg = pickle.load(open('scaler_lg.pkl', 'rb'))
scaler_knn = pickle.load(open('scaler_knn.pkl', 'rb'))

def user_report():
  Adjusted_Close = st.sidebar.slider('Adjusted Close', 100,1000, 1 )
  Volume = st.sidebar.slider('Volume', 1000,50000000, 1 )
  Date = st.sidebar.date_input('Date')
  month = Date.month
  day_of_month =  Date.day
  day_of_week = Date.strftime("%w")
  year = Date.year

  user_report_data = {
      'Adj Close':Adjusted_Close,
      'Volume':Volume,
      'day_of_week':day_of_week,
      'month':month,
      'day_of_month':day_of_month,
      'year':year,
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data

def select_algo():
  algos=st.sidebar.radio('Pick algorithm:', ['KNN','Linear Regression'])
  return algos

#### Interface ####

st.title('Stock Price Prediction :chart_with_upwards_trend:')
st.header('Stock Price Data :clipboard:')
st.sidebar.header(':red[Algorithm] :question:',divider='grey')
algo=select_algo()
st.sidebar.header(':red[Stock Price Data] :clipboard:',divider='grey')
user_data = user_report()
st.write(user_data)
if st.sidebar.button('Submit'):
# FUNCTION   
  print(user_data.values)
  X_scaled_knn=scaler_knn.transform(user_data.values)
  X_scaled_lg=scaler_lg.transform(user_data.values)
  print(X_scaled_knn)
  print(X_scaled_lg)
  if algo=='KNN':
    close = model1.predict(X_scaled_knn)
  else:
    close = model2.predict(X_scaled_lg)
  st.header('Predicted Closing price')
  st.subheader(':heavy_dollar_sign:'+str(np.round(close[0], 3)))
  print(close)