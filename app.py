import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import datetime as dt
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import base64


#image = Image.open('ock.jpg')
#st.image(image,width=700,use_column_width='never')
model1= pickle.load(open('best_model.pkl', 'rb'))
model2=pickle.load(open('lg.pkl', 'rb'))


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
print(algo)
st.sidebar.header(':red[Stock Price Data] :clipboard:',divider='grey')
#image = Image.open('bb.jpg')
#st.image(image, '')

# FUNCTION
user_data = user_report()
st.write(user_data)
scaler = StandardScaler()
X_scaled=scaler.fit_transform(user_data.values)
if algo=='KNN':
  close = model1.predict(X_scaled)
else:
  close = model2.predict(X_scaled)
st.header('Predicted Closing price')

if st.sidebar.button('Submit'):
  st.subheader(':heavy_dollar_sign:'+str(np.round(close[0], 2)))
print(user_data)