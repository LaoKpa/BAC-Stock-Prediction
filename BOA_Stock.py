#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[17]:


import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Dropout
import matplotlib.pyplot as plt


# # Importing Stock data from yahoo

# In[3]:


# get the stock quote
df=web.DataReader('BAC',data_source='yahoo',start='2010-01-01', end='2020-5-12')


# # Data Filter

# In[4]:


df=df[['Open','Volume']]
print(df)


# In[177]:


# Visualising our data
plt.figure(figsize=[14,8])
plt.title("Close price History")
plt.plot(df['Open'])
plt.xlabel('Date',fontsize=16)
plt.ylabel('Stock Price',fontsize=16)
plt.show()


# In[14]:


# seperating training and testing data
# We are going to test 60 days of latest stock price
# The rest data before the latest 60 days will be trained
Training_data=df[:2547].values
Test_data=df[2547:].values


# In[15]:


print("Shape of Test_data ",Test_data.shape)
print(" Shape of Training data ", Training_data.shape)


# # Feature Scaling

# In[18]:


#Scaling the data between one and 0
sc=MinMaxScaler(feature_range=(0,1))
Training_data_scaled=sc.fit_transform(Training_data)


# # Creating Time Steps

# In[19]:


x_train=[]
y_train=[]

#Creating a 60 day timesteps 
for i in range(60,Training_data.shape[0]):
    #Takes data form 0 to 59. The process continues 
    x_train.append(Training_data_scaled[i-60:i])
    # takes the data of 60th position
    y_train.append(Training_data_scaled[i,0])


# In[20]:


# converting x_train and y_train to array
x_train=np.array(x_train)
y_train=np.array(y_train)


# In[22]:


print("The shape of x_train is:",x_train.shape)
print("The shape of y_train is:",y_train.shape)


# # Initialising the RNN

# In[28]:


regressor=Sequential()


# In[30]:


# First layer of LSTM and Dropout
#Dropout is important feature of LSTM wher it drops some portion of the data
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],2)))


# In[32]:


# adding dropout of 20%
regressor.add(Dropout(0.2))


# In[33]:


# Adding the second layer of the LSTM
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))


# In[34]:


# Adding the Third layer of the LSTM
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))


# In[35]:


#Adding fourth Layer of LSTM
# we remove return sequence as we are not returning any sequence
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))


# # Output layer

# In[36]:


regressor.add(Dense(units=1))


# # Adding Optimizer

# In[37]:


# for the optimer we are using adam which is a socratic gradient descent
regressor.compile(optimizer='adam', loss='mean_squared_error')


# # Training our RNN

# In[38]:


# Batch size of 32 is a standard batch size most of the people use
regressor.fit(x_train,y_train,epochs=100,batch_size=32)


# # Making Prediction

# In[122]:


# We are going to concat both training and testing dataset
Train_60_days=df[:2547].tail(60)
Test_60_days=df[2547:]
Train_60_days.head(5)


# In[123]:


new_df=Train_60_days.append(Test_60_days, ignore_index=True)
new_df.head()


# In[124]:


new_df.shape


# In[125]:


inputs=sc.transform(new_df)
inputs.shape[0]


# In[126]:


new_x_test=[]
new_y_test=[]

for i in range(60,inputs.shape[0]):
    new_x_test.append(inputs[i-60:i])
    new_y_test.append(inputs[i,0])
    


# In[127]:


new_x_test=np.array(new_x_test)
new_y_test=np.array(new_y_test)
new_x_test.shape


# In[128]:


final_prediction=regressor.predict(new_x_test)


# In[129]:


sc.scale_


# In[130]:


scaler=(1/3.26904210e-02)
scaler


# In[131]:


final_prediction=final_prediction*scaler
final_prediction=(final_prediction+5.11)
new_y_test=new_y_test*scaler
new_y_test=(new_y_test+5.11)


# In[132]:


final_prediction


# In[ ]:





# In[133]:


plt.figure(figsize=(16,8))
plt.plot(new_y_test, color='red', label="Real BOA Stock")
plt.plot(final_prediction, color='blue', label="Predicted BOA Stock")
plt.title("BOA stock prediction")
plt.xlabel("Time")
plt.ylabel('Stock Price')
plt.legend()
plt.show()


# # Table of Actual price vs Predicted price

# In[134]:


Visualise_Test_data=df[2547:]
Visualise_Test_data['Open_before_pred']=new_y_test
Visualise_Test_data['Actual_Prediction']=final_prediction
Visualise_Test_data


# # Predicting single day

# In[155]:


# Getting updated data for today
new_df=web.DataReader('BAC',data_source='yahoo',start='2020-01-01', end='2020-5-13')


# In[156]:


new_df=new_df[['Open','Volume']]


# In[165]:


Last_30_days=new_df.tail(60)


# In[166]:


print(Last_30_days)


# In[167]:


inputs_single=sc.transform(Last_30_days)


# In[168]:


Single_data=[]
for i in range(60,61):
    Single_data.append(inputs_single[i-60:i])
    


# In[169]:


Single_data=np.array(Single_data)


# In[170]:


Single_data


# In[171]:


Single_data.shape


# In[172]:


Single_final_prediction=regressor.predict(Single_data)


# In[173]:


Single_final_prediction=Single_final_prediction*scaler
Single_final_prediction=(Single_final_prediction+5.11)


# In[174]:


print(" The Stock price for next Day is", Single_final_prediction)


# # Finding the trend

# In[179]:


Visualise_Test_data[:5]


# In[180]:


final_prediction[0]


# In[193]:


Predict_trend=[]
Actual_trend=[]
for i in range(59):
    Predict_trend.append(final_prediction[i]-final_prediction[i+1])
    Actual_trend.append(new_y_test[i]-new_y_test[i+1])


# In[194]:


Trend_df=pd.DataFrame({'Actual_Trend':Actual_trend,'Predict_trend':Predict_trend})


# In[204]:


def color_negative_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = 'green' if val < 0 else 'red'
    return 'color: %s' % color


# In[205]:


s = Trend_df.style.applymap(color_negative_red)
s


# In[ ]:





# In[ ]:




