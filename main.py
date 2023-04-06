from flask import Flask,render_template,request
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

df=pd.read_csv('Mall_Customers.csv')
df['Gender'].replace({'Female': 0, 'Male': 1},inplace=True)
x=df.drop('CustomerID',axis=1)
scaler=MinMaxScaler()
x_scaled=scaler.fit_transform(x)
best_model=KMeans(n_clusters=16)
best_model.fit(x_scaled)

app=Flask('__main__')

@app.route('/')
def connect():
    return render_template('index.html')

@app.route('/input', methods=['GET','POST'])
def input():
    input_dict=request.form
    input_dict_values=list(input_dict.values())
    array=np.zeros(len(input_dict_values))
    array[0] = int(input_dict_values[0])
    array[1] = int(input_dict_values[1])
    array[2] = int(input_dict_values[2])
    array[3] = int(input_dict_values[3])

    input_array = scaler.transform([array])
    prediction=best_model.predict(input_array)[0]
    

    return render_template('index.html',result=prediction)

if __name__ == '__main__':
  app.run(host='0.0.0.0',port=8080)