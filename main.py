import numpy as np 
import pandas as pd 
import matplotlib.cm as cm 
import matplotlib.pyplot as plt 

#Importing data
path = ""  
data_train = pd.read_csv(path + "train.csv") 
data_train.head() 
data_train.shape 
category={'A': 1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'H':8} 
data_train["category"]=data_train["category"].map(category) 
data_train.head()

data_train=data_train[data_train.views!='F'] 
data_train=data_train[data_train.likes!='F'] 
data_train=data_train[data_train.dislikes!='F'] 
data_train=data_train[data_train.comment!='F'] 

data_train["views"] = pd.to_numeric(data_train["views"]) 
data_train["comment"] = pd.to_numeric(data_train["comment"]) 
data_train["likes"] = pd.to_numeric(data_train["likes"]) 
data_train["dislikes"] = pd.to_numeric(data_train["dislikes"]) 
data_train["adview"]=pd.to_numeric(data_train["adview"]) 
column_vidid=data_train['vidid'] 
#Encoding features like category ,duration ,vidid
from sklearn.preprocessing import LabelEncoder 

data_train['duration']=LabelEncoder().fit_transform(data_train['duration']) 
data_train['vidid']=LabelEncoder().fit_transform(data_train['vidid']) 
data_train['published']=LabelEncoder().fit_transform(data_train['published']) 
data_train.head()
# Convert Time_in_sec for duration
import datetime 
import time 

def checki(x): 
    y = x[2:] 
    h = '' 
    m = '' 
    s = '' 
    mm = '' 
    P = ['H','M','S'] 
    for i in y: 
        if i not in P: 
            mm+=i 
        else: 
            if(i=="H"): 
                h = mm 
                mm = '' 
            elif(i == "M"): 
                m = mm 
                mm = '' 
            else: 
                s = mm 
                mm = '' 
    if(h==''): 
        h = '00' 
    if(m == ''): 
        m = '00' 
    if(s==''): 
        s='00' 
    bp = h+':'+m+':'+s 
    return bp 
    
train=pd.read_csv("train.csv") 
mp = pd.read_csv(path + "train.csv")["duration"] 
time  = mp.apply(checki)  

def func_sec(time_string): 
    h, m, s = time_string.split(':') 
    return int(h) * 3600 + int(m) * 60 + int(s)  
    
time1=time.apply(func_sec) 

data_train["duration"]=time1 
data_train.head()

import seaborn as sns

plt.hist(data_train["category"], color='skyblue', edgecolor='black')
plt.xlabel('Category')
plt.ylabel('Frequency')
plt.title('Histogram of Categories')
plt.show()

plt.plot(data_train["adview"], color='red')
plt.xlabel('Index')
plt.ylabel('Adview')
plt.title('Adview over Index')
plt.show()

data_train = data_train[data_train["adview"] < 2000000]

f, ax = plt.subplots(figsize=(10, 8))
corr = data_train.corr()

light_colors = sns.light_palette("seagreen", as_cmap=True)

sns.heatmap(
            corr, 
            mask=np.zeros_like(corr, dtype=bool), 
            cmap=light_colors, 
            square=True, 
            ax=ax, 
            annot=True,
           )
plt.title('Correlation Heatmap')
plt.show()
# Split Data
Y_train  = pd.DataFrame(data = data_train.iloc[:, 1].values, columns = ['target']) 
data_train=data_train.drop(["adview"],axis=1) 
data_train=data_train.drop(["vidid"],axis=1) 
data_train.head() 
from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(
    data_train, Y_train, test_size=0.2, random_state=42
) 
X_train.shape 
# Normalise Data
from sklearn.preprocessing import MinMaxScaler 

scaler = MinMaxScaler() 
X_train=scaler.fit_transform(X_train) 
X_test=scaler.fit_transform(X_test) 
X_train.mean()
# Evaluation Metrics
from sklearn import metrics 

def print_error(X_test, y_test, model_name): 
    prediction = model_name.predict(X_test) 
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, prediction))   
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, prediction))   
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, prediction))) 
    
from sklearn import linear_model 

linear_regression = linear_model.LinearRegression()  
linear_regression.fit(X_train, y_train)  
print_error(X_test,y_test, linear_regression) 

from sklearn.tree import DecisionTreeRegressor 

decision_tree = DecisionTreeRegressor() 
decision_tree.fit(X_train, y_train) 
print_error(X_test,y_test, decision_tree)  

from sklearn.ensemble import RandomForestRegressor 

n_estimators = 200 
max_depth = 25 
min_samples_split=15 
min_samples_leaf=2 
random_forest = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, min_samples_split=min_samples_split  )
random_forest.fit(X_train,y_train)
print_error(X_test,y_test, random_forest)

from sklearn.svm import SVR 

supportvector_regressor = SVR() 
supportvector_regressor.fit(X_train,y_train) 
print_error(X_test,y_test, linear_regression)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ann = keras.models.Sequential([ 
    layers.Dense(6, activation="relu", input_shape=X_train.shape[1:]),  
    layers.Dense(6, activation="relu"), 
    layers.Dense(1) 
])

optimizer = keras.optimizers.Adam()
loss = keras.losses.MeanSquaredError()
ann.compile(optimizer=optimizer, loss=loss, metrics=["mean_squared_error"])
# Fit the model
history = ann.fit(X_train, y_train, epochs=100)

ann.summary()

print_error(X_test, y_test, ann)

import joblib

joblib.dump(decision_tree, "decisiontree_youtubeadview.pkl")

ann.save("ann_youtubeadview.h5")
