import numpy as np
import pandas as pd
from keras.layers import Dense,LSTM,Flatten, GRU
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')

stock_data=pd.read_csv("GOOGL_2006-01-01_to_2018-01-01.csv",parse_dates=['Date'], index_col='Date',date_parser=dateparse)
stock_data.drop('Name',axis=1,inplace=True)

# stock_data.fillna(stock_data.mean(),inplace=True)

# plt.plot(stock_data['Close'])
# plt.xlabel("Date")
# plt.ylabel("Closing Prices")
# plt.title("Stock Trend")

stock_data['Temp']=stock_data['Volume']
stock_data['Volume']=stock_data['Close']
stock_data['Close']=stock_data['Temp']
stock_data.drop('Temp',axis=1,inplace=True)
# stock_data.head()

std_scale=StandardScaler()
std_stockdata = std_scale.fit_transform(stock_data)

# (stock_data.isnull()==True).sum()

# def create_timeseries(data,steps,features):
#   X=[]
#   y=[]
#   for i in range(len(data)-steps-1):
#     t=[]
#     for j in range(0,steps):
#       t.append(data[i+j][0:features])
#     X.append(t)
#     y.append(data[i+steps][4])
#   return X,y

# X,y = create_timeseries(std_stockdata,10,5)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# X_train = np.array(X_train)
# X_test = np.array(X_test)
# y_train = np.array(y_train)
# y_test = np.array(y_test)

# time_scale=(stock_data.iloc[-603:-1,3:4].index)

# X_train=X_train.reshape(X_train.shape[0],10,5)
# X_test=X_test.reshape(X_test.shape[0],10,5)

# def create_model():
#     model=Sequential()
# #     model.add(LSTM(70,return_sequences=True,input_shape=(10,5)))
#     model.add(Dense(90,input_shape=(10,5)))
#     model.add(Dense(90))
#     model.add(Flatten())
#     model.add(Dense(1))

#     model.compile(optimizer='adam',loss='mean_squared_error',metrics=['mse'])
    
#     return model



# model = create_model()
# model.fit(X_train,y_train,epochs=30,batch_size=64)

# mse(y_test,model.predict(X_test))

# plt.figure(figsize=(10,6))
# plt.plot(time_scale,model.predict(X_test),label='Predicted trend')
# plt.plot(time_scale,y_test,label="True trend")
# plt.xlabel("Date")
# plt.ylabel("Standardized Closing Prices")
# plt.legend(fontsize='large')

# X_train.shape

# # # create model
# model = KerasRegressor(build_fn=create_model,verbose=0)
# # # define the grid search parameters
# neurons_lstm1 = np.arange(10,110,10)
# neurons_lstm2 = np.arange(10,110,10)
# # neurons_dense1 = np.arange(10,110,10)
# epochs=np.arange(5,31,5)
# batch_size=[32,64,128,256,512]
# param_grid = dict(epochs=epochs,batch_size=batch_size)
# grid = GridSearchCV(estimator=model,scoring='neg_mean_squared_error',param_grid=param_grid, n_jobs=-1)
# grid_result = grid.fit(X_train, y_train)

# grid_result.best_params_

"""# SVM"""

# poly = PolynomialFeatures(degree=1)
# mapped_data = poly.fit_transform(std_stockdata)
# X,y=create_timeseries(mapped_data,10,mapped_data.shape[1])

X_train_svm, X_test_svm, y_train_svm,y_test_svm = train_test_split(std_stockdata[:,0:4],std_stockdata[:,4],test_size=0.2, shuffle=False)

parameters = [{'kernel': ['rbf'], 'gamma': np.arange(0,4.1,0.1),
               'C': [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]},
                    {'kernel': ['linear'], 'C': [1,10,20,30,40,50,60,70,80,90,100]},
              {'kernel': ['poly'], 'gamma': np.arange(0, 4.1, 0.1), 'degree': [1, 2, 3, 4, 5], 'C': [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}]
svr_model = GridSearchCV(SVR(),parameters,scoring = 'neg_mean_squared_error')
res=svr_model.fit(X_train_svm,y_train_svm)

# plt.plot(y_test,label="True trend")
# plt.plot(poly_regression.predict(X_test),label='Predicted trend')
# plt.legend()

print(res.best_params_)

