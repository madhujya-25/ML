# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values
y = y.reshape(-1,1)


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

 ## Avoiding the dummy variable trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train) 

y_pred = regressor.predict(X_test)

 ##building optimal model using backwards eleimination 
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)

X_opt = X[:,[0,1,2,3,4,5]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_ols.summary())
print(regressor_ols.rsquared)
print(regressor_ols.rsquared_adj) 


X_opt = X[:,[0,1,3,4,5]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_ols.summary())
print(regressor_ols.rsquared)
print(regressor_ols.rsquared_adj) 

X_opt = X_opt[:,[0,2,3,4]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_ols.summary())
print(regressor_ols.rsquared)
print(regressor_ols.rsquared_adj) 

X_opt = X_opt[:,[0,1,3]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_ols.summary())
print(regressor_ols.rsquared)
print(regressor_ols.rsquared_adj) 

## we do not remove the third variable as it 
## results in the decrease of adjusted r squared value
"""X_opt = X_opt[:,[0,1]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_ols.summary())
print(regressor_ols.rsquared)
print(regressor_ols.rsquared_adj)"""  

X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size = 0.2, random_state = 0)
regressor.fit(X_train, y_train) 
y_pred2 = regressor.predict(X_test)

## y_pred2 is the final prediction 




 
























