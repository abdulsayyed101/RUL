import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# merge the csv files into one 

# Read CSV files
df1 = pd.read_csv('C:/Users/abdul/python subj/python_tut/RUL crucible/Crukibal1_6_Dataset1.csv')
df2 = pd.read_csv('C:/Users/abdul/python subj/python_tut/RUL crucible/Crukibal2_6_5_Dataset2.csv')
df3 = pd.read_csv('C:/Users/abdul/python subj/python_tut/RUL crucible/Crukibal3_6_9_Dataset3.csv')

# Concatenate dataframes
merged_df = pd.concat([df1, df2, df3], axis=0)

# Save merged dataframe to a new CSV file
merged_df.to_csv('merged.csv', index=False)

df = pd.read_csv('C:/Users/abdul/python subj/python_tut/RUL crucible/merged.csv')
df.drop(df.columns[4:10], axis=1, inplace=True)  # dropping unwanted cols
print(df)

# splitting in features and target 
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# model evaluation tool
from sklearn.metrics import r2_score   # R2 score should be greater than 0.7 and if greater than 0.9 then it is perfect model 
print(r2_score(y_test, y_pred)*100)
