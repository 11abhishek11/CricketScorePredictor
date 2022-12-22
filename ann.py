# Import the necessary libraries
from keras.models import Sequential
from keras.layers import Dense

# Importing the dataset
import pandas as pd
dataset = pd.read_csv('data/t20.csv')
X = dataset.iloc[:,[7,8,9,10,11,12,13]].values
y = dataset.iloc[:, 14].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Define the model architecture
model = Sequential()
model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# for batch_size in [16, 32, 64, 128]:

# Fit the model to the training data
model.fit(X_train, y_train, epochs=15, batch_size=16)

# Make predictions on the test data
y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# mse = mean_squared_error(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# print("Mean Squared Error:", mse)
# print("Mean Absolute Error:", mae)
print("\nR squared:", r2)

# Get the feature importance scores from the first layer
importances = model.layers[0].get_weights()[0]

# Normalize the importance scores
importances = importances / importances.sum()

import numpy as np
# Print the importance scores for each feature
for i, importance in enumerate(importances):
    print("Feature", i, "importance:", np.linalg.norm(importance))

