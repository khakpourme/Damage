import numpy as np
import pandas as pd
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA # PCA
from sklearn.model_selection import GridSearchCV # Hyper-Parameter Optimization using grid search
from sklearn.metrics import mean_absolute_error, r2_score
import RegscorePy

df = pd.read_csv("pdata_2012.csv", error_bad_lines=False, usecols=range(2, 27) + range(28, 54) + [70], header=0)
df.head(10)
data = df.values
x = data[:, 0:50] # all rows, no label
y = data[:, 50] # all rows of the labeled column
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Create the parameter grid based on the results of random search
param_grid = {
    'bootstrap': [False],
    'max_depth': [None],
    'max_features': [4, 9],
    'min_samples_leaf': [1, 2, 3],
    'min_samples_split': [2, 4],
    'n_estimators': [100, 200, 300, 1000]
}

# pprint(param_grid)

model = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = model, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)

print(df.columns.tolist())

# Labels are the values we want to predict
labels = np.array(df['Sum_NOK'])
# Remove the labels from the features
# axis 1 refers to the columns
df = df.drop('Sum_NOK', axis=1)
# Saving feature names for later use
feature_list = list(df.columns)
# Convert to numpy array
df = np.array(df)
scaler = StandardScaler()
scaler.fit(x_train)
# Apply transform to both the training set and the test set.
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', x_test.shape)
print('Testing Labels Shape:', y_test.shape)

# Make an instance of the Model
pca = PCA(.95)
pca.fit(x_train)
print ("No. of PCs:", pca.n_components_)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)

grid_search.fit(x_train, y_train)
# Make predictions on test data
start = time.time()
predictions = grid_search.predict(x_test)
end = time.time()
duration = end - start
# Performance metrics
mse = mean_absolute_error(y_test, predictions)
print('mse is:', mse)
print('RF Testing Duration: ', duration)
r2 = r2_score(y_test, predictions)
print('r2 is: ', r2)
adjusted_r_squared = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - x_test.shape[1] - 1)
print('adj_r2 is: ', adjusted_r_squared)
print ('AIC is: ', RegscorePy.aic.aic(np.asarray(y_test), np.asarray(predictions), 519))
print("Best parameters:", grid_search.best_params_)
