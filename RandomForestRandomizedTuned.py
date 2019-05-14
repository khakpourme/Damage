import numpy as np
import pandas as pd
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA # PCA
from sklearn.model_selection import RandomizedSearchCV # Hyper-Parameter Optimization
from pprint import pprint
from sklearn.metrics import mean_absolute_error, r2_score
import RegscorePy

df = pd.read_csv("pdata_2012.csv", error_bad_lines=False, usecols=range(2, 27) + range(28, 54) + [70], header=0)

df.head(10)

data = df.values
x = data[:, 0:50] # all rows, no label
y = data[:, 50] # all rows of the labeled column
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

pprint(random_grid)

model = RandomForestRegressor()
model_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2,
                               random_state=42, n_jobs = -1)
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

model_random.fit(x_train, y_train)
# Make predictions on test data
start = time.time()
predictions = model_random.predict(x_test)
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
print("Best parameters:", model_random.best_params_)
