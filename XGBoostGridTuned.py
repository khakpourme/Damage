import numpy as np
import pandas as pd
import time

import xgboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA # PCA
from sklearn.model_selection import GridSearchCV # Hyper-Parameter Optimization using grid search
from sklearn.metrics import mean_absolute_error, r2_score
import RegscorePy


df = pd.read_csv("pdata_2012.csv", error_bad_lines=False, usecols=range(2, 27) + range(28, 54) + [70], header=0)

df.head(10)


# Extracting the training and test datasets

data = df.values
x = data[:, 0:50] # all rows, no label
y = data[:, 50] # all rows of the labeled column
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# For tuning parameters
parameters_for_testing = {
   'colsample_bytree': [0.8, 1],        # Typical values: 0.5-1
   'gamma': [0.03, 0.05, 0.07],
   'min_child_weight': [8, 10, 12],     # Used to control over-fitting. Higher values prevent a model from learning
                                        # relations which might be highly specific to the particular sample selected
                                        # for a tree.
   'learning_rate': [0.05, 0.07, 0.09], # Lower values are generally preferred as they make the model robust to the
                                        # specific characteristics of tree and thus allowing it to generalize well.
   'max_depth': [5, 7],                 # Used to control over-fitting as higher depth will allow model to learn
                                        # relations very specific to a particular sample. Typical values: 3-10
   'n_estimators': [1000],              # Checked up to 10000 no change on Accuracy
   'reg_alpha': [0.5, 0.75, 1],         #
   'reg_lambda': [1e-5],
   'subsample': [0.6, 0.8]              # Values slightly less than 1 make the model robust by reducing the variance.
                                        # Typical values ~0.8 generally work fine but can be fine-tuned further.
}

xgb_model = xgboost.XGBRegressor(random_state=42)

gsearch1 = GridSearchCV(estimator=xgb_model, param_grid=parameters_for_testing, n_jobs=-1, verbose=2,
                         scoring='neg_mean_squared_error', cv=5)
# gsearch1 = RandomizedSearchCV(estimator=xgb_model, param_distributions= parameters_for_testing, n_jobs=-1,
#                             scoring='neg_mean_squared_error', n_iter=100, cv=3, verbose=2)

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

gsearch1.fit(x_train, y_train)
# Make predictions on test data
start = time.time()
predictions = gsearch1.predict(x_test)
end = time.time()
duration = end - start
# Performance metrics
mse = mean_absolute_error(y_test, predictions)
print('mse is:', mse)
print('XGB Testing Duration: ', duration)
r2 = r2_score(y_test, predictions)
print('r2 is: ', r2)
adjusted_r_squared = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - x_test.shape[1] - 1)
print('adj_r2 is: ', adjusted_r_squared)
print ('AIC is: ', RegscorePy.aic.aic(np.asarray(y_test), np.asarray(predictions), 519))
print("Best parameters:", gsearch1.best_params_)
