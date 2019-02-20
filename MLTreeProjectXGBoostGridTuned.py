# coding: utf-8

# # Machine Learning tree project

# importing required packages for data processing and liear algebra

# In[4]:

import numpy as np
import pandas as pd


# importing sklearn packageg for Random Forest, spliting the dataset and scaling the values

# In[5]:
import xgboost

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold # import KFold
from sklearn.decomposition import PCA # PCA
from sklearn.model_selection import RandomizedSearchCV # Hyper-Parameter Optimization
from pprint import pprint
from sklearn.model_selection import GridSearchCV # Hyper-Parameter Optimization using grid search

# In[ ]:

import matplotlib.pyplot as plt


# In[6]:

df = pd.read_csv("pdata_2012.csv", error_bad_lines=False, usecols=range(2, 27) + range(28, 54) + [70], header=0)
# df = df.drop("ID", axis = 1)
# df = df.drop("Navn", axis = 1)

df.head(10)


# Extracting the training and test datasets

# In[7]:

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
# pprint(param_grid)

# xgb_model = xgboost.XGBRegressor(learning_rate=0.1, n_estimators=1000, max_depth=5, min_child_weight=1, gamma=0,
#                                  subsample=0.8, colsample_bytree=0.8, nthread=6, scale_pos_weight=1, seed=27)

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

# K-fold Cross Validation

# kf = KFold(n_splits=5) # Define the split - into 5 folds
# kf.get_n_splits(df) # returns the number of splitting iterations in the cross-validator
# print(kf)
# KFold(n_splits=5, random_state=42, shuffle=True)

# for train_index, test_index in kf.split(x):

# print("TRAIN:", train_index, "TEST:", test_index)
# x_train, x_test = x[train_index], x[test_index]
# y_train, y_test = y[train_index], y[test_index]

scaler = StandardScaler()
scaler.fit(x_train)
# Apply transform to both the training set and the test set.
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', x_test.shape)
print('Testing Labels Shape:', y_test.shape)

# Create the model

# In[8]:

# model = RandomForestRegressor(n_jobs=-1)

# Tryinhg different numbers of n_estimators

# In[2]:

#estimators = np.arange(10, 200, 10)
#scores = []
# for n in estimators:
#     model.set_params(n_estimators=n)
#     model.fit(x_train,y_train)
#     scores.append(model.score(x_test, y_test))
# plt.title("Effect of n_estimators")
# plt.xlabel("n_estimator")
# plt.ylabel("score")
# plt.plot(estimators, scores)
# plt.show()

# # The baseline predictions are the historical averages
# baseline_preds = x_train[:, feature_list.index('average')]
# # Baseline errors, and display average baseline error
# baseline_errors = abs(baseline_preds - y_test)
# print('Average baseline error: ', round(np.mean(baseline_errors), 2))

# # Train the model on training data
# model.fit(x_train, y_train);
# print("yes")
#
# # Use the forest's predict method on the test data
# predictions = model.predict(x_test)
# # Calculate the absolute errors
# errors = abs(predictions - y_test)
# # Print out the mean absolute error (mae)
# print('Mean Absolute Error:', round(np.mean(errors), 2), 'Nok.')
#
# # Calculate mean absolute percentage error (MAPE)
# mape = 100 * (errors / y_test)
# # Calculate and display accuracy
# accuracy = 100 - np.mean(mape)
# print('Accuracy:', round(accuracy, 2), '%.')
#
# # Get numerical feature importances
# importances = list(model.feature_importances_)
# # List of tuples with variable and importance
# feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# # Sort the feature importances by most important first
# feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# # Print out the feature and importances
# for pair in feature_importances:
#     print('Variable: {:20} Importance: {}'.format(*pair))
# # List of features sorted from most to least important
# sorted_importances = [importance[1] for importance in feature_importances]
# sorted_features = [importance[0] for importance in feature_importances]
# # Cumulative importances
# cumulative_importances = np.cumsum(sorted_importances)
#
# # Find number of features for cumulative importance of 95%
# # Add 1 because Python is zero-indexed
# noi = np.where(cumulative_importances > 0.95)[0][0] + 1
# print('Number of features for 95% importance:', noi)
#
# # Extract the names of the most important features
# important_feature_names = [feature[0] for feature in feature_importances[0:(noi-1)]]
# # Find the columns of the most important features
# important_indices = [feature_list.index(feature) for feature in important_feature_names]
# # Create training and testing sets with only the important features
# important_train_features = x_train[:, important_indices]
# important_test_features = x_test[:, important_indices]
#
# # Train the expanded model on only the important features
# model.fit(important_train_features, y_train);
# # Make predictions on test data
# predictions = model.predict(important_test_features)
# # Performance metrics
# errors = abs(predictions - y_test)
# print('Average absolute error after feature reduction:', round(np.mean(errors), 2), 'Nok.')
# # Calculate mean absolute percentage error (MAPE)
# mape = 100 * (errors / y_test)
# # Calculate and display accuracy
# accuracy = 100 - np.mean(mape)
# print('Accuracy after feature reduction:', round(accuracy, 2), '%.')

# Make an instance of the Model
pca = PCA(.95)
pca.fit(x_train)
print ("No. of PCs:", pca.n_components_)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)

gsearch1.fit(x_train, y_train)
# Make predictions on test data
predictions = gsearch1.predict(x_test)
# Performance metrics
errors = abs(predictions - y_test)
print('Average absolute error after PCA:', round(np.mean(errors), 2), 'Nok.')
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy after PCA:', round(accuracy, 2), '%.')

print('best score')
print(gsearch1.best_score_)
print("Best parameters:", gsearch1.best_params_)

# In[ ]:



# ('Best parameters:', {'bootstrap': False, 'min_samples_leaf': 1, 'n_estimators': 400, 'max_features': 'sqrt',
#  'min_samples_split': 2, 'max_depth': None})