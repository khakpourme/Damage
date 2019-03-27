import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import pandas as pd
import category_encoders as ce
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold # import KFold

# Extracting the training and test datasets

# In[7]:
df = pd.read_csv('train_dataset.csv')
ce_hash = ce.HashingEncoder(cols=['Komm', 'Year'], n_components=9)
df = ce_hash.fit_transform(df)
data = df.values
x = data[:, 0:57] # all rows, no label
y = data[:, 57] # all rows of the labeled column
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

xgb_model = xgboost.XGBRegressor(learning_rate =0.05, n_estimators=1000, max_depth=7, min_child_weight=12, gamma=0.03,
                                  subsample=0.6, colsample_bytree=1, reg_alpha=0.5, reg_lambda=1e-05)

xgb_model = xgboost.XGBRegressor(random_state=42)

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

kf = KFold(n_splits=5) # Define the split - into 5 folds
kf.get_n_splits(df) # returns the number of splitting iterations in the cross-validator
print(kf)
KFold(n_splits=5, random_state=42)

for train_index, test_index in kf.split(x):

    # print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    scaler = StandardScaler()
    scaler.fit(x_train)
    # Apply transform to both the training set and the test set.
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    xgb_model.fit(x_train, y_train)
    # Make predictions on test data
    predictions = xgb_model.predict(x_test)
    # Performance metrics
    errors = abs(predictions - y_test)
    print('Mean absolute error:', round(np.mean(errors), 2), 'Nok.')
