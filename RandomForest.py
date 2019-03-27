import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold # import KFold

df = pd.read_csv('training_dataset.csv')
data = df.values
x = data[:, 0:51] # all rows, no label
y = data[:, 51] # all rows of the labeled column

# ('Best parameters:', {'bootstrap': False, 'min_samples_leaf': 1, 'n_estimators': 1000, 'max_features': 'sqrt',
#  'min_samples_split': 2, 'max_depth': None})
model = RandomForestRegressor(n_estimators=1000, random_state=42, bootstrap=False, min_samples_leaf=1, max_features=9,
                              min_samples_split=2, max_depth=None)

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
KFold(n_splits=5, random_state=42, shuffle=True)

for train_index, test_index in kf.split(x):

    print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    scaler = StandardScaler()
    scaler.fit(x_train)
    # Apply transform to both the training set and the test set.
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    print('Training Features Shape:', x_train.shape)
    print('Training Labels Shape:', y_train.shape)
    print('Testing Features Shape:', x_test.shape)
    print('Testing Labels Shape:', y_test.shape)


    # Train the model on training data
    model.fit(x_train, y_train);
    print("yes")

    # Use the forest's predict method on the test data
    predictions = model.predict(x_test)
    # Calculate the absolute errors
    errors = abs(predictions - y_test)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'Nok.')

    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / y_test)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')


