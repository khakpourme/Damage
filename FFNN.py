from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt
import category_encoders as ce
from gasianrank import GaussRankScaler
import seaborn as sns


df = pd.read_csv("pdata_2012.csv", error_bad_lines=False, usecols=range(1, 27) + range(28, 54) + [70], header=0)
print('Original dataset:')
print(df.tail())

ce_hash = ce.HashingEncoder(cols=['Komm', 'Year'], n_components=9)
df = ce_hash.fit_transform(df)
print('After Hashing:')
print(df.tail())

# Extracting the training and test datasets
train_dataset = df.sample(frac=1, random_state=0)
test_dataset = pd.read_csv('testing_dataset.csv').reset_index(drop=True)
sns.pairplot(train_dataset[["Sum_NOK"]], diag_kind="kde")
plt.show()
print('after sampling:')
print(train_dataset.tail())

# Data statistics
train_stats = train_dataset.describe()
train_stats.pop("Sum_NOK")
train_stats = train_stats.transpose()
print('train stats:')
print(train_stats)

# Split features from labels
train_labels = train_dataset.pop('Sum_NOK')
print('train labels:')
print(train_labels.tail())
print('train dataset:')
print(train_dataset.tail())
test_labels = test_dataset.pop('Sum_NOK')
print('test labels:')
print(test_labels.tail())
print('test dataset:')
print(test_dataset.tail())


# Normalizing the data
def norm(x):

    s = GaussRankScaler()
    x_ = s.fit_transform(x)
    assert x_.shape == x.shape

    return x_

x_cols = train_dataset.columns[9:]
x = train_dataset[x_cols]
x_ = norm(x)
y_cols = train_dataset.columns[:9]
y = train_dataset[y_cols]
normed_train_data = pd.concat([y, x_], axis=1)
# normed_test_data = norm(test_dataset)
normed_train_labels = norm(train_labels)
normed_train_labels_df = pd.DataFrame(normed_train_labels)
print(normed_train_labels_df.tail())
print('after normalizing train:')
print(normed_train_data.tail())
print('after normalizing labels:')
print(normed_train_labels.tail())

sns.pairplot(normed_train_labels_df[["Sum_NOK"]], diag_kind="kde")
plt.show()
sns.pairplot(normed_train_data[["BoligAlder","Days20","WindEXP","VUlExpoInd"]], diag_kind="kde")
plt.show()


counter = 0
max_count = 100
mae_list = {}

#  Build a Dense layer Model
def build_model():
    global model
    model = None

    # nnodelist = []

    model = keras.Sequential([
        layers.Dense(281, input_shape=[len(train_dataset.keys())]),
        keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                        beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                        beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                                        gamma_constraint=None),
        layers.Activation(activation=tf.nn.leaky_relu),
        layers.Dropout(0.8),
        layers.Dense(75),
        keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                        beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                        beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                                        gamma_constraint=None),
        layers.Activation(activation=tf.nn.leaky_relu),
        layers.Dropout(0.8),
        layers.Dense(92),
        keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                        beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                        beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                                        gamma_constraint=None),
        layers.Activation(activation=tf.nn.leaky_relu),
        layers.Dropout(0.8),
        layers.Dense(361),
        keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                        beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                        beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                                        gamma_constraint=None),
        layers.Activation(activation=tf.nn.leaky_relu),
        layers.Dropout(0.8),
        layers.Dense(368),
        keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                        beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                        beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                                        gamma_constraint=None),
        layers.Activation(activation=tf.nn.leaky_relu),
        layers.Dropout(0.8),
        layers.Dense(393),
        keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                        beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                        beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                                        gamma_constraint=None),
        layers.Activation(activation=tf.nn.leaky_relu),
        layers.Dropout(0.8),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.Adam(0.0013)
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error'])
    return model


model = build_model()
print(model.summary())


# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


EPOCHS = 100

history = model.fit(
    normed_train_data, normed_train_labels,
    epochs=EPOCHS, validation_split=0.2, verbose=0, batch_size=64,
    callbacks=[PrintDot()], shuffle=True)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
# print(hist.tail())
print(hist['mean_absolute_error'])


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean abs Error [Sum_NOK]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
              label='Val Error')
    plt.ylim([hist.loc[0,'mean_absolute_error'], hist.loc[0,'mean_absolute_error']+5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.plot(hist['epoch'], hist['loss'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_loss'],
             label='Val Error')
    plt.ylim([hist.loc[0,'val_loss'], hist.loc[0,'val_loss']+5])
    plt.legend()
    plt.show()


plot_history(history)
