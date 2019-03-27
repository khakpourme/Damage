from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import category_encoders as ce
import random
from gasianrank import GaussRankScaler
import seaborn as sns


df = pd.read_csv("pdata_2012.csv", error_bad_lines=False, usecols=range(1, 27) + range(28, 54) + [70], header=0)

ce_hash = ce.HashingEncoder(cols=['Komm', 'Year'], n_components=9)
df = ce_hash.fit_transform(df)


# Extracting the training and test datasets
train_dataset = df.sample(frac=1, random_state=0)
test_dataset = pd.read_csv('testing_dataset.csv').reset_index(drop=True)
sns.pairplot(train_dataset[["Sum_NOK"]], diag_kind="kde")
plt.show()

# Data statistics
train_stats = train_dataset.describe()
train_stats.pop("Sum_NOK")
train_stats = train_stats.transpose()

# Split features from labels
train_labels = train_dataset.pop('Sum_NOK')
test_labels = test_dataset.pop('Sum_NOK')


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
normed_train_labels = norm(train_labels)
normed_train_labels_df = pd.DataFrame(normed_train_labels)
sns.pairplot(normed_train_labels_df[["Sum_NOK"]], diag_kind="kde")
plt.show()
sns.pairplot(normed_train_data[["BoligAlder","Days20","WindEXP","VUlExpoInd"]], diag_kind="kde")
plt.show()

counter = 0
max_count = 100
mae_list = {}
for count in xrange(max_count):
    lr = random.uniform(1e-1, 1e-5) #1e-1, 1e-5 1e-3, 1e-4
    dropout = random.uniform(0, 0.99) #0, 0.99 0.01, 0.1
    nlayer = np.random.randint(1, 6) #1, 6  4, 6
    # nnode = np.random.randint(1, 500)
    #  Build a Dense layer Model
    def build_model():
        global model
        model = None
        if nlayer == 1:
            global nnodelist
            nnodelist = []
            for i in xrange(nlayer+1):
                nnode = np.random.randint(1, 400) #1, 500
                nnodelist.append(nnode)
            model = keras.Sequential([
                layers.Dense(nnodelist[0], activation=tf.nn.leaky_relu, input_shape=[len(train_dataset.keys())]),
                layers.Dropout(dropout),
                layers.Dense(nnodelist[1], activation=tf.nn.leaky_relu),
                layers.Dropout(dropout),
                layers.Dense(1)
            ])
        elif nlayer == 2:
            nnodelist = []
            for i in xrange(nlayer+1):
                nnode = np.random.randint(1, 500)
                nnodelist.append(nnode)
            model = keras.Sequential([
                layers.Dense(nnodelist[0], activation=tf.nn.leaky_relu, input_shape=[len(train_dataset.keys())]),
                layers.Dropout(dropout),
                layers.Dense(nnodelist[1], activation=tf.nn.leaky_relu),
                layers.Dropout(dropout),
                layers.Dense(nnodelist[2], activation=tf.nn.leaky_relu),
                layers.Dropout(dropout),
                layers.Dense(1)
            ])
        elif nlayer == 3:
            nnodelist = []
            for i in xrange(nlayer+1):
                nnode = np.random.randint(1, 500)
                nnodelist.append(nnode)
            model = keras.Sequential([
                layers.Dense(nnodelist[0], activation=tf.nn.leaky_relu, input_shape=[len(train_dataset.keys())]),
                layers.Dropout(dropout),
                layers.Dense(nnodelist[1], activation=tf.nn.leaky_relu),
                layers.Dropout(dropout),
                layers.Dense(nnodelist[2], activation=tf.nn.leaky_relu),
                layers.Dropout(dropout),
                layers.Dense(nnodelist[3], activation=tf.nn.leaky_relu),
                layers.Dropout(dropout),
                layers.Dense(1)
            ])
        elif nlayer == 4:
            nnodelist = []
            for i in xrange(nlayer+1):
                nnode = np.random.randint(1, 500)
                nnodelist.append(nnode)
            model = keras.Sequential([
                layers.Dense(nnodelist[0], activation=tf.nn.leaky_relu, input_shape=[len(train_dataset.keys())]),
                layers.Dropout(dropout),
                layers.Dense(nnodelist[1], activation=tf.nn.leaky_relu),
                layers.Dropout(dropout),
                layers.Dense(nnodelist[2], activation=tf.nn.leaky_relu),
                layers.Dropout(dropout),
                layers.Dense(nnodelist[3], activation=tf.nn.leaky_relu),
                layers.Dropout(dropout),
                layers.Dense(nnodelist[4], activation=tf.nn.leaky_relu),
                layers.Dropout(dropout),
                layers.Dense(1)
            ])
        elif nlayer == 5:
            nnodelist = []
            for i in xrange(nlayer+1):
                nnode = np.random.randint(1, 500)
                nnodelist.append(nnode)
            model = keras.Sequential([
                layers.Dense(nnodelist[0], activation=tf.nn.leaky_relu, input_shape=[len(train_dataset.keys())]),
                layers.Dropout(dropout),
                layers.Dense(nnodelist[1], activation=tf.nn.leaky_relu),
                layers.Dropout(dropout),
                layers.Dense(nnodelist[2], activation=tf.nn.leaky_relu),
                layers.Dropout(dropout),
                layers.Dense(nnodelist[3], activation=tf.nn.leaky_relu),
                layers.Dropout(dropout),
                layers.Dense(nnodelist[4], activation=tf.nn.leaky_relu),
                layers.Dropout(dropout),
                layers.Dense(nnodelist[5], activation=tf.nn.leaky_relu),
                layers.Dropout(dropout),
                layers.Dense(1)
            ])

        optimizer = tf.keras.optimizers.Adam(lr)
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


    class LossHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            global counter
            counter = counter + 1
            print('TestNo:', counter, 'lr:', lr, 'dropout:', dropout, 'nlayer:', nlayer, 'nnode:', nnodelist)
        def on_train_end(self, logs={}):
            loss, mae = model.evaluate(normed_train_data, normed_train_labels, verbose=0)
            print('Testing mae: {}'.format(mae))
            global mae_list
            mae_list.update({counter: mae})


    EPOCHS = 10
    history = LossHistory()
    history1 = model.fit(
        normed_train_data, normed_train_labels,
        epochs=EPOCHS, validation_data=None, verbose=0, batch_size=64,
        callbacks=[PrintDot(), history])
print(mae_list)
lowest = min(zip(mae_list.values(), mae_list.keys()))
print('Best is:', lowest)
