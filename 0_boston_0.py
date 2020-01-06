#!/usr/bin/env python3

import matplotlib.pyplot as plt
from keras.datasets import boston_housing
import numpy as np
from keras import models
from keras import layers
from keras.utils.np_utils import to_categorical

(train_data, train_targets), (test_data,
                              test_targets) = boston_housing.load_data()

# 1st option is to create tensor (samples, word indices)
# 2nd option is to create one-hot encode list into 10,000 dimensional vector


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))

    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.

    return results


def to_one_hot(labels, dimension=46):

    results = np.zeros((len(labels), dimension))

    for i, label in enumerate(labels):
        results[i, label] = 1.

    return results


def trainloss_valloss_acc_valacc(loss_values, val_loss_values, acc_values, val_acc_values):

    epochs = range(1, 21)

    plt.subplot(2, 1, 1)
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.gca().set_title('Traiing Validation Loss')
    plt.xlabel('')
    plt.ylabel('')

    plt.subplot(2, 1, 2)
    plt.plot(epochs, acc_values, 'bo', label='Training acc')
    plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
    plt.gca().set_title('Training Validation Accuracy')
    plt.legend()

    plt.show()


# prep data
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

# model definition


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',
                  loss='mse', metrics=['mae'])

    return model


def smooth_curve(points, factor=0.9):
    smoothed_pts = []
    for point in points:
        if smoothed_pts:
            previous = smoothed_pts[-1]
            smoothed_pts.append(previous * factor + point * (1-factor))
        else:
            smoothed_pts.append(point)

    return smoothed_pts


# K-fold validation set
k = 4
num_val_samples = len(train_data) // k
num_epochs = 500
all_mae_histories = []
all_scores = []

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i+1)*num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i+1)*num_val_samples]
    partial_train_data = np.concatenate(
        [train_data[: i * num_val_samples],
            train_data[(i + 1) * num_val_samples:]], axis=0
    )
    partial_train_targets = np.concatenate(
        [train_targets[: i * num_val_samples],
            train_targets[(i + 1) * num_val_samples:]], axis=0
    )

    # model build & train
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)

    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)

print("all_mae_scores", all_scores)
print("mean", np.mean(all_scores))

average_mae_history = [np.mean([x[i] for x in all_mae_histories])
                       for i in range(num_epochs)]
smooth_mae_history = smooth_curve(average_mae_history[10:])

# plot
plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()
