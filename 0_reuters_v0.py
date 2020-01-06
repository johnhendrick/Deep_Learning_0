#!/usr/bin/env python3

import matplotlib.pyplot as plt
from keras.datasets import reuters
import numpy as np
from keras import models
from keras import layers
from keras.utils.np_utils import to_categorical

(train_data, train_labels), (test_data,
                             test_labels) = reuters.load_data(num_words=10000)


word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key)
                           for (key, value) in word_index.items()])

decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?')
                             for i in train_data[0]])

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
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asanyarray(train_labels).astype('float32')
y_test = np.asanyarray(test_labels).astype('float32')

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

print("train labels", train_labels)
print("one_hot_train_labels", one_hot_train_labels.shape)

# model definition
model1 = models.Sequential()
model1.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model1.add(layers.Dense(64, activation='relu'))
model1.add(layers.Dense(46, activation='softmax'))

# creating validation set
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

print("partial_y_train", partial_y_train)
print(partial_y_train.shape)


# compile model
# sparce_categorical_crossentropy loss function is to be used if the model is trained using integer instead of float
model1.compile(optimizer='rmsprop',
               loss='categorical_crossentropy', metrics=['accuracy'])
history = model1.fit(partial_x_train,
                     partial_y_train, epochs=20, batch_size=256, validation_data=(x_val, y_val))
results = model1.evaluate(x_test, one_hot_test_labels)

# prediction
predictions = model1.predict(x_test)

# prints

print('SHOW RESULTS:', results)
print(predictions.shape)

# plotting validation and training loss
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']


trainloss_valloss_acc_valacc(
    loss_values, val_loss_values, acc_values, val_acc_values)
