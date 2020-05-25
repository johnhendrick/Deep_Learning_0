import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from keras import models, layers, regularizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# staging training data

df = pd.read_csv('/home/johnh/Documents/Deep_Learning_0/OneCDS_W_DUN_PANADOLEXTRA_19JUL12_000_1M.csv')
# remove sign off reasons
df['Activity'] = df['Activity'].str.extract(r'(.{0,17})')
df.sort_values(['CASE_ID', 'Timestamp'], inplace=True, ignore_index=True)

# map activity name to number
act_to_int = dict((act, i+1)
                  for i, act in enumerate(list(df.Activity.unique())))
int_to_act = dict((i+1, act)
                  for i, act in enumerate(list(df.Activity.unique())))
df['activity_int'] = df['Activity'].map(act_to_int)

# subset df for testing
df_sub = df.head(20)

# decode function


def decode_activity(seqs):
    decoded_seqs = []
    for seq in seqs:
        decoded_seq = [int_to_act[i] for i in seq]
        decoded_seqs.append(decoded_seq)
    return decoded_seqs

# endcode function


def create_seq(dataframe):
    prev = 'blank'
    encoded_seq = []
    new = []
    for i in range(len(dataframe)):
        # new case_id
        if dataframe.loc[i, "CASE_ID"] != prev and len(new) != 0:
            encoded_seq.append(new)
            new = []
            new.append(dataframe.loc[i, "activity_int"])

        # still within same caseid
        else:
            new.append(dataframe.loc[i, "activity_int"])

        prev = dataframe.loc[i, "CASE_ID"]

    encoded_seq.append(new)
    return pad_sequences(encoded_seq, padding='post')


def scaledown(datain):
    scaler = MinMaxScaler()
    scaled_output = scaler.fit_transform(datain)
    return scaled_output


X_original = create_seq(df)
X_original_df = pd.DataFrame(X_original, index=df.CASE_ID.unique())
X = scaledown(X_original)
X_df = pd.DataFrame(X, index=df.CASE_ID.unique())
x_train = X_df[:int(0.8*len(X))]
x_test = X_df[int(0.8*len(X)):]


# model definition
input_dim = X.shape[1]
encoding_dim = X.shape[1]
model = models.Sequential()

model.add(layers.Dense(encoding_dim, activation='tanh', input_shape=(input_dim,)))
model.add(layers.Dense(int(encoding_dim/2), activation='relu',
                       activity_regularizer=regularizers.l1_l2(l1=0.0005, l2=0.0005)))
model.add(layers.Dense(int(encoding_dim/4), activation='relu'))
model.add(layers.Dense(int(encoding_dim/4), activation='relu'))
model.add(layers.Dense(int(encoding_dim/2), activation='relu'))
model.add(layers.Dense(input_dim, activation='tanh'))


# compile model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train,
                    x_train, epochs=20, batch_size=64, validation_data=(x_test, x_test))

results = model.evaluate(x_test, x_test)

predicted = model.predict(x_test)

mean_squared_error = np.mean(np.power(x_test - predicted, 2), axis=1)
x_test['mse'] = mean_squared_error
# set those above 99.9% as outlier
x_test['outlier'] = 0
x_test.loc[x_test['mse'] > np.quantile(x_test['mse'], 0.999), 'outlier'] = 1

print("Results: ", results)
print("predicted :", predicted)
print("x_test outlier :", x_test)

# get original sequence steps
X_original_df.loc[x_test.loc[x_test['outlier'] == 1].index, :]


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
