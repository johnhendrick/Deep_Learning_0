
from keras.models import Sequential, Model
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers import Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from keras.optimizers import Nadam
from keras.utils import to_categorical
import os
import csv
import time
from datetime import datetime
from math import log
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

eventlog = 'helpdesk.csv'
file_path = os.path.join('..\data', eventlog)

df = pd.read_csv(file_path, encoding = 'utf-8')
df['CompleteTimestamp'] = pd.to_datetime(df['CompleteTimestamp'])

#scaledown
def scaledown(datain):
    scaler = MinMaxScaler()
    scaled_output = scaler.fit_transform(datain)
    return scaled_output

#convert timedelta value to seconds
def t_to_int(timestamp):
    num = int(timestamp.total_seconds())
    return num

#create sequence
#dateframe has to be sorted by CaseID and Timestamp
def create_seq(df, col_name, scaleminmax=False, rolling = False):
    prev = None
    encoded_seq = []
    time_seq = []
    time_seq2 = []
    time_seq3 = []
    time_seq4 = []
    
    new = []
    start_time = None
    last_event_time = None
    time_diff = []
    time_diff2 = []
    time_diff3 = []
    time_diff4 = []

    y_a = None
    y_t = None

    for i in range(len(df)):
        #handle first line
        if i == 0:        
            new.append(df.loc[i, col_name])
            time_diff = [0] #time since last event
            time_diff2 = [0] #time since case start
            time_diff3.append(t_to_int(df.loc[i,'CompleteTimestamp']- df.loc[i,'CompleteTimestamp'].normalize())) #seconds since midnight
            time_diff4.append(df.loc[i,'CompleteTimestamp'].dayofweek + 1) #day of the week. Monday =1
            start_time = df.loc[i,'CompleteTimestamp'] 

        # new case_id
        elif df.loc[i, "CaseID"] != prev and len(new) != 0:

            #stage to endcoded_seq and feature vectors
            encoded_seq.append(new)
            time_seq.append(time_diff)
            time_seq2.append(time_diff2)
            time_seq3.append(time_diff3)
            time_seq4.append(time_diff4)
            
            #start new encoded_seq and feature vectors
            new = []
            new.append(df.loc[i, col_name])
            start_time = df.loc[i,'CompleteTimestamp']
            time_diff = [0] #time since last event
            time_diff2 = [0] #time since case start
            time_diff3 = [t_to_int(start_time- start_time.normalize())] # seconds since midnight
            time_diff4 = [start_time.dayofweek + 1] #day of the week
            
            start_time = df.loc[i, 'CompleteTimestamp']     

        # still within same caseid
        else:
            new.append(df.loc[i, col_name])
            time_diff.append(t_to_int(df.loc[i,'CompleteTimestamp'] - last_event_time))
            time_diff2.append(t_to_int(df.loc[i,'CompleteTimestamp'] - start_time))
            time_diff3.append(t_to_int(df.loc[i,'CompleteTimestamp']- df.loc[i,'CompleteTimestamp'].normalize())) #seconds since midnight
            time_diff4.append(df.loc[i,'CompleteTimestamp'].dayofweek + 1) #day of the week
            
        prev = df.loc[i, "CaseID"]
        last_event_time = df.loc[i,'CompleteTimestamp']

    # append last case
    encoded_seq.append(new) 
    time_seq.append(time_diff)
    time_seq2.append(time_diff2)
    time_seq3.append(time_diff3)
    time_seq4.append(time_diff4)
    

    if rolling == True:
        seq = add_lag_window(encoded_seq,True)
        encoded_seq = to_categorical(seq[0]) #sequential events with OHE
        y_a = seq[1] # next event

        time_delta = add_lag_window(time_seq)
        time_seq = time_delta[0] #sequential time delta
        y_t = time_delta[1] # next time delta

        #supporting features
        time_seq2 = add_lag_window(time_seq2)[0]
        time_seq3 = add_lag_window(time_seq3)[0]
        time_seq4 = add_lag_window(time_seq4)[0]

    if scaleminmax == True:
        encoded_seq = scaledown(pad_sequences(encoded_seq , padding = 'post'))
        time_seq = scaledown(time_seq)
        time_seq2 = scaledown(time_seq2)
        time_seq3 = time_seq3/86400.0      #24hr
        time_seq4 = time_seq4/7            #7d
    else:   
        encoded_seq = pad_sequences(encoded_seq , padding = 'post')
        time_seq = time_seq
        time_seq2 = time_seq2
        time_seq3 = time_seq3
        time_seq4 = time_seq4
         
    return encoded_seq , time_seq, time_seq2, time_seq3, time_seq4, y_a , y_t

#assume ActivityID is encoded and df sorted by CaseID and timestamp
def create_seq2(df, col_name, scaleminmax=False, rolling = False):
    prev = None
    # y_a = np.zeros([len(df)])  # number of activity codes + end flag   # handle OHE at later stage
    # y_t = np.zeros([len(df)])
    y_a = []
    y_t = []
    extra_features = 5
    maxlen = df.groupby("CaseID").count().ActivityID.max()
    X = np.zeros([len(df), maxlen , len(df.ActivityID.unique()) + extra_features] )
         
    for i in range(len(df)):
        
        if df.loc[i, "CaseID"] != prev:
            # X[i-1, -1, -1] = 1 #mark last feature of preivous run as end of event
            X[i, -1 , 0] = 1  #mark first feature as start of event
            last_event_time = df.loc[i, 'CompleteTimestamp']
            start_time = df.loc[i, 'CompleteTimestamp']

        elif df.loc[i, "CaseID"] == prev:
            X[i, :-1,:] = X[i-1, 1:,:] #copy previous elements as lag -1 
                     
        #fill up lag 0 (latest event & features)
        X[i, -1 , df.loc[i, col_name]] = 1
        X[i, -1 , len(df.ActivityID.unique()) + 1] = t_to_int(df.loc[i,'CompleteTimestamp'] - last_event_time) #time since last event
        X[i, -1 , len(df.ActivityID.unique()) + 2] = t_to_int(df.loc[i,'CompleteTimestamp'] - start_time) #time since case start
        X[i, -1 , len(df.ActivityID.unique()) + 3] = t_to_int(df.loc[i,'CompleteTimestamp']- df.loc[i,'CompleteTimestamp'].normalize()) / 86400.0  # seconds since midnight
        X[i, -1 , len(df.ActivityID.unique()) + 4] = (df.loc[i,'CompleteTimestamp'].dayofweek + 1)  /7  #day of the week
 
        prev = df.loc[i, "CaseID"]
        last_event_time = df.loc[i,'CompleteTimestamp']
        
        try :
            if df.loc[i, "CaseID"] != df.loc[i+1, "CaseID"]:   #peek future CaseID. CaseID is different
                y_a.append(-1)
                y_t.append(0)
            else:  #future CaseID is the same as current
                y_a.append(df.loc[i + 1, col_name])  # add future event
                y_t.append( t_to_int(df.loc[i + 1,'CompleteTimestamp'] - last_event_time))
        except LookupError:  #to handle last event in dataframe
                y_a.append(-1)
                y_t.append(0)

    if scaleminmax == True:
        X[:,:,-3] = scaledown(X[:,:,-3])
        X[:,:,-4] = scaledown(X[:,:,-4])

    return X, y_a , y_t

def add_lag_window(cases, act_seq = False):
    matrix_rolling = []
    next_item = []
    for case in cases:
        for i in range(len(case)):     
            #append latest case
            if act_seq == True:
                matrix_rolling.append(case[i])

                #create next sequence
                if i != len(case) - 1:
                    next_item.append(case[i+1])
                #handle last case
                elif i == len(case) - 1:
                    next_item.append(-1)

            if act_seq == False:
                matrix_rolling.append(case[i])               
    return matrix_rolling, next_item




#create ndarray of sequences
seq_transform = create_seq(df, 'ActivityID', scaleminmax=False, rolling = False)
#activity sequence
df_seq = pd.DataFrame(seq_transform[0], index = df.CaseID.unique())
#time since last event
df_time_seq = pd.DataFrame(seq_transform[1], index = df.CaseID.unique())
#time since case start
df_time_seq2 = pd.DataFrame(seq_transform[2], index = df.CaseID.unique())
#seconds since midnight
df_time_seq3 = pd.DataFrame(seq_transform[3], index = df.CaseID.unique())
#day of week
df_time_seq4 = pd.DataFrame(seq_transform[4], index = df.CaseID.unique())

# #create ndarray of sequences with lag
# train_seq = create_seq(df, 'ActivityID', scaleminmax=False, rolling = True)
# #activity sequence
# train_df_seq = pd.DataFrame(train_seq[0])
# #time since last event
# train_df_time_seq = pd.DataFrame(train_seq[1])
# #time since case start
# train_df_time_seq2 = pd.DataFrame(train_seq[2])
# #seconds since midnight
# train_df_time_seq3 = pd.DataFrame(train_seq[3])
# #day of week
# train_df_time_seq4 = pd.DataFrame(train_seq[4])


# joined = np.stack((train_seq[0], train_seq[1], train_seq[2], train_seq[3], train_seq[4]))
# #swap axis in the format samples-timesteps-features
# X = joined.swapaxes(0,1).swapaxes(1,2)

# #next activity
# # -1 for last event
# y_a = to_categorical(np.array(train_seq[5]) - np.array(train_seq[5]).min()) 
# y_t = np.array(train_seq[6])

all = create_seq2(df, 'ActivityID')
X = all[0]
y_a = to_categorical(np.array(all[1]) - np.array(all[1]).min()) 
y_t = np.array(all[2])
maxlen= int(X.shape[-2])



#Build LSTM model
main_input = Input(shape=(X.shape[-2], X.shape[-1]), name='main_input')
# train a 2-layer LSTM with one shared layer
l1 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=True, dropout=0.2)(main_input) # the shared layer
b1 = BatchNormalization()(l1)
l2_1 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(b1) # the layer specialized in activity prediction
b2_1 = BatchNormalization()(l2_1)
l2_2 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(b1) # the layer specialized in time prediction
b2_2 = BatchNormalization()(l2_2)
act_output = Dense(y_a.shape[1], activation='softmax', kernel_initializer='glorot_uniform', name='act_output')(b2_1)
time_output = Dense(1, kernel_initializer='glorot_uniform', name='time_output')(b2_2)

model = Model(inputs=[main_input], outputs=[act_output, time_output])

opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)

model.compile(loss={'act_output':'categorical_crossentropy', 'time_output':'mae'}, optimizer=opt, metrics={'act_output':['categorical_accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.AUC()]})
early_stopping = EarlyStopping(monitor='val_loss', patience=42)
#model_checkpoint = ModelCheckpoint('output_files/models/model_{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

history = model.fit(X, {'act_output':y_a, 'time_output':y_t}, validation_split=0.2, verbose=2, callbacks=[early_stopping, lr_reducer], batch_size=maxlen, epochs=500)


#understand the shape
print(model.summary())


#understand the shape
[(layer.output) for layer in model.layers]

#keys
history.history.keys()

# acc = history.history['acc']
# val_acc = history.history['val_acc']
a_loss = history.history['act_output_loss']
a_val_loss = history.history['val_act_output_loss']
t_loss = history.history['time_output_loss']
t_val_loss = history.history['val_time_output_loss']
a_acc = history.history['act_output_categorical_accuracy']
a_val_acc = history.history['val_act_output_categorical_accuracy']
a_auc = history.history['act_output_auc_1']
a_val_auc = history.history['val_act_output_auc_1']
a_precision = history.history['act_output_precision_1']
a_val_precision = history.history['val_act_output_precision_1']

epochs = range(1, len(a_loss) + 1)
plt.figure().set_size_inches(10, 5.5)
plt.plot(epochs, a_loss, 'ko', label='Training loss', markersize = 3)
plt.plot(epochs, a_val_loss, color = '#F25D18', linewidth = 3, label='Validation loss')
plt.title('Next Activity: Training and Validation Loss', fontsize = 14.5)
plt.legend()

plt.figure().set_size_inches(10, 5.5)
plt.plot(epochs, t_loss, 'ko', label='Training loss', markersize = 3)
plt.plot(epochs, t_val_loss, color = '#F25D18', linewidth = 3, label='Validation loss')
plt.title('Time Delta: Training and Validation Loss', fontsize = 14.5)
plt.legend()

plt.figure().set_size_inches(10, 5.5)
plt.plot(epochs, a_acc, 'ko', label='Training Score', markersize = 3)
plt.plot(epochs, a_val_acc, color = '#F25D18', linewidth = 3, label='Validation Score')
plt.title('Next Activity Training and Validation Accuracy Score', fontsize = 14.5)
plt.legend()

plt.figure().set_size_inches(10, 5.5)
plt.plot(epochs, a_auc, 'ko', label='Training Score', markersize = 3)
plt.plot(epochs, a_val_auc, color = '#F25D18', linewidth = 3, label='Validation Score')
plt.title('Next Activity: Training and Validation AUC Score', fontsize = 14.5)
plt.legend()

plt.figure().set_size_inches(10, 5.5)
plt.plot(epochs, a_precision, 'ko', label='Training Score', markersize = 3)
plt.plot(epochs, a_val_precision, color = '#F25D18', linewidth = 3, label='Validation Score')
plt.title('Next Activity: Training and Validation Precision Score', fontsize = 14.5)
plt.legend()


# plt.plot(epochs, loss, 'ko', label='Training loss')
# plt.plot(epochs, val_loss, color = '#F25D18', linewidth = 3 , label='Validation loss')
plt.show()

print('best accuracy: ',max(a_val_acc))
print(len(a_val_acc))
print('best precision: ',max(a_val_precision))
print(len(a_val_precision))
print('best AUC: ',max(a_val_auc))
print(len(a_val_auc))