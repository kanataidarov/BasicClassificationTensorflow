from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf 

print('Tensorflow version: ', tf.__version__)

# import mnist dataset 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# shapes of imported arrays 
print('x_train shape: ', x_train.shape)
print('y_train shape: ', y_train.shape)
print('x_test shape: ', x_test.shape)
print('y_test shape: ', y_test.shape)

# plot sample dataset
from matplotlib import pyplot as plt 

# plt.imshow(x_train[0], cmap='binary')
# plt.xlabel('Number = %d' % y_train[0])
# plt.show() 

# display labels 
print(set(y_train))

# One Hot encoding 
# 5 encoded to [0,0,0,0,0,1,0,0,0,0]
y_train_encoded = tf.keras.utils.to_categorical(y_train)
y_test_encoded = tf.keras.utils.to_categorical(y_test)

print('y_train_encoded shape = ', y_train_encoded.shape)
print('y_test_encoded shape = ', y_test_encoded.shape)

# Preprocessing: convert n-dim array to vector 
import numpy as np 
from functools import reduce
import operator 
num_inputs_flattened = reduce(operator.mul, list(x_train.shape)[1:])
x_train_reshaped = np.reshape(x_train, (x_train.shape[0], num_inputs_flattened))
x_test_reshaped = np.reshape(x_test, (x_test.shape[0], num_inputs_flattened))

print('x_train_reshaped shape = ', x_train_reshaped.shape)
print('x_test_reshaped shape = ', x_test_reshaped.shape)

# Data normalization
# print('Before normalization = ', set(x_train_reshaped[0]))
x_mean = np.mean(x_train_reshaped)
x_std = np.std(x_train_reshaped)

x_train_norm = (x_train_reshaped - x_mean) / (x_std + 1e-10)
x_test_norm = (x_test_reshaped - x_mean) / (x_std + 1e-10)
# print('After normalization = ', set(x_train_norm[0]))

# Creation of the Model 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 

model = Sequential([
    # relu - linear function for all positive values and 0 for all negative values 
    Dense(128, input_shape=(num_inputs_flattened,), activation='relu'), 
    Dense(128, activation='relu'),
    # softmax - probability scores for all 10 nodes (sum=1)
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='sgd',
    # loss - difference between actual and predicted output must be minimized 
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary() 

# traning model 
epochs = 3 
model.fit(x_train_norm, y_train_encoded, epochs)
_, accuracy = model.evaluate(x_test_norm, y_test_encoded)
print('Model accuracy on test set: ', accuracy*100, ' %')

# Make predictions 
preds = model.predict(x_test_norm)
print('Shape of predictions output: ', preds.shape)

# Plotting results 
plt.figure(figsize=(12, 12))

start_idx = 0

for i in range(25): 
    plt.subplot(5, 5, i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    prediction = np.argmax(preds[start_idx+i])
    ground_truth = y_test[start_idx+i]

    clr = 'g' if prediction == ground_truth else 'r'

    plt.xlabel('i={}, p={}, gt={}'.format(start_idx+i, prediction, ground_truth), color=clr)
    plt.imshow(x_test[start_idx+i], cmap='binary')
plt.show()

plt.plot(preds[8])
plt.show()
