import tensorflow as tf
import numpy as np 
import pickle

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import sklearn, pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
print (tf.__version__)


# Model 
model = Sequential()
model.add(layers.Dense(16, input_dim=30, activation= "relu"))
model.add(layers.Dense(16, activation= "relu"))
model.add(layers.Dense(1))


# Data
with open('./data/inputs.pkl', 'rb') as f:
	x_train = pickle.load(f)
	x_train = np.array(x_train)

with open('./data/outputs.pkl', 'rb') as f:
	y_train = pickle.load(f)
	y_train = np.array(y_train)


# Splitting
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.30, random_state=40)
print(x_train.shape); print(x_test.shape)


# Training
model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])
model.fit(x_train, y_train, epochs=20)

pred_train= model.predict(x_train)
print("Training MSE" + str(np.sqrt(mean_squared_error(y_train,pred_train))))

pred= model.predict(x_test)
print("Test MSE" + str(np.sqrt(mean_squared_error(y_test,pred))))

weights = [layer.get_weights() for layer in model.layers]
print (len(weights))

with open('./reLuNet/sup_weights.pkl', 'wb') as f:
	pickle.dump(weights, f)

bias = [layer.get_weights()[1] for layer in model.layers]
print (len(bias))
with open('./reLuNet/sup_bias.pkl', 'wb') as f:
	pickle.dump(bias, f)



# Saving
model.save("./reLuNet/my_model")