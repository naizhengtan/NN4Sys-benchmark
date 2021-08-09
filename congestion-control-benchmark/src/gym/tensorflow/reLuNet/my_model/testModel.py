# load and evaluate a saved model
from numpy import loadtxt
from tensorflow.keras.models import load_model
import tensorflow as tf

print(tf.__version__)
model = tf.keras.models.load_model('../my_modelR')

# Test on all zeroed input
print (model.predict([[0,0,0,
					  0,0,0,
					  0,0,0,
					  0,0,0,
					  0,0,0,
					  0,0,0,
					  0,0,0,
					  0,0,0,
					  0,0,0,
					  0,0,0]]))

# Output:: [[0.0990748]]

# Test on very high input
print (model.predict([[100,100,100,
					  100,100,100,
					  100,100,100,
					  100,100,100,
					  100,100,100,
					  100,100,100,
					  100,100,100,
					  100,100,100,
					  100,100,100,
					  100,100,100]]))

# Output:: [[-19.428347]]

# Test on very high low input
print (model.predict([[-100,-100,-100,
					  -100,-100,-100,
					  -100,-100,-100,
					  -100,-100,-100,
					  -100,-100,-100,
					  -100,-100,-100,
					  -100,-100,-100,
					  -100,-100,-100,
					  -100,-100,-100,
					  -100,-100,-100]]))

# Output:: [[160.62288]]


print (model.predict([[10.0, 35.0, 150.0,
       10.0, 35.0, 250.0,
       10.0, 35.0, 350.0,
       10.0, 35.0, 400.0,
       10.0, 35.0, 550.0,
       10.0, 35.0, 650.0,
       10.0, 35.0, 700.0,
       10.0, 35.0, 750.0,
       10.0, 35.0, 800.0,
       10.0, 35.0, 1000.0]]))
