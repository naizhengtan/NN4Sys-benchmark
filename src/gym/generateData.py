# Copyright 2019 Nathan Jay and Noga Rotman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# PACKAGES AND DIRECTORIES
import gym
import pickle 
import network_sim
import tensorflow as tf
print (tf.__version__)


from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines import PPO1
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from common.simple_arg_parse import arg_or_default

# from tensorflow.python.tools.freeze_graph import freeze_graph 

arch_str = arg_or_default("--arch", default="32,16")
if arch_str == "":
    arch = []
else:
	# 32 width layer and 16 width layer. 2 layers
    arch = [int(layer_width) for layer_width in arch_str.split(",")]
print("Architecture is: %s" % str(arch))

training_sess = None

class MyMlpPolicy(FeedForwardPolicy):

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(MyMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse, net_arch=[{"pi":arch, "vf":arch}],
                                        feature_extraction="mlp", **_kwargs)
        global training_sess
        training_sess = sess

env = gym.make('PccNs-v0')

gamma = arg_or_default("--gamma", default=0.99)
print("gamma = %f" % gamma)
model = PPO1(MyMlpPolicy, env, verbose=1, schedule='constant', timesteps_per_actorbatch=8192, optim_batchsize=2048, gamma=gamma)

for i in range(0, 1):
    with model.graph.as_default():                                                                   
        saver = tf.train.Saver()                                                                     
        saver.save(training_sess, "./data/pcc_model_%d.ckpt" % i)
    model.learn(total_timesteps=(1600 * 410))

# Use the model to generate data
inputs = []
outputs = []

obs = env.reset()
print (obs)
action, _states = model.predict(obs)
count = 0

while count < 5000:
    # Convert np array to list and store input
    inputs.append(obs)

    action, _states = model.predict(obs)

    # Convert np array to list and store output 
    outputs.append(action)

    obs, rewards, dones, info = env.step(action)

    count += 1



# with open('./tensorflow/inputs.pkl', 'wb') as f:
#     pickle.dump(inputs, f)

# with open('./tensorflow/outputs.pkl', 'wb') as f:
#     pickle.dump(outputs, f)
