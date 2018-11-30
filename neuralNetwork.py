# General Neural Network unit that works with almost all types of Data, Please be sure to specify the nuber of hidden layers and the neurons and the activation functions and all that yourself.
# Also, change the general 'feed_dict' to specify the Data you want it to flow through.

'''Importing all the Modules, Numpy isn't imported, because it is not necessary in front of Tensorflow'''

import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt


# Declaring Hyperparameters, Be sure to change them according to your needs

learning_rate = 0.01
epochs = 100
input_dim = 28*28 #Dimensions of the Input Data
hidden_layer_one = 512 # Number of neurons in the first Hidden layer
hidden_layer_two = 512 # Number of neurons in the Second Hidden layer
hidden_layer_three = 512 # Number of neurons in the Third Layer
output_dim = 10 # Number of the classes we want to predict

# Input Placeholders 
X = tf.placeholder(tf.float32,[None,input_neuron])
Y = tf.placeholder(tf.float32,[None,output_neuron])

# Hidden Layer and Output Layer Weights Initialization
hidden_one_w = tf.Variable(tf.random_uniform([input_neuron,hidden_layer_one],seed=12))
hidden_two_w = tf.Variable(tf.random_uniform([hidden_layer_one,hidden_layer_two],seed=12))
hidden_three_w = tf.Variable(tf.random_uniform([hidden_layer_two,hidden_layer_three],seed=12))
output_w = tf.Variable(tf.random_uniform([hidden_layer_three,output_neuron],seed=12))

# Hidden Layer and Output Layer Biases Initialization
hidden_one_bias = tf.Variable(tf.random_uniform([hidden_layer_one],seed=12))
hidden_two_bias = tf.Variable(tf.random_uniform([hidden_layer_two],seed=12))
hidden_three_bias = tf.Variable(tf.random_uniform([hidden_layer_three],seed=12))
output_bias = tf.Variable(tf.random_uniform([output_neuron],seed=12))

# Main Computation Graph for Matrix Multiplication and Addition
h_1 = tf.add(tf.matmul(X,hidden_one_w),hidden_one_bias)
h_2 = tf.add(tf.matmul(h_1,hidden_two_w),hidden_two_bias)
h_3 = tf.add(tf.matmul(h_2,hidden_three_w),hidden_three_bias)
out = tf.add(tf.matmul(h_3,output_w),output_bias)

# Costs and Optimzers
cost = (out-Y)**2
optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)

# Variables Initialization
init = tf.global_variables_initializer()

# Running the Session
with tf.Session() as sess:
	sess.run(init)
	for i in range(epochs):
		c = sess.run([optimizer,cost]), feed_dict={X: X_data, Y: Y_data} # Put your data here in feed_dict
		# Some more code according to your needs
		# Yeah, more code here