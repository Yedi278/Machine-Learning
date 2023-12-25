import numpy as np

def init_param(layer_size,image_lenght):

    W1 = np.random.rand(layer_size,image_lenght) -.5
    b1 = np.random.rand(layer_size) -.5

    W2 = np.random.rand(layer_size,layer_size) -.5
    b2 = np.random.rand(layer_size) -.5

    W3 = np.random.rand(10,layer_size) -.5
    b3 = np.random.rand(10) -.5
    
    return W1,b1,W2,b2,W3,b3

def sigmoid(Z):
    return np.exp(Z) /( np.ones(Z.shape) + np.exp(Z))

def fwd_prop(w1,b1,w2,b2,w3,b3,a0):
    
    z1 = w1.dot(a0) + b1
    a1 = sigmoid(z1)

    z2 = w2.dot(a1) + b2
    a2 = sigmoid(z2)

    z3 = w3.dot(a1)+b3
    a3 = sigmoid(z3)

    return z1,a1,z2,a2,z3,a3

def cost(X,Y):
    return np.sum(np.power(X-Y,2))

def true_val(data,index):

    Y = np.zeros(10)
    Y[data[0,index]] = 1

    return Y