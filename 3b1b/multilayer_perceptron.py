import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from imaging import print_image

# one_image = data[1:, index ]


data_frame = pd.read_csv("train.csv")
data = data_frame.to_numpy(np.uint8).T

image_lenght, n_images = data[1:,:].shape



def init_param(layer_size,image_lenght):

    W1 = np.random.rand(layer_size,image_lenght) -.5
    b1 = np.random.rand(layer_size) -.5

    W2 = np.random.rand(layer_size,layer_size) -.5
    b2 = np.random.rand(layer_size) -.5

    Wo = np.random.rand(10,layer_size) -.5
    bo = np.random.rand(10) -.5
    
    return W1,b1,W2,b2,Wo,bo


def sigmoid(Z):
    return np.exp(Z) /( np.ones(Z.shape) + np.exp(Z))

def fwd_prop(w1,b1,w2,b2,wo,bo,a0):

    z1 = sigmoid(w1.dot(a0) + b1)

    z2 = sigmoid(w2.dot(z1) + b2)

    out = sigmoid(wo.dot(z1)+bo)

    return out

def cost(X,Y):

    return np.sum(np.power(X-Y,2))

def true_val(data,index):

    Y = np.zeros(10)
    Y[data[0,index]] = 1

    return Y

if __name__ == '__main__':

    index = 3
    image = data[1:, index]/255
    label = data[0,index]

    w1,b1,w2,b2,wo,bo = init_param(16,image_lenght)

    output = fwd_prop(w1,b1,w2,b2,wo,bo,image)

    Y = true_val(data,index)
    print(label)
    c = cost(output,Y)
    print(c)