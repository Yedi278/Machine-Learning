import numpy as np

def init_param(layer_size,image_lenght):

    W1 = np.random.rand(layer_size,image_lenght) -.5
    b1 = np.random.rand(layer_size) -.5

    W2 = np.random.rand(10,layer_size) -.5
    b2 = np.random.rand(10) -.5

    # W3 = np.random.rand(10,layer_size) -.5
    # b3 = np.random.rand(10) -.5
    
    return W1,b1,W2,b2

def sigmoid(Z):
    return np.exp(Z) /( np.ones(Z.shape) + np.exp(Z))

def der_sigmoid(Z):

    s = sigmoid(Z)
    return s*(np.ones(s.shape)-s)

def fwd_prop(a0,w1,b1,w2,b2,w3=0,b3=0):
    
    z1 = w1.dot(a0) + b1
    a1 = sigmoid(z1)

    z2 = w2.dot(a1) + b2
    a2 = sigmoid(z2)

    # z3 = w3.dot(a1)+b3
    # a3 = sigmoid(z3)

    return z1,a1,z2,a2

def cost(X,Y):
    return np.sum(np.power(X-Y,2))

def true_val(data,index):

    Y = np.zeros(10)
    Y[data[0,index]] = 1

    return Y

def bck_prop(a0,a1,a2,z1,z2,w1,w2,b1,b2,Y,alpha):

    C = Y - a2

    grd_b2 = der_sigmoid(z2)
    grd_w2 = grd_b2*(a1)

    grd_b1 = grd_b2*(der_sigmoid(z1))*(w2)
    grd_w1 = grd_b1*(a0)

    dw2 = C/grd_w2
    db2 = C/grd_b2
    dw1 = C/grd_w1
    db1 = C/grd_b1

    w1 -= alpha*dw1
    b1 -= alpha*db1
    w2 -= alpha*dw2
    b2 -= alpha*db2

    return w1,b1,w2,b2

if __name__ == '__main__':

    pass