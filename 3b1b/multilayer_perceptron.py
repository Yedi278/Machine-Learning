import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from imaging import print_image
import grad_descent as gd
# one_image = data[1:, index ]


data_frame = pd.read_csv("train.csv")
data = data_frame.to_numpy(np.uint8).T

image_lenght, n_images = data[1:,:].shape

def run(w1,b1,w2,b2,w3,b3,a0,y):
    
    output = gd.fwd_prop(w1,b1,w2,b2,w3,b3,a0)
    c = gd.cost(a0,y)


if __name__ == '__main__':

    index = 3
    image = data[1:, index]/255
    label = data[0,index]

    w1,b1,w2,b2,wo,bo = gd.init_param(16,image_lenght)

    