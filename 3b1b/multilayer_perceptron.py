import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from imaging import print_image
import grad_descent as gd
# one_image = data[1:, index ]


data_frame = pd.read_csv(r"C:\Users\Yehan\Documents\Python\Machine-Learning\3b1b\train.csv")
data = data_frame.to_numpy(np.uint8).T

image_lenght, n_images = data[1:,:].shape



if __name__ == '__main__':

    index = 3
    image = data[1:, index]/255
    label = data[0,index]

    w1,b1,w2,b2 = gd.init_param(10,image_lenght)

    z1,a1,z2,a2 = gd.fwd_prop(image,w1,b1,w2,b2)

    Y = gd.true_val(data,index)

    w1,b1,w2,b2 = gd.bck_prop(image,a1,a2,z1,z2,w1,w2,b1,b2,Y,.1)
    
    print(w1,b1,w2,b2)