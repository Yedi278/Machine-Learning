import cv2
import numpy as np

def print_image(data,index,resize_dim=(300,300)):

    image = data[1:,index]

    a = np.zeros((28,28))

    for i in range(28):
        for j in range(28):

            a[i,j] = image[(i*28)+j]

    a = cv2.resize(a, resize_dim, interpolation = cv2.INTER_AREA)

    cv2.imshow("image",a)
    cv2.waitKey(0)
    cv2.destroyAllWindows()