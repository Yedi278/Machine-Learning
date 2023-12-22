import numpy as np
import pandas as pd

file = '/home/peppo/Documents/Machine-Learning/train.csv'

data = pd.read_csv(file)
data = np.array(data)


print(data.shape)
print(data)