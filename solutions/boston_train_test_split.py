
from sklearn.model_selection import train_test_split
import numpy as np

data = np.asarray(dataset.data, dtype='float32')
target = np.asarray(dataset.target, dtype='float32')

X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.15, random_state=37)
