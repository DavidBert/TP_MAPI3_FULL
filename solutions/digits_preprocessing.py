from sklearn import preprocessing
from sklearn.model_selection import train_test_split

data = np.asarray(dataset.data, dtype='float32')
target = np.asarray(dataset.target, dtype='int32')

X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.15, random_state=37)

# mean = 0 ; standard deviation = 1.0
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
