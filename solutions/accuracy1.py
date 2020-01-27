from sklearn.metrics import accuracy_score

y_pred = net.forward(X_test)
print(f'accuracy:{accuracy_score(np.argmax(y_pred, axis=1), y_test)}')
