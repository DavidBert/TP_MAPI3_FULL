def accuracy(net, X, y):
    y_pred = net.forward(X)
    return accuracy_score(np.argmax(y_pred, axis=1), y)


epoch_train_acc = []
epoch_test_acc = []
for epoch in range(15):
    fit_one_epoch(X_train, y_train)
    epoch_train_acc.append(accuracy(net, X_train, y_train))
    epoch_test_acc.append(accuracy(net, X_test, y_test))
    
plt.plot(epoch_train_acc)
plt.plot(epoch_test_acc)
