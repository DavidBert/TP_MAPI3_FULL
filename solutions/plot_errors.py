errors_idx = np.where(y_pred != y_test)[0]
for idx in errors_idx:
    plt.figure()
    plt.imshow(scaler.inverse_transform(X_test[idx]).reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(f'real:{y_test[idx]} prediction:{y_pred[idx]}')
