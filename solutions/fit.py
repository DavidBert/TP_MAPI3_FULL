def fit(net, loss, optimizer, X, y):
    y_pred = net.forward(X)
    prediction_loss = loss.loss(y_pred, y)
    grad = loss.grad(y_pred, y)
    net.backward(grad)
    optimizer.step(net)
    return prediction_loss
