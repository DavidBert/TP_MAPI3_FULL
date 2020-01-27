class MSE():

    def loss(self, y_pred, y_true):
        #returns a scalar
        return np.mean((y_pred - y_true) ** 2)
    
    def grad(self, y_pred, y_true):
        #returns a tensor of gradients
        return 2 * (y_pred - y_true)
