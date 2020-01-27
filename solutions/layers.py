
class Linear(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        #weights is a tensor of dimension: (input_size x output_size) initialized...
        #biais is a tensor of dimension: (output_size)
        self.layer_type = 'linear'
        self.params ={}
        self.grads = {}
        self.params["weights"] = np.random.uniform(size=(input_size, output_size), high=0.1, low=-0.1)
        self.params["bias"] = np.random.uniform(size=output_size, high=0.1, low=-0.1)

    def forward(self, inputs):
        # outputs = inputs @ weights + biais
        #we will store the inputs they will be used in the backward method
        self.inputs = inputs
        return np.matmul(inputs, self.params["weights"]) + self.params["bias"]
    
    def backward(self, grad):
        # we need so sum gradients over the batch axis
        self.grads["weights"] = np.matmul(self.inputs.T, grad)
        self.grads["bias"] = np.sum(grad, axis=0)
        return np.matmul(grad, self.params["weights"].T)
