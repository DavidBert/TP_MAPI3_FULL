class Activation(Layer):
    def __init__(self, f, f_prime):
        super().__init__()
        self.layer_type = 'activation'
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs):
        self.intputs = inputs
        return self.f(inputs)
    
    def backward(self, grad):
        """ if y = f(x) and x = g(z)
        then dy/dz = f'(x) * g'(z)
        In our case g'(z) correspond to the incomming gradients
        """
        return self.f_prime(self.intputs) * grad
