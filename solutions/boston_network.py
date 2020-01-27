layer_1 = Linear(input_size=13, output_size=40)
activation_1 = Activation(f=relu, f_prime=relu_prime)
layer_2 = Linear(input_size=40, output_size=40)
activation_2 = Activation(f=relu, f_prime=relu_prime)
layer_3 = Linear(input_size=40, output_size=1)

layers = [layer_1, activation_1, layer_2, activation_2, layer_3]
net = NeuralNetwork(layers)