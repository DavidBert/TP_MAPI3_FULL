layer1 = Linear(input_size=64, output_size=10)
layer2 = Activation(f=relu, f_prime=relu_prime)
layer3 = Linear(input_size=10, output_size=10)

layers = [layer1, layer2, layer3]
loss = CrossEntropy()
net = NeuralNetwork(layers)
optimizer = SGD(lr=0.05)
