import numpy as np

class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

inputs = [[1.2, 5.1, 2.1, 1.6],
            [2.0, 5.0, -1.0, 2.0],
            [-1.5, -0.27, 0.17, 0.87]]

layer_one = Layer_Dense(4,5)
layer_two = Layer_Dense(5,3)
layer_three = Layer_Dense(3,1)

print(f"On giving\n{inputs}\nas input to layer one we get:\n")
layer_one.forward(inputs)
print(f"{layer_one.output}\nas output\n\n")     
print(f"On giving\n{layer_one.output}\ninput to layer two we get:\n")
layer_two.forward(layer_one.output)
print(f"{layer_two.output}\nas output\n\n")
print(f"On giving\n{layer_two.output}\nas input to layer one we get:\n")
layer_three.forward(layer_two.output)
print(f"{layer_three.output}\nas output\n\n")