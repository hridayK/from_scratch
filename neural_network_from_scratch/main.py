from audioop import bias
import numpy as np

inputs = [1.2, 5.1, 2.1]
weights = [[3.1, 2.1, 8.7], [0.5, -0.91, -0.5], [-0.26, -0.27, 0.17]]
biases = [3, 2, 0.5]

output = np.dot(weights, inputs) + biases
print(output)