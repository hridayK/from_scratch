from audioop import bias
import numpy as np

inputs = [[1.2, 5.1, 2.1, 1.6],
            [2.0, 5.0, -1.0, 2.0],
            [-1.5, -0.27, 0.17, 0.87]]
weights = [[3.1, 2.1, 8.7, -0.35], 
            [0.5, -0.91, -0.5, 1.1], 
            [-0.26, -0.27, 0.17, 4.3]]
biases = [3, 2, 0.5]

output = np.dot(np.array(inputs), np.array(weights).T) + biases
print(output)