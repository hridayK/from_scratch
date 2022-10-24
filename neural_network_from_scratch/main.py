from audioop import bias


inputs = [1.2, 5.1, 2.1]
weights = [[3.1, 2.1, 8.7], [0.5, -0.91, -0.5], [-0.26, -0.27, 0.17]]
biases = [3, 2, 0.5]

layer_outputs = []
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += (n_input*weight)
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)
print(layer_outputs)