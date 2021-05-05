# *****************************************************************
# Start Neuron Layer Class
#
# Class represented each layer of artificial neural network. 
# There are many nodes in each layer.
# *****************************************************************
import neuron
class NeuronLayer:
    def __init__(self, num_neurons, bias):

        # Each neuron in a layer shares the same bias
        self.bias = bias if bias else random.random()

        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(neuron.Neuron(self.bias))

    def inspect(self):
        #print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print('  Weight:', self.neurons[n].weights[w])
            print('  Bias:', self.bias)

    def feed_forward(self, inputs):
        outputs = []
        #print("number of Neuron: ", len(self.neurons))
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs

    def get_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs

# ******* End of Neuron Layer Class ******* #
# ***************************************** #