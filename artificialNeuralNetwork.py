# *******************************************************************
# Start of Aritificial Neural Netowrk Class
#
# This class represente a whole nueral network tha contain all layer
# of node and essential methods in calculating ANN.
# *******************************************************************
import random
import neuronLayer
class ArtificialNeuralNetwork:
    LEARNING_RATE = 0.5

    def __init__(self, num_inputs, num_hidden, num_outputs, num_layers = 1, hidden_layer_weights = None, hidden_layer_bias = None, output_layer_weights = None, output_layer_bias = None):
        self.num_inputs = num_inputs
        self.num_layers = num_layers

        self.hidden_layer = [0] * num_layers
        for l in range(num_layers):
            self.hidden_layer[l] = neuronLayer.NeuronLayer(num_hidden, hidden_layer_bias)
        self.init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights)
        #print(self.hidden_layer[0].neurons[0].weights)

        self.output_layer = neuronLayer.NeuronLayer(num_outputs, output_layer_bias)
        self.init_weights_from_hidden_layer_neurons_to_output_layer_neurons(output_layer_weights)

    # Method used to initialize weights for neuron of each network layer 
    def init_weights_from_inputs_to_hidden_layer_neurons(self, hidden_layer_weights):
        for l in range(self.num_layers):
            weight_num = 0
            for h in range(len(self.hidden_layer[l].neurons)):
                if l == 0 :
                    for i in range(self.num_inputs):
                        if not hidden_layer_weights:
                            self.hidden_layer[l].neurons[h].weights.append(random.random())
                        else:
                            self.hidden_layer[l].neurons[h].weights.append(hidden_layer_weights[weight_num])
                        weight_num += 1
                else :
                    for i in range(len(self.hidden_layer[l].neurons)):
                        if not hidden_layer_weights:
                            self.hidden_layer[l].neurons[h].weights.append(random.random())
                        else:
                            self.hidden_layer[l].neurons[h].weights.append(hidden_layer_weights[weight_num])
                        weight_num += 1

    # Method used to initialize weight for each neuron of output layer
    def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(self, output_layer_weights):
            weight_num = 0
            for o in range(len(self.output_layer.neurons)):
                for h in range(len(self.hidden_layer[len(self.hidden_layer)-1].neurons)):
                    if not output_layer_weights:
                        self.output_layer.neurons[o].weights.append(random.random())
                    else:
                        self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
                    weight_num += 1

    def inspect(self):
        print('------')
        print('* Inputs: {}'.format(self.num_inputs))
        print('------')
        for l in range(self.num_layers):
            print('Hidden Layer: {}'.format(l))
            self.hidden_layer[l].inspect()
        print('------')
        print('* Output Layer')
        self.output_layer.inspect()
        print('------')

    # Method call to calculate feed forward with multilayer neural network
    def feed_forward(self, inputs):
        #print(inputs)
        layer_feed_forward_results = [0] * self.num_layers
        for l in range(self.num_layers):
            if (l == 0) :
                #print("OK!")
                layer_feed_forward_results[l] = self.hidden_layer[l].feed_forward(inputs)
                #print(layer_feed_forward_results[l])
            else:
                #print("Ok2!: ", l)
                layer_feed_forward_results[l] = self.hidden_layer[l].feed_forward(layer_feed_forward_results[l-1])
                #print(layer_feed_forward_results[l])
        return self.output_layer.feed_forward(layer_feed_forward_results[len(self.hidden_layer)-1])

    # Method for back propagation calculation process
    # Uses online learning, ie updating the weights after each training case
    def train(self, training_inputs, training_outputs):
        #print(training_inputs)
        self.feed_forward(training_inputs)

        # 1. Output neuron deltas
        # Calculation of ∂E/∂Netⱼ (∂E/∂yⱼ) of each node in output layer
        partial_derivative_errors_with_respect_to_output_neuron_total_net_input = [0] * len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):
            partial_derivative_errors_with_respect_to_output_neuron_total_net_input[o] = self.output_layer.neurons[o].calculate_partial_derivative_error_with_respect_to_total_net_input(training_outputs[o])

        # 2. Hidden neuron deltas
        # Calculation of ∂E/∂Netⱼ (∂E/∂yⱼ) of each node of each hidden layer
        partial_derivative_errors_with_respect_to_hidden_neuron_total_net_input = [0] * self.num_layers
        for l in range(self.num_layers):
            partial_derivative_errors_with_respect_to_hidden_neuron_total_net_input[l] = [0] * len(self.hidden_layer[l].neurons)
            for h in range(len(self.hidden_layer[l].neurons)):
                # We need to calculate the derivative of the error with respect to the output of each hidden layer neuron
                # dE/dyⱼ = Σ ∂E/∂Netⱼ * ∂z/∂yⱼ = Σ ∂E/∂Netⱼ * wᵢⱼ
                derivative_error_with_respect_to_hidden_neuron_output = 0
                for o in range(len(self.output_layer.neurons)):
                    derivative_error_with_respect_to_hidden_neuron_output += partial_derivative_errors_with_respect_to_output_neuron_total_net_input[o] * self.output_layer.neurons[o].weights[h]
                # ∂E/∂Netⱼ = dE/dyⱼ * ∂Netⱼ/∂
                partial_derivative_errors_with_respect_to_hidden_neuron_total_net_input[l][h] = derivative_error_with_respect_to_hidden_neuron_output * self.hidden_layer[l].neurons[h].calculate_partial_derivative_total_net_input_with_respect_to_input()

        # 3. Update output neuron weights
        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):
                # ∂Eⱼ/∂wᵢⱼ = ∂E/∂Netⱼ * ∂Netⱼ/∂wᵢⱼ
                partial_derivative_error_with_respect_to_weight = partial_derivative_errors_with_respect_to_output_neuron_total_net_input[o] * self.output_layer.neurons[o].calculate_partial_derivative_total_net_input_with_respect_to_weight(w_ho)
                # Δw = α * ∂Eⱼ/∂wᵢ
                self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * partial_derivative_error_with_respect_to_weight

        # 4. Update hidden neuron weights
        for l in range(self.num_layers):
            for h in range(len(self.hidden_layer[l].neurons)):
                for w_ih in range(len(self.hidden_layer[l].neurons[h].weights)):
                    # ∂Eⱼ/∂wᵢ = ∂E/∂Netⱼ * ∂zⱼ/∂wᵢ
                    partial_derivative_error_with_respect_to_weight = partial_derivative_errors_with_respect_to_hidden_neuron_total_net_input[l][h] * float(self.hidden_layer[l].neurons[h].calculate_partial_derivative_total_net_input_with_respect_to_weight(w_ih))
                    # Δw = α * ∂Eⱼ/∂wᵢ
                    self.hidden_layer[l].neurons[h].weights[w_ih] -= self.LEARNING_RATE * partial_derivative_error_with_respect_to_weight

    def calculate_total_error(self, training_sets):
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.feed_forward(training_inputs)
            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].calculate_error(training_outputs[o])
        return total_error

# ******* End of Artificial Neural Network Class ******* #
# ****************************************************** #