# *****************************************************************
# Start of Neuron Class
#
# Class represented each neuron node of artificial neural network.
# *****************************************************************
class Neuron:

    def __init__(self, bias):
        self.bias = bias
        self.weights = []

    # Method to calculate final output of each neuron node
    # This method used an activation function called sigmoid function
    def calculate_output(self, inputs):
        self.inputs = inputs
        self.output = self.sigmoid(self.calculate_total_net_input())
        return self.output

    # Method to calculate weighted sum of each neuron node
    # Following the formular yᵢ = Σxᵢwᵢ + bias
    def calculate_total_net_input(self):
        total = 0
        #print("Neuron: ", self.inputs)
        #print("weight: ", self.weights)
        for i in range(len(self.inputs)):
            total += float(self.inputs[i]) * self.weights[i]
        return total + self.bias

    # This is activation applied the logistic function to the output of the neuron
    def sigmoid(self, total_net_input):
        return 1 / (1 + math.exp(-total_net_input))

    # The error for each neuron is calculated by the Mean Square Error method:
    def calculate_error(self, target_output):
        return 0.5 * (target_output - self.output) ** 2

    # Method calculate partial derivative of total error with respect to actual output (∂E/∂yⱼ)
    # ∂E/∂yⱼ = -(target output - actual output)
    # Note that the actual output of the output neuron is often written as yⱼ and target output as tⱼ
    # => ∂E/∂yⱼ = -(tⱼ - yⱼ)
    def calculate_partial_derivative_error_with_respect_to_output(self, target_output):
        return -(target_output - self.output)

    # Method calculate partial derivative of actual output with respect to total weighted sum (dyⱼ/dNetⱼ)
    # The actual output of the neuron is calculate using sigmoid logistic function:
    # yⱼ = 1 / (1 + e^(-Netⱼ))
    # The derivative (not partial derivative since there is only one variable) of the output then is:
    # dyⱼ/dNetⱼ = yⱼ * (1 - yⱼ)
    def calculate_partial_derivative_total_net_input_with_respect_to_input(self):
        return self.output * (1 - self.output)

    # Partial derivative of weighted sum (net) with respect to wieght (∂Netⱼ/∂wᵢ)
    # The total net input is the weighted sum of all the inputs to the neuron and their respective weights:
    # = Netⱼ = x₁w₁ + x₂w₂ ...
    # The partial derivative of the total net input with respective to a given weight (with everything else held constant) then is:
    # = ∂Netⱼ/∂wᵢ = some constant + 1 * xᵢw₁^(1-0) + some constant ... = xᵢ
    def calculate_partial_derivative_total_net_input_with_respect_to_weight(self, index):
        #print(self.inputs[index])
        return self.inputs[index]

    # Method calculate partial derivative of total errors with respect to weight of i (∂E/∂zⱼ) 
    # This value is also known as the delta (δ)
    # δ = ∂E/∂zⱼ = ∂E/∂yⱼ * dyⱼ/dzⱼ
    def calculate_partial_derivative_error_with_respect_to_total_net_input(self, target_output):
        return self.calculate_partial_derivative_error_with_respect_to_output(target_output) * self.calculate_partial_derivative_total_net_input_with_respect_to_input();

# ******* End of Neuron Class ******* #
# *********************************** #