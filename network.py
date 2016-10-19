import numpy as np

class Utilities:
    @classmethod
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    @classmethod
    def sigmoid_prime(self, z):
        """Derivative of the sigmoid function."""
        return self.sigmoid(z)*(1-self.sigmoid(z))

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """
        Return output of the network if a is the input
        """
        for b, w in zip(self.biases, self.weights):
            a = Utilities.sigmoid(np.dot(w, a)+b)

        return a


    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        Runs stociatic gradient decent for dataset. Train neural network.
        """
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))

            else:
                print("Epoch {0} complete".format(j))


    def update_mini_batch(self, mini_batch, eta):
        """
        Use stochiatic gradient decent to update weights and biases, in mini_batch size
        """
        sum_d_b = [np.zeros(b.shape) for b in self.biases]
        sum_d_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_b, delta_w = self.backprop(x, y)
            sum_d_b = [d+n for d, n in zip(sum_d_b, delta_b)]
            sum_d_w = [d+n for d, n in zip(sum_d_w, delta_w)]

        self.weights = [w-(eta/len(mini_batch))*s for w, s in zip(self.weights, sum_d_w)]
        self.biases = [b-(eta/len(mini_batch))*s for b, s in zip(self.biases, sum_d_b)]

    def backprop(self, x, y):
        """
        Backpropagation algorithm.
        """
        # FeedForward Algorithm
        activation = x
        activations = [x]
        z_vectors = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            activation = Utilities.sigmoid(z)
            # Save results to calculate gradients later
            z_vectors.append(z)
            activations.append(activation)

        # Calculate Output error
        delta = self.cost_derivative(activations[-1], y)
        delta *= Utilities.sigmoid_prime(z_vectors[-1])
        deltas = [delta]

        for l in range(self.num_layers-2, 0, -1):
            d_sig = Utilities.sigmoid_prime(z_vectors[l-1])
            delta = np.dot(self.weights[l].transpose(), delta) * d_sig
            deltas.append(delta)

        deltas = list(reversed(deltas)) # put delta in order for cost function calculation

        # Backpropagate and output
        d_bias = [np.zeros(b.shape) for b in self.biases]
        d_weights = [np.zeros(w.shape) for w in self.weights]

        for l in range(len(deltas)):
            d_weights[l] = np.dot(deltas[l], activations[l].transpose())
            d_bias[l] = deltas[l]

        return d_bias, d_weights
        

    def evaluate(self, test_data):
        """
        Return the number of test inputs for which the neural
        network outputs the correct result.
        """
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        "Return partial derivative of dC/da of the final output"
        return (output_activations-y)

