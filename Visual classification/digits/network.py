import numpy as np
import random

class Network():
    
    def __init__(self, sizes):
        """
        The list sizes contains the number of neurons in the respective layers.
        Example if we want to create a Network object with 2 input neurons,
        3 hidden neurons, 1 output neuron : 
        
        net = Network([2, 3, 1])
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(sizes[:-1], sizes[1:])]

    
    def feedforward(self, a):
        """
        Return the output of the network if "a" is input
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid_vec(np.dot(w, a)+b)
        return a
            
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        mini batch : list of tuples (x, y)
                
        eta : learning rate
        
        If test_data is supplied the program will evaluate the network after
        each epoch of training and print out partial progress
        """
        
        if test_data: n_test = len(test_data)
        
        n = len(training_data)
        
        for j in xrange(epochs):
            random.shuffle(training_data)
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
        Update the network's weights and biases by applying gradient descent 
        using backpropagation to a single mini batch. 
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            
        self.weights = [w - (eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                        for b, nb in zip(self.biases, nabla_b)]
                            

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        #feedforward
        activation = x
        activations = [x] #list to store all the activations layer by layer
        zs = [] #list to store all the z vectors, layer by layer
    
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid_vec(z)
            activations.append(activation)
            
        #backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime_vec(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            spv = sigmoid_prime_vec(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * spv
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            
        return (nabla_b, nabla_w)
        
        
    def evaluate(self, test_data):
        """
        Return the number of test inputs for which the neural network
        ouputs the correct result
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x==y) for (x, y) in test_results)
        
    def cost_derivative(self, output_activations, y):
        """
        Return the vector of partial derivatives for the output activations
        """
        return (output_activations-y)
        
                                       
            
            
            
            
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
    
##vectorize defines a vectorized function which takes a nested sequence of 
##objects as inputs and returns a numpy array as output applying the function
##to all the members of the input sequence. 
sigmoid_vec = np.vectorize(sigmoid) 


def sigmoid_prime(z):
    """
    Derivative of the sigmoid function
    """
    return sigmoid(z)*(1-sigmoid(z))
    
sigmoid_prime_vec = np.vectorize(sigmoid_prime)