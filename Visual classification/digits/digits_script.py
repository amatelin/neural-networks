##With V1 of network
#import network
#import mnist_loader
#
#training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
#net = network.Network([784, 100, 10])
#
#net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

##With V2 of network
import mnist_loader
import network2
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data, 30, 10 ,0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)