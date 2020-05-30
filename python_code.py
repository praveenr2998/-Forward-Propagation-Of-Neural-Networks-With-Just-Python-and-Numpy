# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from random import seed



# GENERALLY DEFINING A LAYER 

def initialize_network(num_inputs,num_hidden_layers,num_nodes_hidden,num_nodes_output):
    

    # The whole network is going to defined as a dictionary    
    network={}
    num_nodes_previous = num_inputs # number of nodes in the previous layer
    

    # loop through each layer and randomly initialize the weights and biases associated with each layer
    for layer in range(num_hidden_layers+1):
        
        if layer == num_hidden_layers:
            layer_name = 'output'  # name last layer in the network output
            
        else:
            layer_name = 'layer_{}'.format(layer+1) # otherwise give the layer a number
            num_nodes = num_nodes_hidden[layer]
            
        
        # Initialising weights and bias for each node
        network[layer_name]={}
        for node in range(num_nodes):
            node_name = 'node_{}'.format(node+1)
            network[layer_name][node_name]={
                'weights' : np.around(np.random.uniform(size=1),decimals=2),
                'bias' : np.around(np.random.uniform(size=1),decimals=2),
                }
        num_nodes_previous = num_nodes
        
    return network # Return the network
            
            
            




# COMPUTED WEIGHTED_SUM AT EACH NODE OF EACH LAYER (For this a generalised function in created)
def compute_weighted_sum(inputs, weights, bias):
    return np.sum(inputs * weights)+bias





# COMPUTE NODE_ACTIVATION (Here we are using sigmoid activation function)
def node_activation(weighted_sum):
    return 1.0 / (1.0 + np.exp(-1 * weighted_sum))


# FORWARD PROPAGATION
def forward_propagate(network, inputs):
    
    layer_inputs = list(inputs) # start with the input layer as the input to the first hidden layer
    
    for layer in network:
        
        layer_data = network[layer]
        
        layer_outputs = [] 
        for layer_node in layer_data:
        
            node_data = layer_data[layer_node]
        
            # compute the weighted sum and the output of each node at the same time 
            node_output = node_activation(compute_weighted_sum(layer_inputs, node_data['weights'], node_data['bias']))
            layer_outputs.append(np.around(node_output[0], decimals=4))
            
        if layer != 'output':
            print('The outputs of the nodes in hidden layer number {} is {}'.format(layer.split('_')[1], layer_outputs))
    
        layer_inputs = layer_outputs # set the output of this layer to be the input to next layer

    network_predictions = layer_outputs
    return network_predictions
            
            
            


# CREATING OUR OWN DESIRED SMALL NETWORK WITH DESIRED INPUTS,HIDDEN_LAYERS AND OUTPUT
small_network=initialize_network(5,3,[3,2,3],1)
# GENERATE SOME RANDOM INPUT NUMBERS
np.random.seed(12)
inputs = np.around(np.random.uniform(size=5), decimals=2)
            
predictions=forward_propagate(small_network,inputs)
print('The predicted value by the network for the given input is {}'.format(np.around(predictions[0], decimals=4)))
            