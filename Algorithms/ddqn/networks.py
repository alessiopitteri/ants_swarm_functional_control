import torch
import torch.nn as nn
import yaml
import torch.nn.init as init
import torch.nn.functional as F


def get_activation(name):
    """Returns the activation function based on the string name."""
    activations = {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "linear": nn.Identity(),
        "softmax":nn.Softmax(1)
    }
    return activations.get(name, nn.ReLU())  # Default to ReLU

def initialize_weights(layer, kernel_initializer=None, bias_initializer=None):
    """Applies weight and bias initialization based on configuration."""
    if kernel_initializer:
        a, b = kernel_initializer
        init.uniform_(layer.weight, a, b)

    if bias_initializer is not None:
        init.constant_(layer.bias, bias_initializer)


class MLP_critic(nn.Module):
    def __init__(self, config, state_dim, action_dim):
        super(MLP_critic, self).__init__()
        modules = []

        # Concatenate the state and action dimensions
        input_dim = state_dim

        # Create the hidden layers
        for layer in config["layers"]:
            modules.append(nn.Linear(input_dim, layer["units"]))
            modules.append(get_activation(layer["activation"]))
            # Add batch normalization if specified
            if layer["use_batchnorm"]:
                modules.append(nn.BatchNorm1d(layer["units"]))

            input_dim = layer["units"]

        # Create the dense layers
        for layer in config["dense_layers"]:
            modules.append(nn.Linear(input_dim, layer["units"]))
            modules.append(get_activation(layer["activation"]))
            input_dim = layer["units"]
        
        # Create the output layer
        output_layer = nn.Linear(input_dim, action_dim)
        modules.append(output_layer)
        modules.append(get_activation(config["output"]["activation"]))
        initialize_weights(output_layer)
        
        self.model = nn.Sequential(*modules)

    def forward(self, state):
        return self.model(state)
    

class CNN_critic(nn.Module):
    def __init__(self, config, state_dim, action_dim):
        super(CNN_critic, self).__init__()
        self.cnn_layers = nn.Sequential()
        input_channels = state_dim[0] 

        # CNN Layers
        conv_modules = []
        for layer in config["cnn_layers"]:
            conv_modules.append(nn.Conv2d(input_channels, layer['filters'],
                                          kernel_size=layer['kernel_size'],
                                          stride=layer['stride'],
                                          padding='valid'))
            if layer["use_batchnorm"]:
                conv_modules.append(nn.BatchNorm2d(layer["filters"]))

            conv_modules.append(get_activation(layer["activation"]))
            input_channels = layer['filters']
        
        self.cnn_layers = nn.Sequential(*conv_modules)
        
        # Compute the flattened size of CNN output
        with torch.no_grad():
            dummy_input = torch.zeros(1, *state_dim)
            cnn_out_dim = self.cnn_layers(dummy_input).view(1, -1).size(1)
        
        # Fully Connected Layers
        fc_modules = []
        fc_input_dim = cnn_out_dim
        for layer in config["dense_layers"]:
            fc_modules.append(nn.Linear(fc_input_dim, layer["units"]))
            
            if layer["use_batchnorm"]:
                fc_modules.append(nn.BatchNorm1d(layer["units"]))

            fc_modules.append(get_activation(layer["activation"]))
            
            fc_input_dim = layer["units"]
        
        # Output Layer
        self.fc_layers = nn.Sequential(*fc_modules)
        self.output_layer = nn.Linear(fc_input_dim, action_dim)  # Critic outputs Q-value
        initialize_weights(self.output_layer,
                            config["output"].get("kernel_initializer"),
                            config["output"].get("bias_initializer"))
        
    def forward(self, state):
        state_out = self.cnn_layers(state)
        state_out = torch.flatten(state_out, 1) #state_out.view(state_out.size(0), -1)  # Flatten CNN output
        fc_out = self.fc_layers(state_out)
        q_value = self.output_layer(fc_out)
        return q_value

