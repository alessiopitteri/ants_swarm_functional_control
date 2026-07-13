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


class MLP_actor(nn.Module):
    def __init__(self, config, state_dim, action_dim):
        super(MLP_actor, self).__init__()
        modules = []
        input_dim = state_dim

        # Create the hidden layers
        for layer in config["layers"]:
            modules.append(nn.Linear(input_dim, layer["units"]))

            # Add batch normalization if specified
            if layer["use_batchnorm"]:
                modules.append(nn.BatchNorm1d(num_features=layer["units"]))
            modules.append(get_activation(layer["activation"]))

            input_dim = layer["units"]

        # Create the dense layers
        for layer in config["dense_layers"]:
            modules.append(nn.Linear(input_dim, layer["units"]))
            if layer["use_batchnorm"]:
                modules.append(nn.BatchNorm1d(layer["units"]))
            modules.append(get_activation(layer["activation"]))
            input_dim = layer["units"]
        
        # Create the output layer
        output_layer = nn.Linear(input_dim, action_dim)
        modules.append(output_layer)
        modules.append(get_activation(config["output"]["activation"]))
        kernel_initializer = config["output"]["kernel_initializer"]
        bias_initializer = config["output"]["bias_initializer"]
        initialize_weights(output_layer,
                            kernel_initializer,
                            bias_initializer)
        self.model = nn.Sequential(*modules)

    def forward(self, state):
        return self.model(state)
    

class MLP_critic(nn.Module):
    def __init__(self, config, state_dim, action_dim):
        super(MLP_critic, self).__init__()
        modules = []

        # Concatenate the state and action dimensions
        input_dim = state_dim + action_dim

        # Create the hidden layers
        for layer in config["layers"]:
            modules.append(nn.Linear(input_dim, layer["units"]))

            # Add batch normalization if specified
            if layer["use_batchnorm"]:
                modules.append(nn.BatchNorm1d(num_features=layer["units"]))

            modules.append(get_activation(layer["activation"]))

            input_dim = layer["units"]

        # Create the dense layers
        for layer in config["dense_layers"]:
            modules.append(nn.Linear(input_dim, layer["units"]))
            if layer["use_batchnorm"]:
                modules.append(nn.BatchNorm1d(layer["units"]))
            modules.append(get_activation(layer["activation"]))
            input_dim = layer["units"]
        
        # Create the output layer
        output_layer = nn.Linear(input_dim, 1)
        modules.append(output_layer)
        modules.append(get_activation(config["output"]["activation"]))
        initialize_weights(output_layer)
        
        self.model = nn.Sequential(*modules)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.model(x)


class CNN_actor(nn.Module):
    def __init__(self, config, state_dim, action_dim):
        super(CNN_actor, self).__init__()
        input_channels = state_dim[0]

        conv_modules = []
        # Create the hidden layers
        for layer in config["cnn_layers"]:
            conv_modules.append(nn.Conv2d(input_channels, layer['filters'],
                                      kernel_size=layer['kernel_size'],
                                        stride=layer['stride'],
                                        padding='valid'))
            
            # Add batch normalization if specified
            if layer["use_batchnorm"]:
                conv_modules.append(nn.BatchNorm2d(layer["filters"]))
            conv_modules.append(get_activation(layer["activation"]))
            input_channels = layer['filters']
        
        self.cnn_layers = nn.Sequential(*conv_modules)

        # Compute the flattened size of CNN output
        with torch.no_grad():
            dummy_input = torch.zeros(1, *state_dim)
            cnn_out_dim = self.cnn_layers(dummy_input).view(1, -1).size(1)

        input_dim = cnn_out_dim

        modules = []
        # Create the dense layers
        for layer in config["dense_layers"]:
            modules.append(nn.Linear(input_dim, layer["units"]))
            if layer["use_batchnorm"]:
                modules.append(nn.BatchNorm1d(layer["units"]))
            modules.append(get_activation(layer["activation"]))
            input_dim = layer["units"]
        
        # Create the output layer
        output_layer = nn.Linear(input_dim, action_dim)
        modules.append(output_layer)
        modules.append(get_activation(config["output"]["activation"]))
        kernel_initializer = config["output"]["kernel_initializer"]
        bias_initializer = config["output"]["bias_initializer"]
        initialize_weights(output_layer,
                            kernel_initializer,
                            bias_initializer)
        self.dense_layers = nn.Sequential(*modules)

    def forward(self, state):
        state_out = self.cnn_layers(state)
        state_out = state_out.view(state_out.size(0), -1) 
        pi = self.dense_layers(state_out)
        return pi

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
            conv_modules.append(get_activation(layer["activation"]))
            if layer["use_batchnorm"]:
                conv_modules.append(nn.BatchNorm2d(layer["filters"]))
            input_channels = layer['filters']
        
        self.cnn_layers = nn.Sequential(*conv_modules)
        
        # Compute the flattened size of CNN output
        with torch.no_grad():
            dummy_input = torch.zeros(1, *state_dim)
            cnn_out_dim = self.cnn_layers(dummy_input).view(1, -1).size(1)
        
        # Action processing layers
        action_modules = []
        action_input_dim = action_dim
        for layer in config["action_layers"]:
            action_modules.append(nn.Linear(action_input_dim, layer["units"]))
            if layer["use_batchnorm"]:
                action_modules.append(nn.BatchNorm1d(layer["units"]))
            action_modules.append(get_activation(layer["activation"]))
            action_input_dim = layer["units"]
        self.action_layers = nn.Sequential(*action_modules)
        
        # Fully Connected Layers after concatenation
        fc_modules = []
        fc_input_dim = cnn_out_dim + action_input_dim
        for layer in config["dense_layers"]:
            fc_modules.append(nn.Linear(fc_input_dim, layer["units"]))
            if layer["use_batchnorm"]:
                fc_modules.append(nn.BatchNorm1d(layer["units"]))
            fc_modules.append(get_activation(layer["activation"]))
            fc_input_dim = layer["units"]
        
        # Output Layer
        self.fc_layers = nn.Sequential(*fc_modules)
        self.output_layer = nn.Linear(fc_input_dim, 1)  # Critic outputs Q-value
        initialize_weights(self.output_layer,
                            config["output"].get("kernel_initializer"),
                            config["output"].get("bias_initializer"))
        
    def forward(self, state, action):
        state_out = self.cnn_layers(state)
        state_out = state_out.view(state_out.size(0), -1)  # Flatten CNN output
        action_out = self.action_layers(action)
        combined = torch.cat([state_out, action_out], dim=1)
        fc_out = self.fc_layers(combined)
        q_value = self.output_layer(fc_out)
        return q_value
