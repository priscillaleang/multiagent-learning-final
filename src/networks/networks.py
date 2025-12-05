import tensorflow as tf
from typing import List, Optional


class MLP(tf.keras.Model):
    """
    Multi-Layer Perceptron (MLP) neural network for deep reinforcement learning.
    
    Supports both Q-networks (no softmax) and policy networks (with softmax).
    Uses He normal initialization for weights and ReLU activation for hidden layers.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [128, 128],
        activation: str = 'relu',
        use_softmax: bool = False
    ):
        """
        Initialize MLP network.
        
        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output (number of actions for Q-networks/policies)
            hidden_dims: List of hidden layer dimensions, default [128, 128]
            activation: Activation function for hidden layers, default 'relu'
            use_softmax: If True, apply softmax to output (for policies), default False
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.use_softmax = use_softmax
        
        # He normal initializer for weights (good for ReLU)
        initializer = tf.keras.initializers.HeNormal()
        
        # Build hidden layers
        self.hidden_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layer = tf.keras.layers.Dense(
                hidden_dim,
                activation=activation,
                kernel_initializer=initializer,
                bias_initializer='zeros'
            )
            self.hidden_layers.append(layer)
            prev_dim = hidden_dim
        
        # Output layer (no activation, we'll apply softmax conditionally in call)
        self.output_layer = tf.keras.layers.Dense(
            output_dim,
            activation=None,  # No activation, we handle softmax in call()
            kernel_initializer=initializer,
            bias_initializer='zeros'
        )
    
    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim) or (input_dim,)
            training: Whether the model is in training mode (unused but kept for API compatibility)
        
        Returns:
            Output tensor of shape (batch_size, output_dim) or (output_dim,)
            If use_softmax=True, output is probability distribution over actions.
            If use_softmax=False, output is raw Q-values.
        """
        # Apply hidden layers sequentially
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        
        # Apply output layer
        x = self.output_layer(x, training=training)
        
        # Apply softmax if specified (for policy networks)
        if self.use_softmax:
            x = tf.nn.softmax(x, axis=-1)
        
        return x
    
    def get_config(self):
        """Return configuration dictionary for serialization."""
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dims': self.hidden_dims,
            'activation': self.activation,
            'use_softmax': self.use_softmax
        })
        return config

