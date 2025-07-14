"""Unified CuMind neural network components."""

from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import nnx

from ..utils.logger import log


class ResidualBlock(nnx.Module):
    """A standard residual block with two convolutional layers."""

    def __init__(self, channels: int, rngs: nnx.Rngs):
        """Initializes the residual block.

        Args:
            channels: The number of input and output channels.
            rngs: Random number generators for layer initialization.
        """
        log.debug(f"Initializing ResidualBlock with {channels} channels.")
        self.conv1 = nnx.Conv(channels, channels, kernel_size=(3, 3), padding=1, rngs=rngs)
        self.bn1 = nnx.BatchNorm(channels, use_running_average=True, rngs=rngs)
        self.conv2 = nnx.Conv(channels, channels, kernel_size=(3, 3), padding=1, rngs=rngs)
        self.bn2 = nnx.BatchNorm(channels, use_running_average=True, rngs=rngs)

    def __call__(self, x: chex.Array) -> chex.Array:
        """Performs a forward pass through the residual block.

        Args:
            x: The input tensor of shape (batch, height, width, channels).

        Returns:
            The output tensor with the same shape as the input.
        """
        residual = x
        x_array = jnp.asarray(x, dtype=jnp.float32)
        x_array = self.conv1(x_array)
        x_array = self.bn1(x_array)
        x_array = nnx.relu(x_array)
        x_array = self.conv2(x_array)
        x_array = self.bn2(x_array)
        return nnx.relu(x_array + residual)


class BaseEncoder(nnx.Module):
    """Abstract base class for observation encoders."""

    def __init__(self, observation_shape: Tuple[int, ...], hidden_dim: int, num_blocks: int, rngs: nnx.Rngs):
        """Initializes the base encoder.

        Args:
            observation_shape: The shape of the input observations.
            hidden_dim: The dimension of the output hidden representation.
            num_blocks: The number of processing blocks (e.g., residual or dense).
            rngs: Random number generators for layer initialization.
        """
        self.observation_shape = observation_shape
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.rngs = rngs

    def __call__(self, observation: chex.Array) -> chex.Array:
        """Encodes an observation into a hidden state."""
        raise NotImplementedError("BaseEncoder is an abstract class.")


class VectorEncoder(BaseEncoder):
    """An encoder for 1D vector observations using fully connected layers."""

    def __init__(self, observation_shape: Tuple[int, ...], hidden_dim: int, num_blocks: int, rngs: nnx.Rngs):
        """Initializes the vector encoder.

        Args:
            observation_shape: The shape of the 1D observation (e.g., (4,)).
            hidden_dim: The output hidden dimension.
            num_blocks: The number of fully connected blocks.
            rngs: Random number generators for layer initialization.
        """
        super().__init__(observation_shape, hidden_dim, num_blocks, rngs)
        log.debug(f"Initializing VectorEncoder with input dim {observation_shape[0]}, hidden dim {hidden_dim}, and {num_blocks} blocks.")

        input_dim = observation_shape[0]
        self.layers = []

        current_dim = input_dim
        for i in range(num_blocks):
            output_dim = hidden_dim if i == num_blocks - 1 else hidden_dim
            self.layers.append(nnx.Linear(current_dim, output_dim, rngs=rngs))
            current_dim = output_dim

    def __call__(self, observation: chex.Array) -> chex.Array:
        """Encodes a 1D observation into a hidden state.

        Args:
            observation: A flattened observation of shape (batch, obs_dim).

        Returns:
            A hidden state tensor of shape (batch, hidden_dim).
        """
        x = jnp.asarray(observation, dtype=jnp.float32)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = nnx.relu(x)
        return x


class ConvEncoder(BaseEncoder):
    """An encoder for 3D image observations using convolutional layers."""

    def __init__(self, observation_shape: Tuple[int, ...], hidden_dim: int, num_blocks: int, conv_channels: int, rngs: nnx.Rngs):
        """Initializes the convolutional encoder.

        Args:
            observation_shape: The shape of the 3D observation (height, width, channels).
            hidden_dim: The output hidden dimension.
            num_blocks: The number of residual blocks.
            conv_channels: The number of channels for the convolutional layers.
            rngs: Random number generators for layer initialization.
        """
        super().__init__(observation_shape, hidden_dim, num_blocks, rngs)
        log.debug(f"Initializing ConvEncoder with input shape {observation_shape}, hidden dim {hidden_dim}, and {num_blocks} blocks.")

        self.initial_conv = nnx.Conv(observation_shape[-1], conv_channels, kernel_size=(3, 3), strides=(2, 2), padding=1, rngs=rngs)
        self.residual_blocks = [ResidualBlock(conv_channels, rngs) for _ in range(num_blocks)]
        self.final_dense = nnx.Linear(conv_channels, hidden_dim, rngs=rngs)

    def __call__(self, observation: chex.Array) -> chex.Array:
        """Encodes a 3D observation into a hidden state.

        Args:
            observation: An image observation of shape (batch, height, width, channels).

        Returns:
            A hidden state tensor of shape (batch, hidden_dim).
        """
        x = jnp.asarray(observation, dtype=jnp.float32)
        x = nnx.relu(self.initial_conv(x))

        for block in self.residual_blocks:
            x = jnp.asarray(block(x), dtype=jnp.float32)

        pooled = jnp.mean(x, axis=(1, 2))
        return self.final_dense(pooled)


class RepresentationNetwork(nnx.Module):
    """The representation network (r_theta) that encodes raw observations into hidden states."""

    def __init__(self, observation_shape: Tuple[int, ...], hidden_dim: int, num_blocks: int, conv_channels: int, rngs: nnx.Rngs):
        """Initializes the representation network.

        Args:
            observation_shape: The shape of the input observations.
            hidden_dim: The dimension of the hidden representation.
            num_blocks: The number of processing blocks in the encoder.
            conv_channels: The number of channels for the convolutional layers.
            rngs: Random number generators for layer initialization.
        """
        log.info(f"Initializing RepresentationNetwork with hidden dim {hidden_dim} and {num_blocks} blocks.")

        self.observation_shape = observation_shape
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.conv_channels = conv_channels
        self.encoder = self._create_encoder(rngs)

    def _create_encoder(self, rngs: nnx.Rngs) -> BaseEncoder:
        """Creates the appropriate encoder based on the observation shape."""
        log.debug(f"Creating encoder for observation shape: {self.observation_shape}")
        if len(self.observation_shape) == 1:
            log.info("Using VectorEncoder for 1D observations.")
            return VectorEncoder(self.observation_shape, self.hidden_dim, self.num_blocks, rngs)
        if len(self.observation_shape) == 3:
            log.info("Using ConvEncoder for 3D observations.")
            return ConvEncoder(self.observation_shape, self.hidden_dim, self.num_blocks, self.conv_channels, rngs)
        log.critical(f"Unsupported observation shape: {self.observation_shape}")
        raise ValueError(f"Unsupported observation shape: {self.observation_shape}")

    def __call__(self, observation: chex.Array) -> chex.Array:
        """Encodes an observation into a hidden state.

        Args:
            observation: The input observation tensor.

        Returns:
            A hidden state tensor of shape (batch, hidden_dim).
        """
        return self.encoder(observation)


class DynamicsNetwork(nnx.Module):
    """The dynamics network (g_theta) that predicts the next hidden state and reward."""

    def __init__(self, hidden_dim: int, internal_hidden_dim: int, action_space_size: int, num_layers: int, rngs: nnx.Rngs):
        """Initializes the dynamics network.

        Args:
            hidden_dim: The dimension of the hidden states.
            internal_hidden_dim: The dimension of the internal hidden layers.
            action_space_size: The number of possible actions.
            num_layers: The number of hidden layers.
            rngs: Random number generators for layer initialization.
        """
        log.info(f"Initializing DynamicsNetwork with hidden dim {hidden_dim}, internal hidden dim {internal_hidden_dim}, action space size {action_space_size}, and {num_layers} hidden layers.")
        
        self.hidden_dim = hidden_dim
        self.internal_hidden_dim = internal_hidden_dim
        self.action_space_size = action_space_size
        self.num_layers = num_layers

        self.action_embedding = nnx.Embed(action_space_size, hidden_dim, rngs=rngs)
        self.layers = []
        self.layers.append(nnx.Linear(hidden_dim, internal_hidden_dim, rngs=rngs))
        for _ in range(num_layers - 2):
            self.layers.append(nnx.Linear(internal_hidden_dim, internal_hidden_dim, rngs=rngs))
        self.layers.append(nnx.Linear(internal_hidden_dim, hidden_dim, rngs=rngs))
        self.reward_head = nnx.Linear(internal_hidden_dim, 1, rngs=rngs)

    def __call__(self, hidden_state: chex.Array, action: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Predicts the next hidden state and reward.

        Args:
            hidden_state: The current hidden state of shape (batch, hidden_dim).
            action: The action indices of shape (batch,).

        Returns:
            A tuple containing the next hidden state and the predicted reward.
        """
        action_embedded = self.action_embedding(jnp.asarray(action, dtype=jnp.int32))
        x = jnp.asarray(hidden_state, dtype=jnp.float32) + action_embedded

        for layer in self.layers:
            residual = x
            x = nnx.relu(layer(x))
            x = x + residual

        reward = self.reward_head(x)
        return x, reward


class PredictionNetwork(nnx.Module):
    """The prediction network (f_theta) that predicts the policy and value from a hidden state."""

    def __init__(self, hidden_dim: int, internal_hidden_dim: int, action_space_size: int, num_layers: int, rngs: nnx.Rngs):
        """Initializes the prediction network.

        Args:
            hidden_dim: The dimension of the hidden states.
            internal_hidden_dim: The dimension of the internal hidden layers.
            num_layers: The number of hidden layers in the networks.
            action_space_size: The number of possible actions for the policy head.
            rngs: Random number generators for layer initialization.
        """
        log.info(f"Initializing PredictionNetwork with hidden dim {hidden_dim}, internal hidden dim {internal_hidden_dim}, action space size {action_space_size}, and {num_layers} hidden layers.")

        self.hidden_dim = hidden_dim
        self.internal_hidden_dim = internal_hidden_dim
        self.action_space_size = action_space_size
        self.num_layers = num_layers

        self.layers = []
        self.layers.append(nnx.Linear(hidden_dim, internal_hidden_dim, rngs=rngs))
        for _ in range(num_layers - 1):
            self.layers.append(nnx.Linear(internal_hidden_dim, internal_hidden_dim, rngs=rngs))
        self.policy_head = nnx.Linear(internal_hidden_dim, action_space_size, rngs=rngs)
        self.value_head = nnx.Linear(internal_hidden_dim, 1, rngs=rngs)

    def __call__(self, hidden_state: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Predicts the policy logits and value from a hidden state.

        Args:
            hidden_state: A hidden state tensor of shape (batch, hidden_dim).

        Returns:
            A tuple of (policy_logits, value).
        """
        x = jnp.asarray(hidden_state, dtype=jnp.float32)
        for layer in self.layers:
            x = nnx.relu(layer(x))
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value


class CuMindNetwork(nnx.Module):
    """The complete CuMind network, combining representation, dynamics, and prediction."""

    def __init__(self, observation_shape: Tuple[int, ...], action_space_size: int, hidden_dim: int, prediction_internal_hidden_dim: int, dynamics_internal_hidden_dim: int, prediction_num_layers: int, dynamics_num_layers: int, num_blocks: int, conv_channels: int, rngs: nnx.Rngs):
        """Initializes the complete CuMind network.

        Args:
            observation_shape: The shape of the input observations.
            action_space_size: The number of possible actions.
            hidden_dim: The dimension of the hidden representations.
            prediction_internal_hidden_dim: The dimension of the internal hidden layers in the Prediction network.
            dynamics_internal_hidden_dim: The dimension of the internal hidden layers in the Dynamics network.
            prediction_num_layers: The number of hidden layers in the Prediction network.
            dynamics_num_layers: The number of hidden layers in the Dynamics network.
            num_blocks: The number of processing blocks in the Representation network.
            conv_channels: The number of channels for the convolutional layers.
            rngs: Random number generators for layer initialization.
        """
        log.info(f"Initializing CuMindNetwork for observation shape {observation_shape} and action space size {action_space_size}.")
        self.representation_network = RepresentationNetwork(observation_shape, hidden_dim, num_blocks, conv_channels, rngs)
        self.dynamics_network = DynamicsNetwork(hidden_dim, dynamics_internal_hidden_dim, action_space_size, dynamics_num_layers, rngs)
        self.prediction_network = PredictionNetwork(hidden_dim, prediction_internal_hidden_dim, action_space_size, prediction_num_layers, rngs)

    def initial_inference(self, observation: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Performs the initial inference step from an observation.

        Args:
            observation: The input observation tensor.

        Returns:
            A tuple of (hidden_state, policy_logits, value).
        """
        log.debug(f"Initial inference with observation shape: {observation.shape}")
        hidden_state = self.representation_network(observation)
        policy_logits, value = self.prediction_network(hidden_state)
        return hidden_state, policy_logits, value

    def recurrent_inference(self, hidden_state: chex.Array, action: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        """Performs a recurrent inference step from a hidden state and action.

        Args:
            hidden_state: The current hidden state.
            action: The action to take.

        Returns:
            A tuple of (next_hidden_state, reward, policy_logits, value).
        """
        log.debug(f"Recurrent inference with hidden state shape: {hidden_state.shape} and action shape: {action.shape}")
        next_hidden_state, reward = self.dynamics_network(hidden_state, action)
        policy_logits, value = self.prediction_network(next_hidden_state)
        return next_hidden_state, reward, policy_logits, value
