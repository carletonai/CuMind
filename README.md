# MuZero

A clean, elegant implementation of the MuZero algorithm in PyTorch.

## Features

- **Minimal dependencies**: Only PyTorch, NumPy, and Gymnasium
- **Clean architecture**: Modular design with separate components
- **Type hints**: Full type annotations for better code quality
- **Simple configuration**: Single config class for all parameters
- **Fast development**: Uses `uv` for dependency management

## Installation

### Prerequisites

This project uses `uv` for fast dependency management. Install it first:

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via pip
pip install -U uv
```

### Development Setup

PyTorch is an optional dependency to avoid large downloads during development.

```bash
# Clone the repository
git clone <your-repo-url>
cd muzero

# Install minimal dependencies (no PyTorch - fast!)
uv sync --group dev

# Install with PyTorch for CPU (when you need ML functionality)
uv sync --group dev --extra cpu

# Install with PyTorch for CUDA (for GPU training)
uv sync --group dev --extra cuda
```

## Quick Start

```python
from muzero import MuZeroAgent, MuZeroConfig

# Create configuration for CartPole (1D observations)
config = MuZeroConfig(
    action_space_size=2,
    observation_shape=(4,),  # 1D vector
    hidden_dim=64,
    num_simulations=25
)

# Create agent
agent = MuZeroAgent(config)

# Select action
action = agent.select_action(observation)
```

For image-based environments (like Atari):

```python
# Create configuration for Atari (3D observations)
config = MuZeroConfig(
    action_space_size=4,
    observation_shape=(3, 84, 84),  # 3D image: channels, height, width
    hidden_dim=64,
    num_simulations=50
)
```

## Development Commands

The project uses `uv` instead of traditional package managers. Here are the equivalent commands:

### Building and Setup
```bash
# Install all dependencies
uv sync --extra cpu

# Check installation 
uv run python -c "import muzero; print('MuZero installed successfully')"
```

### Code Quality
```bash
# Format code 
uv run ruff format

# Lint code 
uv run ruff check

# Fix linting issues in-line 
uv run ruff check --fix

# Type checking 
uv run mypy src/

# Run all quality checks 
uv run ruff format && uv run ruff check && uv run mypy src/
```

### Testing
```bash
# Run all tests 
uv run pytest

# Run tests with verbose output 
uv run pytest -v

# Run specific test file 
uv run pytest tests/test_muzero.py

# Run tests with coverage 
uv run pytest --cov=src/muzero
```

### Cleaning the Environment

To remove build artifacts, lock files, and cached dependencies:

```bash
# Remove uv lock file
rm uv.lock

# Clean uv's cache (removes downloaded wheels, etc.)
uv cache clean

# Remove build artifacts and temporary files
uv clean
```

### Run Example
```bash
uv run python examples/cartpole_example.py
```
## Architecture

### Modular Encoder Design

The project uses a clean, modular architecture for handling different observation types:

```python
# Base encoder class
class BaseEncoder(nn.Module):
    """Base class for observation encoders."""

# For 1D vector observations (CartPole, etc.)
class VectorEncoder(BaseEncoder):
    """Encoder for 1D vector observations."""

# For 3D image observations (Atari, etc.)
class ConvEncoder(BaseEncoder):
    """Encoder for 3D image observations."""

# Automatic encoder selection
class RepresentationNetwork(nn.Module):
    """Converts observations to hidden state representation."""
    def _create_encoder(self, observation_shape, hidden_dim, num_blocks):
        if len(observation_shape) == 1:
            return VectorEncoder(observation_shape, hidden_dim, num_blocks)
        elif len(observation_shape) == 3:
            return ConvEncoder(observation_shape, hidden_dim, num_blocks)
```

This design eliminates conditional logic and makes the code more maintainable and extensible.

## Contributing

1. **Setup**: `uv sync --extra cpu`
2. **Code**: Make your changes
3. **Quality**: `uv run ruff format && uv run ruff check && uv run mypy src/`
4. **Test**: `uv run pytest`
5. **Submit**: Create a pull request

For the best experience, run the full quality pipeline:
```bash
uv run ruff format && uv run ruff check && uv run mypy src/ && uv run pytest
```


## License

[MIT](LICENSE)