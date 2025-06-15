# MuZero

A clean, elegant implementation of the MuZero algorithm in PyTorch.

## Contributing
For guidelines on contributing to this project, please see the [CONTRIBUTING](CONTRIBUTING) document.

## Features

- **Minimal dependencies**: Only PyTorch, NumPy, and Gymnasium
- **Clean architecture**: Modular design with separate components
- **Type hints**: Full type annotations for better code quality
- **Simple configuration**: Single config class for all parameters
- **Fast development**: Uses `uv` for project management

## Installation

### Get Started

```bash
pip install -U uv
```

### Development Setup

To get started quickly, use CPU mode to avoid large downloads:

```bash
# Clone the repository
git clone git@github.com:carletonai/muzero.git
cd muzero

# Install dependencies
uv sync 
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
```

### Testing
```bash
# Run all tests 
uv run pytest

# Run tests with verbose output 
uv run pytest -v

# Run specific test file 
uv run pytest tests/test_mcts.py
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

# Remove pytest/ruff cache 
rm -rf .*_cache
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.