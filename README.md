# MuZero

A clean, elegant implementation of the MuZero algorithm in PyTorch with modular design for different observation types.

## Features

- **Minimal dependencies**: Only PyTorch, NumPy, and Gymnasium
- **Modular architecture**: Separate encoders for 1D vector and 3D image observations
- **Type hints**: Full type annotations for better code quality
- **Simple configuration**: Single config class for all parameters
- **Fast development**: Uses `uv` for dependency management

## Installation

### Prerequisites

This project uses `uv` for fast dependency management. Install it first:

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
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
# Install all dependencies (equivalent to "make install")
uv sync --group dev --extra cpu

# Check installation (equivalent to "make check-install")
uv run python -c "import muzero; print('✓ MuZero installed successfully')"
```

### Code Quality
```bash
# Format code (equivalent to "make format")
uv run ruff format

# Lint code (equivalent to "make lint")
uv run ruff check

# Fix linting issues automatically (equivalent to "make lint-fix")
uv run ruff check --fix

# Type checking (equivalent to "make typecheck")
uv run mypy src/

# Run all quality checks (equivalent to "make quality")
uv run ruff format && uv run ruff check && uv run mypy src/
```

### Testing
```bash
# Run all tests (equivalent to "make test")
uv run pytest

# Run tests with verbose output (equivalent to "make test-verbose")
uv run pytest -v

# Run specific test file (equivalent to "make test-file")
uv run pytest tests/test_muzero.py

# Run tests with coverage (equivalent to "make test-coverage")
uv run pytest --cov=src/muzero
```

### Running Examples
```bash
# Run CartPole example (equivalent to "make run-example")
uv run python examples/cartpole_example.py

# Run with specific config
uv run python examples/cartpole_example.py --episodes 100
```

### All-in-One Commands
```bash
# Full development setup (equivalent to "make dev")
uv sync --group dev --extra cpu

# Full quality pipeline (equivalent to "make all" or "make ci")
uv run ruff format && uv run ruff check && uv run mypy src/ && uv run pytest

# Clean build (equivalent to "make clean")
uv clean && uv sync --group dev --extra cpu
```

### Understanding uv Flags

- `--group dev`: Install development dependencies (testing, linting, etc.)
- `--extra cpu`: Install PyTorch CPU version (lighter, faster)
- `--extra cuda`: Install PyTorch CUDA version (for GPU training)
- `uv sync`: Install/update dependencies based on lock file
- `uv run`: Run a command in the project's virtual environment

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

## Project Status

✅ **All systems operational:**
- ✅ Code formatting (ruff)
- ✅ Linting (ruff) 
- ✅ Type checking (mypy)
- ✅ Tests passing (pytest)
- ✅ Modular encoder architecture
- ✅ Support for 1D and 3D observations
- ✅ Fast development workflow with uv

## Contributing

1. **Setup**: `uv sync --group dev --extra cpu`
2. **Code**: Make your changes
3. **Quality**: `uv run ruff format && uv run ruff check && uv run mypy src/`
4. **Test**: `uv run pytest`
5. **Submit**: Create a pull request

For the best experience, run the full quality pipeline:
```bash
uv run ruff format && uv run ruff check && uv run mypy src/ && uv run pytest
```

## Adding New Observation Types

To add support for new observation types, simply:

1. Create a new encoder class inheriting from `BaseEncoder`
2. Update the `_create_encoder` method in `RepresentationNetwork`
3. Add tests for the new observation type

Example for 2D observations:
```python
class GridEncoder(BaseEncoder):
    """Encoder for 2D grid observations."""
    
    def __init__(self, observation_shape, hidden_dim, num_blocks):
        super().__init__(observation_shape, hidden_dim, num_blocks)
        # Implementation here
        
    def forward(self, observation):
        # Forward pass implementation
        return output
```

## License

MIT