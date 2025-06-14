# Contributing to MuZero

Thank you for your interest in improving MuZero! Please follow these guidelines to ensure a smooth contribution process.

## Branch Naming

Use these patterns for branch names:

- `feature/description` — New features
- `internal/description` — Refactoring or internal changes
- `bugfix/description` — Bug fixes
- `dev` — Development branch
- `master` — Protected branch
**Examples:**
- `feature/add-atari-support`
- `internal/refactor-mcts`
- `bugfix/fix-reward-scaling`
- `my-feature` (not allowed)
- `fix-bug` (not allowed)

## Development Setup

1. **Clone the repo**
    ```bash
    git clone https://github.com/carletonai/muzero.git
    cd muzero
    ```
2. **Install dependencies**
    ```bash
    uv sync --dev
    ```
3. **Run tests**
    ```bash
    uv run pytest
    ```

## Code Quality

- **Lint & Format:**  
  ```bash
  uv run ruff check .
  uv run ruff format .
  ```
- **Type Check:**  
  ```bash
  uv run mypy src/
  ```
- **Test:**  
  ```bash
  uv run pytest tests/ -v
  ```

## Pull Requests

1. Create a branch (see naming above)
2. Make changes with tests
3. Ensure all checks pass
4. Open a pull request with a clear description

## Style Guide

- Follow PEP 8
- Use type hints everywhere
- Add docstrings to public functions/classes
- Max line length: 100
- Use descriptive names

## Testing

- Add unit tests for new code
- Keep test coverage as high as we can
- Use clear, descriptive test names
- Include both positive and negative cases

## Documentation

- Update `README.md` for new features
- Provide code examples for complex features

## Questions?

- Check issues/discussions first
- Open a new issue with the "question" label if needed
- Contact maintainers if unsure

Thank you for contributing! 🚀
