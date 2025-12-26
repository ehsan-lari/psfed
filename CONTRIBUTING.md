# Contributing to PSFed

Thank you for your interest in contributing to PSFed! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/psfed.git
   cd psfed
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

## Code Style

We use the following tools to maintain code quality:

- **Ruff**: Linting and formatting
- **MyPy**: Static type checking
- **Pytest**: Testing

Before submitting a PR, ensure:

```bash
# Run linter
ruff check src/psfed tests

# Run formatter
ruff format src/psfed tests

# Run type checker
mypy src/psfed

# Run tests
pytest
```

## Type Hints

All public functions and methods must have type hints:

```python
def select(
    self,
    num_parameters: int,
    round_num: int,
    *,
    client_id: str | None = None,
) -> Mask:
    """Select parameters to share.
    
    Args:
        num_parameters: Total number of parameters in the model.
        round_num: Current federation round (1-indexed).
        client_id: Optional client identifier for client-specific selection.
    
    Returns:
        A Mask indicating which parameters to share.
    
    Raises:
        ValueError: If num_parameters <= 0 or round_num < 1.
    """
    ...
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/psfed --cov-report=html

# Run specific test file
pytest tests/unit/test_flattener.py

# Run specific test
pytest tests/unit/test_flattener.py::test_flatten_unflatten_roundtrip
```

### Writing Tests

- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Use descriptive test names: `test_<what>_<condition>_<expected>`
- Use pytest fixtures for common setup

```python
def test_random_mask_selector_produces_correct_count():
    selector = RandomMaskSelector(fraction=0.5)
    mask = selector.select(num_parameters=100, round_num=1)
    assert mask.count == 50
```

## Pull Request Process

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and commit with clear messages:
   ```bash
   git commit -m "Add: new mask selector based on Fisher information"
   ```

3. **Push your branch** and create a PR:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **PR Requirements:**
   - All tests pass
   - Type checking passes
   - Linting passes
   - Documentation updated (if applicable)
   - Changelog updated

## Adding a New Mask Selector

To add a new mask selection strategy:

1. **Create the selector** in `src/psfed/core/selectors.py`:

   ```python
   class MyNewSelector(MaskSelector):
       """One-line description.
       
       Longer description explaining the strategy.
       
       Args:
           fraction: Fraction of parameters to select.
           my_param: Description of your parameter.
       
       Example:
           >>> selector = MyNewSelector(fraction=0.5, my_param=10)
           >>> mask = selector.select(num_parameters=1000, round_num=1)
       """
       
       def __init__(self, fraction: float = 0.5, my_param: int = 10) -> None:
           super().__init__(fraction=fraction)
           self.my_param = my_param
       
       def select(
           self,
           num_parameters: int,
           round_num: int,
           **kwargs: Any,
       ) -> Mask:
           # Implementation
           ...
   ```

2. **Export it** in `src/psfed/__init__.py`:
   ```python
   from psfed.core.selectors import MyNewSelector
   __all__ = [..., "MyNewSelector"]
   ```

3. **Add tests** in `tests/unit/test_selectors.py`

4. **Update documentation** if needed

## Reporting Issues

When reporting issues, please include:

- Python version
- PyTorch version
- Flower version
- Minimal reproducible example
- Full error traceback

## Questions?

Feel free to open an issue with the "question" label.
