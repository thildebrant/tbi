# Contributing to TBI Lesion Analysis Pipeline

Thank you for your interest in contributing to the TBI Lesion Analysis Pipeline! This document provides guidelines for contributing to the project.

## Code of Conduct

This project is committed to providing a welcoming and inclusive environment for all contributors. Please be respectful and considerate in all interactions.

## How to Contribute

### Reporting Issues

- Use the GitHub issue tracker to report bugs or request features
- Include detailed information about your environment (OS, Python version, etc.)
- Provide reproducible examples when reporting bugs
- Include relevant error messages and stack traces

### Suggesting Enhancements

- Open an issue to discuss the proposed enhancement
- Describe the use case and expected benefits
- Consider backward compatibility and performance implications

### Submitting Code Changes

1. **Fork the repository** and create a feature branch
2. **Make your changes** following the coding standards below
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Submit a pull request** with a clear description of changes

## Development Setup

1. Clone your fork:
   ```bash
   git clone https://github.com/your-username/tbi.git
   cd tbi
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements_training.txt
   ```

4. Install development dependencies:
   ```bash
   pip install pytest pytest-cov black flake8 mypy
   ```

## Coding Standards

### Python Code Style

- Follow PEP 8 style guidelines
- Use type hints for function parameters and return values
- Keep functions focused and under 50 lines when possible
- Add docstrings for all public functions and classes

### Code Formatting

- Use Black for code formatting
- Run `black .` before committing
- Use 4 spaces for indentation

### Code Quality

- Run `flake8` to check for style issues
- Run `mypy` for type checking
- Ensure all tests pass before submitting

### Commit Messages

- Use clear, descriptive commit messages
- Start with a verb in present tense (e.g., "Add", "Fix", "Update")
- Reference issue numbers when applicable

## Testing

- Write unit tests for new functionality
- Ensure existing tests continue to pass
- Aim for good test coverage
- Run tests with: `pytest tests/`

## Documentation

- Update README.md for user-facing changes
- Add docstrings for new functions and classes
- Update inline comments for complex logic
- Keep configuration examples up to date

## Medical Imaging Considerations

- Ensure proper handling of medical data privacy
- Validate input data formats and ranges
- Include appropriate warnings for medical use
- Consider performance implications for large datasets

## Review Process

1. All pull requests require review
2. Maintainers will review for:
   - Code quality and style
   - Functionality and correctness
   - Documentation completeness
   - Test coverage
3. Address feedback and make requested changes
4. Once approved, maintainers will merge the PR

## Getting Help

- Open an issue for questions or problems
- Join discussions in existing issues
- Check the documentation for common questions

Thank you for contributing to the TBI Lesion Analysis Pipeline! 