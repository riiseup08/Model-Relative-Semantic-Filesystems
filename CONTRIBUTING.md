# Contributing to pymrsf

Thank you for your interest in contributing to pymrsf! This document provides guidelines and instructions for contributing.

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- Git
- (Optional) A GGUF model file for testing local features

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/pymrsf.git
   cd pymrsf
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[dev,all]"
   ```

4. **Set up your environment**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

5. **Run tests to verify setup**
   ```bash
   pytest
   ```

## 🔧 Development Workflow

### Code Style

We use the following tools to maintain code quality:
- **black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Format your code before committing:
```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/pymrsf
```

Or install the git hook so this runs automatically on every commit:
```bash
pip install pre-commit
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pymrsf --cov-report=html

# Run specific test file
pytest tests/test_fixes.py

# Local-provider tests (tests/test_local.py) auto-skip unless a GGUF model is
# present. Point them at one with: PYMRSF_TEST_MODEL=/path/to/model.gguf

# Run with verbose output
pytest -v
```

### Adding New Features

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Write your code**
   - Add your feature implementation
   - Write tests for your feature
   - Update documentation as needed

3. **Test your changes**
   ```bash
   pytest
   black src/ tests/
   flake8 src/ tests/
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: brief description of your feature"
   ```

5. **Push and create a pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

## 📝 Commit Message Guidelines

Use clear, descriptive commit messages:
- `Add: new feature or functionality`
- `Fix: bug fix`
- `Update: changes to existing functionality`
- `Docs: documentation updates`
- `Test: adding or updating tests`
- `Refactor: code refactoring`
- `Style: formatting, missing semicolons, etc.`

Example:
```
Add: async support for chunk scoring

- Implement score_chunk_async function
- Add asyncio tests
- Update documentation with async examples
```

## 🐛 Reporting Bugs

When reporting bugs, please include:
- **Description**: Clear description of the issue
- **Steps to reproduce**: Minimal code example
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Environment**: Python version, OS, provider (local/openai/anthropic)
- **Logs**: Relevant error messages or logs

## 💡 Feature Requests

We welcome feature requests! Please:
- Check existing issues first to avoid duplicates
- Clearly describe the feature and its use case
- Explain why it would be useful to the project
- Consider whether it fits the project's scope

## 🧪 Testing Guidelines

- Write tests for all new features
- Maintain or improve code coverage
- Test both success and failure cases
- Test with different providers (local, OpenAI, Anthropic)
- Use fixtures for common test setup

Example test structure:
```python
import pytest
from pymrsf import score_chunk

def test_score_chunk_basic():
    """Test basic chunk scoring functionality."""
    result = score_chunk("Test content", query="test query")
    assert "rag_score" in result
    assert 0 <= result["rag_score"] <= 100

@pytest.mark.asyncio
async def test_score_chunk_async():
    """Test async chunk scoring."""
    from pymrsf.rag import score_chunk_async
    result = await score_chunk_async("Test", query="test")
    assert result is not None
```

## 📚 Documentation

- Update README.md for user-facing changes
- Add docstrings to new functions and classes
- Update .env.example for new configuration options
- Add examples for new features

## 🔒 Security

If you discover a security vulnerability:
- **DO NOT** open a public issue
- Email the maintainers directly
- Provide details about the vulnerability
- Allow time for a fix before public disclosure

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🤝 Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the community
- Show empathy towards other contributors

## ❓ Questions?

If you have questions:
- Check existing issues and discussions
- Open a new issue with the "question" label
- Reach out to the maintainers

---

Thank you for contributing to pymrsf! 🎉
