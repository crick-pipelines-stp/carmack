# Copilot Instructions for This Repository

## Installation Instructions
- Use python 3.12
- Install the dev environment with `pip install -e ".[dev,tests]"`

## Agent Instructions
- Always discuss the task with the user before starting to write code.
- start with high-level design and architecture, then move to implementation details.
- Ask for clarification if the task is not clear.
- If the task is too complex, break it down into smaller tasks and implement them one by one.
- Never start writing code without a green light from the user.
- Once green lit, automatically run `pytest` after each code change to ensure tests pass.

## General Principles
- Follow **Test-Driven Development (TDD)**: write tests before writing implementation code.
- Use `pytest` as the test framework.
- Do not include unused code; only write what’s needed to make the current test pass.
- Keep code DRY (Don't Repeat Yourself) and KISS (Keep It Simple, Stupid).
- Write code that is easy to read and understand.
- Use descriptive names for functions, variables, and classes.
- Avoid premature optimization; focus on clarity and correctness first.
- Use comments to explain complex logic, but avoid obvious comments.
- Use docstrings for all public functions and classes to describe their purpose and usage.

## Project Conventions
- The main source code lives in `carmack/`.
- The project a single `pyproject.toml` file for configuration.
- Use `pytest` for running tests, and ensure all tests pass before committing code.
- Use `black` and `isort` for code formatting and `ruff` for linting.


## Code Style
- Use [PEP8](https://peps.python.org/pep-0008/) formatting.
- Use type hints in all function signatures.
- Use f-strings for string interpolation.
- Prefer list comprehensions and generator expressions where appropriate.
- Put new lines at the end of files.
- Use spaces around operators and after commas.
- Use 4 spaces for indentation.
- Avoid using tabs for indentation.
- Use double quotes for strings, except for docstrings which should use triple double quotes.
- Put new lines between code above and comments below.

## Testing Guidelines
- All new code must be accompanied by tests.
- Structure tests in the `tests/` directory, with the file name starting with `test_`. The test directory is flat, the file names should be a mirror of the source code structure. For example `my_project/core.py` should have tests in `tests/test_core.py`.
- Put tests in a class and use `pytest` fixtures for setup and teardown.
- Write clear, descriptive test names that indicate what the test is verifying.
- Use the assert_that library for assertions to improve readability.
- Use `pytest.mark.parametrize` to test multiple scenarios in a single test function.
- Tests should be clear, concise, and isolated (no I/O or network unless mocked).
