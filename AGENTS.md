# Commands

## Environment
This repository uses a `conda` environment to manage dependencies. The name of this environment will be user-specific, and most likely will be the interpreter configured for this folder or workspace in VSCode. If you are uncertain about the environment name, please ask the user. **Always ensure you have the conda environment active when working within the workspace.**

## Make
We have the following make targets that may be useful in feature development. The make commands will NOT work unless you have the appropriate conda environment active.

make check                          Run development checks (preferred end-stage validation wrapper of formatting, static type checking, non-"slow" tests, doc build, and doc tests, designed to mirror Jenkins CI.)

### Testing

make test-all                       Run all tests
make test-e2e                       Run end-to-end tests
make test-integration               Run integration tests
make test-unit                      Run unit tests

### Formatting

make format                         Format code (isort and black)
make lint                           Check for formatting errors
make mypy                           Check for type hinting errors

The make commands themselves are centralized in a separate repository upon which vivarium depends, called vivarium_build_utils.

There are other make targets, but in general these will be less relevant to your tasks and you should always explicitly ask the user if you are considering using a make target that is not listed above.

# Code Style

On our team, we type-hint all new python code. We generally use NumPy Doc style, with the exception that we do not add types in the dosctrings for parameters or returns, since they are given by the type hinting. Don't add a return section if the function returns None. Always add a full docstring to new public methods and functions, but for private methods and functions a single line is usually sufficient. Docstrings should be in the imperative mood.

