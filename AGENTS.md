# Commands

## Environment
This repository uses a `conda` environment to manage dependencies. The name of this environment will be user-specific, and most likely will be the interpreter configured for this folder or workspace in VSCode. If you are uncertain about the environment name, please ask the user. Always ensure you have the conda environment active when working within the workspace.

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

# Code Style

On our team, we type-hint all new python code. We generally use NumPy style, with the exception that we do not add types for parameters or returns, since they are given by the type hinting.

