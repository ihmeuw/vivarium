"""Backward-compatibility shim for the standalone ``vivarium`` package.

The real code has moved to ``vivarium-engine`` (importable as
``vivarium.engine``). This empty wheel exists so that

    pip install vivarium

continues to resolve and pulls in the new package plus the ``vivarium-compat``
import hook. The hook redirects ``import vivarium`` to ``vivarium.engine.*``
equivalents with a ``DeprecationWarning``.

See https://github.com/ihmeuw/vivarium-suite for the new location.
"""

from pathlib import Path

from setuptools import setup

long_description = (Path(__file__).parent / "README.rst").read_text()

setup(
    name="vivarium",
    description=(
        "Backward-compatibility shim. The real package is now vivarium-engine."
    ),
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/ihmeuw/vivarium-suite",
    author="The vivarium developers",
    author_email="vivarium.dev@gmail.com",
    license="BSD-3-Clause",
    classifiers=[
        "Development Status :: 7 - Inactive",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries",
    ],
    packages=[],
    py_modules=[],
    install_requires=[
        "vivarium-engine>=5.0.0",
        "vivarium-compat>=0.6.0",
    ],
    python_requires=">=3.10",
    # Version is derived from the git tag at build time (e.g. v4.2.0 -> 4.2.0).
    # Tag, then `python -m build`, then `twine upload`.
    # `write_to` is a build-time side effect: it writes _version.py into the
    # in-tree src/ layout so ReadTheDocs can `import vivarium` from src/ for
    # the legacy docs build. The file is not included in the wheel itself
    # (packages=[] keeps the wheel a pure metapackage).
    use_scm_version={
        "write_to": "src/vivarium/_version.py",
        "write_to_template": '__version__ = "{version}"\n',
        "tag_regex": r"^(?P<prefix>v)?(?P<version>[^\+]+)(?P<suffix>.*)?$",
    },
    setup_requires=["setuptools_scm"],
)
