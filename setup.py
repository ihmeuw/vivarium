import json
import sys

from packaging.version import parse

with open("python_versions.json", "r") as f:
    supported_python_versions = json.load(f)

python_versions = [parse(v) for v in supported_python_versions]
min_version = min(python_versions)
max_version = max(python_versions)

if not (
    min_version <= parse(".".join([str(v) for v in sys.version_info[:2]])) <= max_version
):
    # Python 3.5 does not support f-strings
    py_version = ".".join([str(v) for v in sys.version_info[:3]])
    error = (
        "\n----------------------------------------\n"
        "Error: Vivarium runs under python {min_version}-{max_version}.\n"
        "You are running python {py_version}".format(
            min_version=min_version, max_version=max_version, py_version=py_version
        )
    )
    print(error, file=sys.stderr)
    sys.exit(1)

from pathlib import Path

from setuptools import find_packages, setup

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    src_dir = base_dir / "src"

    about = {}
    with (src_dir / "vivarium" / "__about__.py").open() as f:
        exec(f.read(), about)

    with (base_dir / "README.rst").open() as f:
        long_description = f.read()

    install_requirements = [
        "layered_config_tree>=2.1.0",
        "numpy<2.0.0",
        "pandas",
        "pyyaml>=5.1",
        "scipy",
        "click",
        "tables",
        "networkx",
        "loguru",
        "pyarrow",
        "dill",
        # Type stubs
        "pandas-stubs",
        "networkx-stubs",
        "types-PyYAML",
    ]

    setup_requires = ["setuptools_scm"]

    interactive_requirements = [
        "IPython",
        "ipywidgets",
        "jupyter",
    ]

    test_requirements = [
        "pytest",
        "pytest-cov",
        "pytest-mock",
        "vivarium_testing_utils",
    ]

    lint_requirements = [
        "black==22.3.0",
        "isort==5.13.2",
        "mypy",
    ]

    doc_requirements = [
        "sphinx>=4.0,<8.0.0",
        "sphinx-rtd-theme>=0.6",
        "sphinx-click",
        "IPython",
        "matplotlib",
        "sphinxcontrib-video",
        "sphinx-autodoc-typehints",
    ]

    setup(
        name=about["__title__"],
        description=about["__summary__"],
        long_description=long_description,
        license=about["__license__"],
        url=about["__uri__"],
        author=about["__author__"],
        author_email=about["__email__"],
        classifiers=[
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
            "Natural Language :: English",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: POSIX",
            "Operating System :: POSIX :: BSD",
            "Operating System :: POSIX :: Linux",
            "Operating System :: Microsoft :: Windows",
            "Programming Language :: Python",
            "Programming Language :: Python :: Implementation :: CPython",
            "Topic :: Education",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Artificial Life",
            "Topic :: Scientific/Engineering :: Mathematics",
            "Topic :: Scientific/Engineering :: Medical Science Apps.",
            "Topic :: Scientific/Engineering :: Physics",
            "Topic :: Software Development :: Libraries",
        ],
        package_dir={"": "src"},
        packages=find_packages(where="src"),
        include_package_data=True,
        install_requires=install_requirements,
        tests_require=test_requirements,
        extras_require={
            "docs": doc_requirements,
            "test": test_requirements,
            "interactive": interactive_requirements,
            "dev": doc_requirements
            + test_requirements
            + lint_requirements
            + interactive_requirements,
        },
        entry_points="""
                [console_scripts]
                simulate=vivarium.interface.cli:simulate
            """,
        zip_safe=False,
        use_scm_version={
            "write_to": "src/vivarium/_version.py",
            "write_to_template": '__version__ = "{version}"\n',
            "tag_regex": r"^(?P<prefix>v)?(?P<version>[^\+]+)(?P<suffix>.*)?$",
        },
        setup_requires=setup_requires,
    )
