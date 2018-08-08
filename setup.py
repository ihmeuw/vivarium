#!/usr/bin/env python
import os

from setuptools import setup, find_packages


if __name__ == "__main__":

    base_dir = os.path.dirname(__file__)
    src_dir = os.path.join(base_dir, "src")

    about = {}
    with open(os.path.join(src_dir, "vivarium", "__about__.py")) as f:
        exec(f.read(), about)

    with open(os.path.join(base_dir, "README.rst")) as f:
        long_description = f.read()

    install_requirements = [
        'numpy',
        'pandas',
        'pyyaml',
        'scipy',
        'click',
        'tables',
    ]

    interactive_requirements = [
        'IPython',
        'ipywidgets',
        'jupyter',
        'graphviz',
    ]

    test_requirements = [
        'pytest',
        'pytest-mock',
    ]

    doc_requirements = [
        'sphinx',
        'sphinx-autodoc-typehints',
        'sphinx-rtd-theme',
        'IPython',
        'matplotlib'
    ]

    setup(
        name=about['__title__'],
        version=about['__version__'],

        description=about['__summary__'],
        long_description=long_description,
        license=about['__license__'],
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
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: Implementation :: CPython",
            "Topic :: Education",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Artificial Life",
            "Topic :: Scientific/Engineering :: Mathematics",
            "Topic :: Scientific/Engineering :: Medical Science Apps.",
            "Topic :: Scientific/Engineering :: Physics",
            "Topic :: Software Development :: Libraries",
        ],

        package_dir={'': 'src'},
        packages=find_packages(where='src'),
        include_package_data=True,

        install_requires=install_requirements,
        tests_require=test_requirements,
        extras_require={
            'docs': doc_requirements,
            'test': test_requirements,
            'interactive': interactive_requirements,
            'dev': doc_requirements + test_requirements + interactive_requirements,
        },

        entry_points="""
                [console_scripts]
                simulate=vivarium.interface.cli:simulate
            """,

        zip_safe=False,
    )
