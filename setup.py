import sys
if sys.version_info < (3, 6) or sys.version_info >= (3, 9):
    # Python 3.5 does not support f-strings
    py_version = '.'.join([str(v) for v in sys.version_info[:3]])
    error = ('\n----------------------------------------\n'
            'Error: Vivarium runs under python 3.6-3.8.\n'
            'You are running python {py_version}'.format(py_version = py_version))
    print(error, file=sys.stderr)
    sys.exit(1)


from pathlib import Path

from setuptools import setup, find_packages


if __name__ == "__main__":

    base_dir = Path(__file__).parent
    src_dir = base_dir / 'src'

    about = {}
    with (src_dir / "vivarium" / "__about__.py").open() as f:
        exec(f.read(), about)

    with (base_dir / "README.rst").open() as f:
        long_description = f.read()

    install_requirements = [
        'numpy',
        'pandas',
        'pyyaml>=5.1',
        'scipy',
        'click',
        'tables',
        'networkx',
        'loguru',
    ]

    interactive_requirements = [
        'IPython',
        'ipywidgets',
        'jupyter',
    ]

    test_requirements = [
        'pytest',
        'pytest-mock',
    ]

    doc_requirements = [
        'sphinx>=4.0',
        'sphinx-rtd-theme',
        'sphinx-click',
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
