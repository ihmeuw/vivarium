from setuptools import setup, find_packages


setup(
    name='vivarium',
    version='0.5',
    packages=find_packages(),
    include_package_data=True,
    entry_points="""
            [console_scripts]
            simulate=vivarium.interface.cli:simulate
        """,
    install_requires=[
        'pandas',
        'numpy',
        'scipy',
        'pyaml',
        'click',
    ],
    extras_require={
        'testing': [
            'pytest',
            'pytest-mock',
        ]
    }

)
