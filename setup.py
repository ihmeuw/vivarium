from setuptools import setup, find_packages


setup(
    name='vivarium',
    version='0.5',
    packages=find_packages(),
    include_package_data=True,
    scripts=['simulate'],
    install_requires=[
        'pandas',
        'numpy',
        'scipy',
        'pyaml',
    ]
)
