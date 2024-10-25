from setuptools import setup, find_packages

setup(
    name="schrodinger_engine",
    version="0.1",
    description="A project simulating the periodical unidimensionnal Scrödinger equation.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    # url="https://github.com/votreusername/maggiver",
    author="Etienne Rosin",
    author_email="etienne.alex.rosin@cern.ch",
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'tqdm',
        'cmasher'  
        ],
    include_package_data=True,
)
