import setuptools
from slydm import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="slydm",
    version=__version__,
    author="Patrick Staudt",
    author_email="patrickstaudt1@gmail.com",
    description=(
        "Code used to make a sly determination of dark matter properties using"
        " local circular speed."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/patricks1/dm_den",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'h5py',
        'progressbar',
        'astropy',
        'tabulate',
        'scikit-learn',
        'adjusttext==1.0',
        'cmasher',
        'lmfit'
    ],
)
