#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import List

import setuptools

# The directory containing this file
HERE = Path(__file__).parent.resolve()

# The text of the README file
NAME = 'TEDEouS'

version_info = {}
with open('./tedeous/version.py') as fp:
    exec(fp.read(), version_info)
VERSION = version_info['__version__']

AUTHOR = 'AH and NSS Lab'
SHORT_DESCRIPTION = 'TEDEouS - Torch Exhaustive Differential Equations Solver. Differential equation solver, based on pytorch library'
README = Path(HERE, 'README.rst').read_text(encoding='utf-8')
URL = 'https://github.com/ITMO-NSS-team/torch_DE_solver'
REQUIRES_PYTHON = '>=3.8'
LICENSE = 'MIT License'


def _readlines(*names: str, **kwargs) -> List[str]:
    """
    Reads lines from a file, processing each line to remove leading/trailing whitespace, which is useful for preparing configuration files or data files used in defining and solving differential equations with neural networks.
    
        Args:
          *names: Path segments to the file. Specifies the location of the file to be read.
          **kwargs: Keyword arguments. May include 'encoding' to specify the file encoding.
    
        Returns:
          A list of strings, where each string is a line from the file with leading/trailing whitespace removed.
    """
    encoding = kwargs.get('encoding', 'utf-8')
    lines = Path(__file__).parent.joinpath(*names).read_text(encoding=encoding).splitlines()
    return list(map(str.strip, lines))


def _extract_requirements(file_name: str):
    """
    Extracts the functional requirements for defining and training a neural network model to solve a differential equation.
    
    This method reads a file, typically a 'requirements.txt' or similar, and returns a list of non-empty lines that do not start with '#'.
    These lines represent the necessary packages or dependencies required to set up the neural network-based differential equation solver.
    By extracting these requirements, the project ensures that all necessary components are available for defining, training, and evaluating the neural network models.
    
    Args:
        file_name (str): The name of the file to read.
    
    Returns:
        list[str]: A list of strings, where each string is a requirement.
    """
    return [line for line in _readlines(file_name) if line and not line.startswith('#')]


def _get_requirements(req_name: str):
    """
    Extracts and returns a list of requirements based on the provided name.
    
    This function serves as a utility to retrieve specific requirements,
    allowing for modular access to necessary components for defining and
    solving differential equations with neural networks.
    
    Args:
        req_name (str): The name of the requirement to extract.
    
    Returns:
        list: A list of requirements extracted based on the given name.
    """
    requirements = _extract_requirements(req_name)
    return requirements


setuptools.setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email='itmo.nss.team@gmail.com',
    description=SHORT_DESCRIPTION,
    long_description=README,
    long_description_content_type='text/x-rst',
    url=URL,
    python_requires=REQUIRES_PYTHON,
    license=LICENSE,
    packages=setuptools.find_packages(exclude=['test*']),
    include_package_data=True,
    install_requires=_get_requirements('requirements.txt'),
    #extras_require={
     #   key: _get_requirements(Path('other_requirements', f'{key}.txt'))
     #   for key in ('docs', 'examples', 'extra', 'profilers')
    #},
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
