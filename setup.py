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
    encoding = kwargs.get('encoding', 'utf-8')
    lines = Path(__file__).parent.joinpath(*names).read_text(encoding=encoding).splitlines()
    return list(map(str.strip, lines))


def _extract_requirements(file_name: str):
    return [line for line in _readlines(file_name) if line and not line.startswith('#')]


def _get_requirements(req_name: str):
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
