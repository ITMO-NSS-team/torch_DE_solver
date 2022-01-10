#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 13:36:47 2021

@author: mike_ubuntu
"""

from setuptools import setup, find_packages
from os.path import dirname, join
from pathlib import Path
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

name = 'TEDEouS' # Add name
version= '0.0.3' # Add version
description = 'TEDEouS - Torch Exhaustive Differential Equations Solver. Differential equation solver, based on pytorch library'
author = 'Alexander Hvatov'
author_email = 'itmo.nss.team@gmail.com' # add email
url = 'https://github.com/ITMO-NSS-team/torch_DE_solver'

def read(*names, **kwargs):
    with open(
            join(dirname(__file__), *names),
            encoding=kwargs.get('encoding', 'utf8')
    ) as fh:
        return fh.read()


def extract_requirements(file_name):
    return [r for r in read(file_name).split('\n') if r and not r.startswith('#')]


def get_requirements():
    requirements = extract_requirements('requirements.txt')
    return requirements

setup(
      name = name,
      version = version,
      description = description,
      author = author,
      author_email = author_email,
      url = url,
      classifiers = [      
              'Development Status :: 3 - Alpha',
              'Programming Language :: Python :: 3',
              'License :: OSI Approved :: MIT License',
              'Operating System :: OS Independent',
      ],
      packages = find_packages(include = ['TEDEouS']), 
      python_requires =' >=3.8'
      )
