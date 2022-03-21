# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 14:13:37 2022

@author: Sashka
"""

import json

def read_config(name):
    with open(name, 'r') as config_file:
        config_data = json.load(config_file)
    return config_data




