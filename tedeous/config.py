from email.policy import default
from typing import Union, Optional
import json


def read_config(name: str) -> json:
    """
    Reads a configuration file containing parameters for defining and training neural network models used to solve differential equations.
    
    This function is essential for setting up the neural network architecture, training parameters, and other configurations required to approximate solutions to differential equations.
    
    Args:
        name (str): The path to the JSON configuration file.
    
    Returns:
        json: A JSON object containing the configuration data, which dictates the structure and training process of the neural network model.
    """
    with open(name, 'r') as config_file:
        config_data = json.load(config_file)
    return config_data

DEFAULT_CONFIG = """
{
"Optimizer": {
"learning_rate":1e-4,
"lambda_bound":10,
"optimizer":"Adam"
},
"Cache":{
"use_cache":true,
"cache_dir":"../cache/",
"cache_verbose":false,
"save_always":false,
"model_randomize_parameter":0
},
"NN":{
"batch_size":null,
"lp_par":null,
"grid_point_subset":["central"],
"h":0.001
},
"Verbose":{
	"verbose":true,
	"print_every":null
},
"StopCriterion":{
"eps":1e-5,
"tmin":1000,
"tmax":1e5 ,
"patience":5,
"loss_oscillation_window":100,
"no_improvement_patience":1000   	
},
"Matrix":{
"lp_par":null,
"cache_model":null
}
}
"""

default_config = json.loads(DEFAULT_CONFIG)


def check_module_name(module_name: str) -> bool:
    """
    Checks if the given module name is valid.
    
    This function verifies if the provided module name exists within the
    default configuration. This ensures that the neural network-based solver
    is using a correctly defined module, which is crucial for setting up
    the differential equation problem and its solution approach.
    
    Args:
        module_name: The name of the module to check (first level of config).
    
    Returns:
        True if the module name is found in the default configuration,
        False otherwise.
    """
    if module_name in default_config.keys():
        return True
    else:
        return False


def check_param_name(module_name: str, param_name: str) -> bool:
    """
    Checks if a given parameter name exists within a specified module in the default configuration.
    
    This function verifies whether a parameter is valid for a particular module.
    It ensures that the configuration being accessed during the neural network's 
    differential equation solving process is correctly structured, preventing errors 
    due to misconfiguration.
    
    Args:
        module_name: The name of the module (first level key in the config).
        param_name: The name of the parameter to check within the module.
    
    Returns:
        True if the parameter name exists within the specified module in the default configuration, False otherwise.
    """
    if param_name in default_config[module_name].keys():
        return True
    else:
        return False

class Config:
    """
    Represents a configuration object for the solver.
    
        The configuration can be initialized with default values and updated
        from a custom configuration file.
    
        Attributes:
            config_path: Path to a custom configuration file.
    """

    def __init__(self, *args):
        """
        Initializes the configuration for solving differential equations using neural networks.
        
                This method sets up the configuration parameters, starting with a default configuration.
                If a path to a custom configuration file is provided, it attempts to load and merge those parameters
                into the default configuration, allowing users to customize the solving process. This ensures
                flexibility in defining network architectures, training parameters, and other settings relevant
                to the neural network-based differential equation solver. The method performs checks on module and
                parameter names within the custom configuration to ensure validity and prevent errors.
        
                Args:
                    *args: Accepts an optional path to a custom configuration file as the first argument. Additional arguments are ignored.
        
                Returns:
                    None. The method modifies the `self.params` attribute, which holds the configuration used by the solver.
        """

        self.params = default_config
        if len(args) == 1:
            try:
                custom_config = read_config(args[0])
            except Exception:
                print('Error reading config. Default config assumed.')
                custom_config = default_config
            for module_name in custom_config.keys():
                if check_module_name(module_name):
                    for param in custom_config[module_name].keys():
                        if check_param_name(module_name, param):
                            self.params[module_name][param] = custom_config[module_name][param]
                        else:
                            print('Wrong parameter name: ok.wrong for {}.{}. Defalut parameters assumed.'.format(
                                module_name, param))
                else:
                    print(
                        'Wrong module name: wrong.maybeok for {}.smth. Defalut parameters assumed.'.format(module_name))

        elif len(args) > 1:
            print('Too much initialization args, using default config')

    def set_parameter(self, parameter_string: str, value: Union[bool, float, int, None]):
        """
        Allows direct modification of individual configuration parameters, bypassing the need to load an entire configuration file. This is useful for fine-tuning specific aspects of the neural network-based differential equation solver.
        
                Args:
                    parameter_string: A string specifying the parameter to modify, in the format 'module.parameter'.
                    value: The new value for the specified parameter.  Can be a boolean, float, integer, or None.
        
                Returns:
                    None. The method modifies the configuration parameters in place.
        """

        module_name, param = parameter_string.split('.')
        if check_module_name(module_name):
            if check_param_name(module_name, param):
                self.params[module_name][param] = value
            else:
                print(
                    'Wrong parameter name: ok.wrong for {}.{}. Defalut parameters assumed.'.format(module_name, param))
        else:
            print('Wrong module name: wrong.maybeok for {}.smth. Defalut parameters assumed.'.format(module_name))
