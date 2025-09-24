import torch
from tedeous.callbacks.callback import Callback
import os



class SaveModel(Callback):
    """
    Class for saving model during train
    """

    def __init__(self,
                 path_to_folder: str,
                 every_step : int = 1):
        """
        Initializes the SaveModel callback.
        
        This callback is designed to save the model's state periodically during training. 
        It ensures that progress is not lost and allows for the retrieval of the model 
        at different stages of training for analysis or further use.
        
        Args:
            path_to_folder (str): The path to the folder where the model checkpoints will be saved.
            every_step (int): The frequency (in steps) at which the model should be saved. Defaults to 1, meaning the model is saved every step.
        
        Returns:
            None
        """
        super().__init__()
        self.path_to_folder = path_to_folder
        self.every_step = every_step
    
    def save_model(self):
        """
        Saves the trained neural network model to a file. This allows for later use without retraining, preserving the learned approximation of the differential equation's solution.
        
                Args:
                    self: The SaveModel instance.
        
                Returns:
                    None. The model is saved as 'model-{t}.pt' in the specified folder. 't' represents the training step, allowing you to track the model's evolution during training.
        """
        model_name = "model-{}.pt".format(self.model.t-1)
        save_path = os.path.join(self.path_to_folder, model_name)
        torch.save(self.model.net, save_path)

    def on_epoch_end(self, logs=None):
        """
        Saves the model's weights at specific training intervals. This ensures that progress is preserved during the training process of the neural network used to approximate the solution of a differential equation.
        
                Args:
                    logs (dict, optional): Metric results for the current epoch. Defaults to None.
        
                Returns:
                    None
        """
        if (self.model.t-1) % self.every_step == 0:
            self.save_model()