#Create __init__
import numpy as np 
import autodiff

class Config:
    def __init__(self, mode='forward'):
        assert mode in ['forward', 'reverse'], "The mode should be either forward or reverse"
        self._mode = mode
        self.reverse_graph = [] if self._mode == 'reverse' else None
    
    def __repr__(self):
        msg = "The current ad mode is {}. The reverse_graph attribute is set to {}".format(self.mode, self.reverse_graph)
        return msg

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, input_mode):
        assert input_mode in [
            'forward', 'reverse'], "The mode should be either forward or reverse"
        self._mode = input_mode
        self.reverse_graph = [] if self._mode == 'reverse' else None
        print(self)


    #@classmethod
    #def set_mode(self, mode):
      
config = Config() 
#Initialize the config with the forward mode. 
# If ever we want to do reverse mode AD, we'll have to to set it manually. 

