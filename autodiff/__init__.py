#Create __init__
import numpy as np 
import autodiff

class Config:
    def __init__(self, mode='forward'):
        assert mode in ['forward', 'reverse'], "The mode should be either forward or reverse"
        self.mode = mode
        self.reverse_graph = [] if self.mode == 'reverse' else None
    
    def __repr__(self):
        msg = "The current ad mode is {}. The reverse_graph attribute is set to {}".format(self.mode, self.reverse_graph)
        return msg

    #@classmethod
    #def set_mode(self, mode):
        
config = Config()