
from typing import List

import numpy as np
from .token_param import TokenParam

class TokenStat:
    def __init__(self, token_param: TokenParam):
        self.token_param = token_param        
        
    def __repr__(self):        
        return "<TokenStat {:}: p={:.1f}, μ={:.1f}, ρ={:.1f}>".format(
            self.token(),
            self.token_param.full_prob,
            self.semanticness(),
            self.contextualness()
        )
    
    def token(self):
        return self.token_param.token_text
    
    def semanticness(self):
        tp = self.token_param
        return -np.log(tp.replaced_prob / tp.full_prob)
    
    def contextualness(self):
        tp = self.token_param
        return -np.log(tp.defmasked_prob / tp.full_prob)

class SequenceStat:
    def __init__(self, token_params: List[TokenParam]):
        self.token_params = token_params
    
    def __repr__(self):
        return ("<SequenceStat Full/Masked/Replaced" + 
            ": {:.2f}/{:.2f}/{:.2f}>".format(
                self.full_nll(),
                self.masked_nll(),
                self.replaced_nll()
        ))
    
    def full_nll(self):
        full_probs = np.log([x.full_prob for x in self.token_params])
        return -full_probs.mean()
    
    def random_nll(self):
        random_probs = np.log([x.random_prob for x in self.token_params])
        return -random_probs.mean()
    
    def replaced_nll(self):
        replaced_probs = np.log([x.replaced_prob for x in self.token_params])
        return -replaced_probs.mean()
    
    def masked_nll(self):
        masked_probs = np.log([x.defmasked_prob for x in self.token_params])
        return -masked_probs.mean()