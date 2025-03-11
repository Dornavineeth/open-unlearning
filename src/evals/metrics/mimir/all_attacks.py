"""
    Enum class for attacks. Also contains the base attack class.
"""

from enum import Enum
import torch

# Attack definitions
class AllAttacks(str, Enum):
    LOSS = "loss"
    REFERENCE_BASED = "ref"
    ZLIB = "zlib"
    MIN_K = "min_k"
    MIN_K_PLUS_PLUS = "min_k++"
    NEIGHBOR = "ne"
    GRADNORM = "gradnorm"
    RECALL = "recall"
    DC_PDD = "dc_pdd" 
    # QUANTILE = "quantile" # Uncomment when tested implementation is available


# Base attack class
class Attack:
    def __init__(self, model):
        self.model = model
    
    def preprocess_batch(self, batch):
        """Process a batch to compute required statistics. Return dict of required values."""
        raise NotImplementedError
        
    def score_sample(self, sample_values):
        """Score a single sample using precomputed values from preprocess_batch."""
        raise NotImplementedError

    def attack(self, probs=None, tokens=None, **kwargs):
        """Process batch data and score individual samples."""
        batch_values = self.preprocess_batch(tokens)
        score = self.score_sample(batch_values)
        return score
