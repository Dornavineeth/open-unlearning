"""
    Straight-forward LOSS attack, as described in https://ieeexplore.ieee.org/abstract/document/8429311
"""
import torch as ch
from evals.metrics.mimir.all_attacks import Attack
from evals.metrics.utils import evaluate_probability

class LOSSAttack(Attack):

    def __init__(self, model):
        super().__init__(model)

    def preprocess_batch(self, tokens):
        """Compute probabilities and losses for the batch."""
        return {
            'eval_results': evaluate_probability(self.model, tokens)
        }
        
    def score_sample(self, sample_values):
        """Return the average loss for the sample."""
        return sample_values['eval_results'][0]['avg_loss']
