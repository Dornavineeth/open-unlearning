"""
    Min-k % Prob Attack: https://arxiv.org/pdf/2310.16789.pdf
"""
import torch
import numpy as np
from evals.metrics.mimir.all_attacks import Attack
from evals.metrics.utils import tokenwise_logprobs

class MinKProbAttack(Attack):
    def __init__(self, model, k=0.2):
        super().__init__(model)
        self.k = k
        
    def preprocess_batch(self, tokens):
        """Get token-wise log probabilities for the batch."""
        return {
            'log_probs': tokenwise_logprobs(self.model, tokens, grad=False)
        }

    def score_sample(self, sample_values):
        """Score a single sample using the min-k% probability attack."""
        lp = sample_values['log_probs'].cpu().numpy()
        if lp.size == 0:
            return 0
            
        ngram_avgs = np.array(lp)  # Each logprob is its own n-gram with window=1
        num_k = max(1, int(len(ngram_avgs) * self.k))
        sorted_vals = np.sort(ngram_avgs)
        return -np.mean(sorted_vals[:num_k])
