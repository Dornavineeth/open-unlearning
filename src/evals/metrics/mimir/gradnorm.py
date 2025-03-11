"""
    Gradient-norm attack. Proposed for MIA in multiple settings, and particularly 
    experimented for pre-training data and LLMs in https://arxiv.org/abs/2402.17012
"""
import torch
import numpy as np
from evals.metrics.mimir.all_attacks import Attack
from evals.metrics.utils import tokenwise_logprobs

class GradNormAttack(Attack):
    def __init__(self, model, p):
        super().__init__(model)
        if p not in [1, 2, float('inf')]:
            raise ValueError(f"Invalid p-norm value: {p}")
        self.p = p

    def preprocess_batch(self, tokens):
        """Compute gradients w.r.t model parameters."""
        self.model.zero_grad()
        log_probs = tokenwise_logprobs(self.model, tokens, grad=True)
        loss = -torch.mean(log_probs)
        loss.backward()
        
        grad_norms = []
        for param in self.model.parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.detach().norm(p=self.p))
        grad_norm = torch.stack(grad_norms).mean()
        
        self.model.zero_grad()
        return {'grad_norm': grad_norm}

    def score_sample(self, sample_values):
        """Return negative gradient norm as the attack score."""
        return -sample_values['grad_norm'].cpu().numpy()
