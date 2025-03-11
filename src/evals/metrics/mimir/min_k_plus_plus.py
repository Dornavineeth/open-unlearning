import torch as ch
import numpy as np
from evals.metrics.mimir.all_attacks import Attack
from evals.metrics.utils import tokenwise_logprobs

class MinKPlusPlusAttack(Attack):

    def __init__(self, model, k=0.2):
        super().__init__(config=None, target_model=model, ref_model=None)
        self.k = k

    @ch.no_grad()
    def _attack(self, probs, tokens=None, **kwargs):
        """
        Min-K%++ Attack.
        For each sample, obtain token-wise log probabilities and normalize them using a weighted 
        mean and standard deviation computed from the exponentiated log probabilities.
        Then, with a sliding window (size=1, stride=1) compute n-gram averages (which here are the 
        normalized log probabilities themselves), sort these averages, and compute the mean of the 
        lowest k% values. The final score is the negative average over all samples.
        """
        k = kwargs.get("k", 0.2)
        window = kwargs.get("window", 1)   # fixed to 1
        stride = kwargs.get("stride", 1)   # fixed to 1

        # Get token-wise log probabilities as a list of tensors.
        log_probs_list = tokenwise_logprobs(self.target_model, tokens, grad=False)
        
        sample_scores = []
        for lp_tensor in log_probs_list:
            lp = lp_tensor.cpu().numpy()  # shape: (L,)
            if lp.size == 0:
                continue

            # Compute weighted normalization using the token probabilities:
            p = np.exp(lp)  # convert log-probs to probabilities
            sum_p = np.sum(p)
            if sum_p == 0:
                weighted_mu = np.mean(lp)
                weighted_var = np.var(lp)
            else:
                weighted_mu = np.sum(p * lp) / sum_p
                weighted_sq = np.sum(p * lp**2) / sum_p
                weighted_var = weighted_sq - weighted_mu**2

            # Avoid division by zero:
            if weighted_var <= 0:
                normalized = lp - weighted_mu
            else:
                normalized = (lp - weighted_mu) / np.sqrt(weighted_var)

            # With window=1 and stride=1, each token is an n-gram average.
            ngram_avgs = normalized  # essentially, one value per token
            ngram_avgs = np.array(ngram_avgs)
            num_k = max(1, int(len(ngram_avgs) * k))
            sorted_vals = np.sort(ngram_avgs)
            sample_scores.append(np.mean(sorted_vals[:num_k]))
            
        return -np.mean(sample_scores) if sample_scores else 0

    def preprocess_batch(self, tokens):
        """Get token-wise log probabilities for the batch."""
        return {
            'log_probs': tokenwise_logprobs(self.target_model, tokens, grad=False)
        }

    def score_sample(self, sample_values):
        """Score using min-k++ probability attack with probability-weighted normalization."""
        lp = sample_values['log_probs'].cpu().numpy()
        if lp.size == 0:
            return 0

        # Compute weighted normalization using the token probabilities
        p = np.exp(lp)
        sum_p = np.sum(p)
        if sum_p == 0:
            weighted_mu = np.mean(lp)
            weighted_var = np.var(lp)
        else:
            weighted_mu = np.sum(p * lp) / sum_p
            weighted_sq = np.sum(p * lp**2) / sum_p
            weighted_var = weighted_sq - weighted_mu**2

        # Normalize logprobs
        if weighted_var <= 0:
            normalized = lp - weighted_mu
        else:
            normalized = (lp - weighted_mu) / np.sqrt(weighted_var)

        num_k = max(1, int(len(normalized) * self.k))
        sorted_vals = np.sort(normalized)
        return -np.mean(sorted_vals[:num_k])