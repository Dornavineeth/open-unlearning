"""
    Reference-based attacks.
"""
from evals.metrics.mimir.all_attacks import Attack
from evals.metrics.utils import evaluate_probability

class ReferenceAttack(Attack):
    def __init__(self, model, reference_model):
        super().__init__(model)
        self.reference_model = reference_model

    def preprocess_batch(self, tokens):
        """Compute loss for both target and reference models."""
        target_results = evaluate_probability(self.model, tokens)
        ref_results = evaluate_probability(self.reference_model, tokens)
        return {
            'target_results': target_results,
            'ref_results': ref_results
        }

    def score_sample(self, sample_values):
        """Score using difference between target and reference model losses."""
        target_loss = sample_values['target_results'][0]['avg_loss']
        ref_loss = sample_values['ref_results'][0]['avg_loss']
        return target_loss - ref_loss
