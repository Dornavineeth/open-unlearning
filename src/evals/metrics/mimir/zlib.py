"""
    zlib-normalization Attack: https://www.usenix.org/system/files/sec21-carlini-extracting.pdf
"""

import torch as ch
import zlib

from evals.metrics.mimir.all_attacks import Attack
from evals.metrics.utils import evaluate_probability, extract_target_texts_from_processed_data


class ZLIBAttack(Attack):

    def __init__(self, model, tokenizer):
        super().__init__(model)
        self.tokenizer = tokenizer

    def preprocess_batch(self, tokens):
        """Compute loss and extract text for compression."""
        eval_results = evaluate_probability(self.model, tokens)
        texts = extract_target_texts_from_processed_data(self.tokenizer, tokens)
        return {
            'eval_results': eval_results,
            'texts': texts
        }

    def score_sample(self, sample_values):
        """Score using loss normalized by compressed text length."""
        loss = sample_values['eval_results'][0]['avg_loss']
        text = sample_values['texts'][0]
        zlib_entropy = len(zlib.compress(text.encode("utf-8")))
        return loss / zlib_entropy
