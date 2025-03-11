"""
    Implementation of the attack proposed in 'Scalable Membership Inference Attacks via Quantile Regression'
    https://arxiv.org/pdf/2307.03694.pdf
"""
import torch as ch
from mimir.models import QuantileReferenceModel, Model
from transformers import TrainingArguments
from sklearn.metrics import mean_squared_error
from transformers import TrainingArguments, Trainer
from datasets import Dataset

from evals.metrics.mimir.all_attacks import Attack
from evals.metrics.utils import evaluate_probability, extract_target_texts_from_processed_data

class QuantileReferenceModel(Model):
    """
        Wrapper for reference model, specifically used for quantile regression
    """
    def __init__(self, config, name: str):
        super().__init__(config)
        self.device = self.config.env_config.device_aux
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(
            name, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            name,
            num_labels=2,
            max_position_embeddings=1024)
        # Modify model's last linear layer to have only 1 output
        self.model.classifier.linear_out = nn.Linear(self.model.classifier.linear_out.in_features, 1)
        self.load_model_properties()


class CustomTrainer(Trainer):
    def __init__(
        self,
        alpha_fpr,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.alpha_fpr = alpha_fpr

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = ch.mean(
            ch.max(
                self.alpha_fpr * (logits - labels),
                (1 - self.alpha_fpr) * (labels - logits),
            )
        )
        return (loss, outputs) if return_outputs else loss


class QuantileAttack(Attack):
    def __init__(self, model, alpha=0.1):
        super().__init__(model)
        self.alpha = alpha
        self._setup_quantile_model()
    
    def _setup_quantile_model(self):
        # Initialize quantile model here
        self.quantile_model = None  # Implement proper initialization
        
    def preprocess_batch(self, tokens):
        """Get evaluation results and texts."""
        eval_results = evaluate_probability(self.model, tokens)
        texts = extract_target_texts_from_processed_data(self.tokenizer, tokens)
        quantile_scores = self._get_quantile_scores(texts)
        return {
            'eval_results': eval_results,
            'quantile_scores': quantile_scores
        }

    def score_sample(self, sample_values):
        """Return quantile score minus target loss."""
        target_loss = sample_values['eval_results'][0]['avg_loss']
        quantile_score = sample_values['quantile_scores'][0]
        return quantile_score - target_loss
