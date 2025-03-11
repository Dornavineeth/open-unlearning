import numpy as np
from torch.utils.data import DataLoader

def run_mia_attack(model, data, collator, attack_class, **kwargs):
    """Generalized function to run MIA attacks and return aggregated results."""
    dataloader = DataLoader(data, batch_size=1, collate_fn=collator)
    attacker = attack_class(model)
    
    scores_by_index = {
        f"sample_{idx}": {"score": attacker.attack(tokens=batch)}
        for idx, batch in enumerate(dataloader)
    }
    
    return {
        "agg_value": np.mean([v["score"] for v in scores_by_index.values()]),
        "value_by_index": scores_by_index
    }

@unlearning_metric(name="mia_min_k")
def mia_min_k(model, **kwargs):
    from evals.metrics.mimir.min_k import MinKProbAttack
    k = kwargs.get("k", 0.1)
    return run_mia_attack(model, kwargs["data"], kwargs["collators"], lambda m: MinKProbAttack(m, k=k))


@unlearning_metric(name="mia_min_k++")
def mia_min_k_plus_plus(model, **kwargs):
    from evals.metrics.mimir.min_k_plus_plus import MinKPlusPlusAttack
    k = kwargs.get("k", 0.1)
    return run_mia_attack(model, kwargs["data"], kwargs["collators"], lambda m: MinKPlusPlusAttack(m, k=k))


@unlearning_metric(name="mia_gradnorm")
def mia_gradnorm(model, **kwargs):
    from evals.metrics.mimir.gradnorm import GradNormAttack
    p = kwargs.get("p", float('inf'))
    return run_mia_attack(model, kwargs["data"], kwargs["collators"], lambda m: GradNormAttack(m, p=p))


@unlearning_metric(name="mia_loss")
def mia_loss(model, **kwargs):
    from evals.metrics.mimir.loss import LOSSAttack
    return run_mia_attack(model, kwargs["data"], kwargs["collators"], LOSSAttack)


@unlearning_metric(name="mia_zlib")
def mia_zlib(model, **kwargs):
    from evals.metrics.mimir.zlib import ZLIBAttack
    return run_mia_attack(model, kwargs["data"], kwargs["collators"], ZLIBAttack)


@unlearning_metric(name="mia_reference")
def mia_reference(model, **kwargs):
    from evals.metrics.mimir.reference import ReferenceAttack
    if "reference_model" not in kwargs:
        raise ValueError("Reference model must be provided in kwargs under 'reference_model'")
    return run_mia_attack(model, kwargs["data"], kwargs["collators"], ReferenceAttack, reference_model=kwargs["reference_model"])


@unlearning_metric(name="mia_recall")
def mia_recall(model, **kwargs):
    from evals.metrics.mimir.recall import ReCaLLAttack
    recall_dict = kwargs.get("recall_dict", {})
    return run_mia_attack(model, kwargs["data"], kwargs["collators"], lambda m: ReCaLLAttack(m, recall_dict=recall_dict))