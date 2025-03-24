"""
    Attack implementations.
"""
from evals.metrics.base import unlearning_metric
from evals.metrics.mia.loss import LOSSAttack
from evals.metrics.mia.min_k import MinKProbAttack
from evals.metrics.mia.min_k_plus_plus import MinKPlusPlusAttack
from evals.metrics.mia.gradnorm import GradNormAttack
from evals.metrics.mia.zlib import ZLIBAttack
from evals.metrics.mia.reference import ReferenceAttack

from evals.metrics.mia.utils import mia_auc


@unlearning_metric(name="mia_loss")
def mia_loss(model, **kwargs):
    return mia_auc(LOSSAttack, model, data=kwargs["data"],
                   collator=kwargs["collators"],
                   batch_size=kwargs["batch_size"])

@unlearning_metric(name="mia_min_k")
def mia_min_k(model, **kwargs):
    return mia_auc(MinKProbAttack, model, data=kwargs["data"],
                   collator=kwargs["collators"],
                   batch_size=kwargs["batch_size"],
                   k=kwargs["k"])

@unlearning_metric(name="mia_min_k_plus_plus")
def mia_min_k_plus_plus(model, **kwargs):
    return mia_auc(MinKPlusPlusAttack, model, data=kwargs["data"],
                   collator=kwargs["collators"],
                   batch_size=kwargs["batch_size"],
                   k=kwargs["k"])

@unlearning_metric(name="mia_gradnorm")
def mia_gradnorm(model, **kwargs):
    return mia_auc(GradNormAttack, model, data=kwargs["data"],
                   collator=kwargs["collators"],
                   batch_size=kwargs["batch_size"],
                   p=kwargs["p"])

@unlearning_metric(name="mia_zlib")
def mia_zlib(model, **kwargs):
    return mia_auc(ZLIBAttack, model, data=kwargs["data"],
                   collator=kwargs["collators"],
                   batch_size=kwargs["batch_size"],
                   tokenizer=kwargs.get("tokenizer"))

@unlearning_metric(name="mia_reference")
def mia_reference(model, **kwargs):
    if "reference_model" not in kwargs:
        raise ValueError("Reference model must be provided in kwargs")
    return mia_auc(ReferenceAttack, model, data=kwargs["data"],
                   collator=kwargs["collators"],
                   batch_size=kwargs["batch_size"],
                   reference_model=kwargs["reference_model"])