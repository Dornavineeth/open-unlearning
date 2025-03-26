import numpy as np
from scipy.stats import ks_2samp
from evals.metrics.base import unlearning_metric, logger


@unlearning_metric(name="forget_quality")
def forget_quality(model, **kwargs):
    forget_tr_stats = np.array(
        [
            evals["score"]
            for evals in kwargs["pre_compute"]["forget"]["value_by_index"].values()
        ]
    )
    reference_logs = kwargs.get("reference_logs", None)
    if reference_logs:
        reference_logs = reference_logs["retain_model_logs"]
        retain_tr_stats = np.array(
            [
                evals["score"]
                for evals in reference_logs["retain"]["value_by_index"].values()
            ]
        )
        fq = ks_2samp(forget_tr_stats, retain_tr_stats)
        pvalue = fq.pvalue
    else:
        logger.warning(
            "retain_model_logs not provided in reference_logs, setting forget_quality to None"
        )
        pvalue = None
    return {"agg_value": pvalue}


@unlearning_metric(name="privleak")
def privleak(model, **kwargs):
    """Gives a relative comparison of a statistic computed on a model to a reference retain model"""
    score = kwargs["pre_compute"]["forget"]["agg_value"]
    try:
        ref = kwargs["reference_logs"]["retain_model_logs"]["retain"]["agg_value"]
    except Exception as _:
        logger.warning(
            f"retain_model_logs evals not provided for privleak, using default retain auc of {kwargs['ref_value']}"
        )
        ref = kwargs["ref_value"]
    return {'agg_value': (score-ref)/(ref+1e-10)*100}