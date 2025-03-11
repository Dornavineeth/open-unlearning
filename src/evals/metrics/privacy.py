import numpy as np
from scipy.stats import ks_2samp
from torch.utils.data import DataLoader

from evals.metrics.base import unlearning_metric, logger
from evals.metrics.utils import run_batchwise_evals, tokenwise_logprobs


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
        retain_tr_stats = np.array(
            [
                evals["score"]
                for evals in kwargs["reference_logs"]["retain_model_logs"]["retain"][
                    "value_by_index"
                ].values()
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


@unlearning_metric(name="minKpc_negative_logprob")
def minKpc_negative_logprob(model, **kwargs):
    """Compute the min-k percentile average of token-wise model probabilities by data points"""
    
    def min_k_logic_batch(model, batch, percentile):
        """Compute minK% attack score for each sample in a batch."""
        token_wise_logprobs = tokenwise_logprobs(model, batch)
        mink_means = []
        for result in token_wise_logprobs:
            scores = np.sort(result.cpu().numpy())
            top_k = max(1, int(percentile / 100 * len(scores)))
            mink_mean = -1 * np.mean(scores[:top_k])
            mink_means.append(mink_mean)
        return [{"score": float(neglogprob)} for neglogprob in mink_means]
    
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]

    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)

    fun_args = {"percentile": kwargs["percentile_K"]}
    return {
        "value_by_index": run_batchwise_evals(
            model,
            dataloader,
            min_k_logic_batch,
            fun_args,
            "Calculating avg token-wise lowest K% percentile logprobs across batches",
        )
    }


@unlearning_metric(name="privleak")
def privleak(model, **kwargs): # re-name to privleak
    """Compares a statistic computed on a model to a reference retain model"""
    score = kwargs["pre_compute"]["target_model_stat"]["agg_value"]
    ref = kwargs["pre_compute"]["ref_model_stat"]["agg_value"]
    return {'agg_value': (score-ref)/ref*100}