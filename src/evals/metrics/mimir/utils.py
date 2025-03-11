from evals.metrics.mimir.all_attacks import AllAttacks
from evals.metrics.mimir.loss import LOSSAttack
from evals.metrics.mimir.reference import ReferenceAttack
from evals.metrics.mimir.zlib import ZLIBAttack
from evals.metrics.mimir.min_k import MinKProbAttack
from evals.metrics.mimir.min_k_plus_plus import MinKPlusPlusAttack
from evals.metrics.mimir.gradnorm import GradNormAttack
from evals.metrics.mimir.recall import ReCaLLAttack
# from evals.metrics.mimir.neighborhood import NeighborhoodAttack
# from evals.metrics.mimir.quantile import QuantileAttack  # Enable when fully tested

def get_attacker(attack: str):
    mapping = {
        AllAttacks.LOSS: LOSSAttack,
        AllAttacks.REFERENCE_BASED: ReferenceAttack,
        AllAttacks.ZLIB: ZLIBAttack,
        AllAttacks.MIN_K: MinKProbAttack,
        AllAttacks.MIN_K_PLUS_PLUS: MinKPlusPlusAttack,
        AllAttacks.GRADNORM: GradNormAttack,
        AllAttacks.RECALL: ReCaLLAttack,
        # AllAttacks.QUANTILE: QuantileAttack  # Enable when fully tested
    }
    attack_cls = mapping.get(attack, None)
    if attack_cls is None:
        raise ValueError(f"Attack {attack} not found")
    return attack_cls
