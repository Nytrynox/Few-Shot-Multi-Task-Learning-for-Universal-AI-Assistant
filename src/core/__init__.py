"""Core meta-learning algorithms and interfaces."""

from .meta_learner import create_meta_learner, MetaLearningAlgorithm, MultiTaskMetaLearner
from .maml import MAMLLearner, MAMLPlusPlus
from .prototypical import PrototypicalNetworks, RelationNetworks

__all__ = [
    "create_meta_learner",
    "MetaLearningAlgorithm", 
    "MultiTaskMetaLearner",
    "MAMLLearner",
    "MAMLPlusPlus",
    "PrototypicalNetworks",
    "RelationNetworks"
]
