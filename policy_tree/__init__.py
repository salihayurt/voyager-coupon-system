"""
Policy Tree Cohorting Module for Voyager Coupon System

This module provides cohort-based policy recommendations using shallow decision trees
trained on Voyager's recommendation outputs. It groups users into explainable cohorts
with feasible actions that respect segment constraints.
"""

__version__ = "1.0.0"
__author__ = "Voyager Team"

from .constraints import ALLOWED_ACTIONS, SEGMENT_CONSTRAINTS
from .feasible import choose_action, choose_actions_batch
from .tags import make_tags, extract_tags_batch
from .features import build_features, bin_scores
from .train import fit_policy_tree, save_model, load_model
from .inference import generate_cohorts, load_cohorts_from_artifact, filter_cohorts, get_cohort_preview, validate_cohorts

__all__ = [
    "ALLOWED_ACTIONS",
    "SEGMENT_CONSTRAINTS", 
    "choose_action",
    "choose_actions_batch",
    "make_tags",
    "extract_tags_batch",
    "build_features",
    "bin_scores",
    "fit_policy_tree",
    "save_model",
    "load_model",
    "generate_cohorts",
    "load_cohorts_from_artifact",
    "filter_cohorts",
    "get_cohort_preview", 
    "validate_cohorts"
]
