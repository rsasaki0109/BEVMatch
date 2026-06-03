"""Evaluation framework (§13): retrieval/alignment metrics and recipes."""

from bevmatch.eval.alignment_eval import (
    AlignmentReport,
    evaluate_alignment,
    pose_errors,
)
from bevmatch.eval.change_eval import ChangePRF, change_prf, false_changes
from bevmatch.eval.metrics import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)
from bevmatch.eval.retrieval_eval import RetrievalReport, evaluate_retrieval

__all__ = [
    "recall_at_k",
    "precision_at_k",
    "mean_reciprocal_rank",
    "evaluate_retrieval",
    "RetrievalReport",
    "evaluate_alignment",
    "AlignmentReport",
    "pose_errors",
    "change_prf",
    "ChangePRF",
    "false_changes",
]
