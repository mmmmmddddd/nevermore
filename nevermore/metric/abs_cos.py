from typing import Any, Callable, List, Optional

import torch
from torch import Tensor
from torchmetrics.functional.regression.cosine_similarity import (
    _cosine_similarity_update
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat


class Abs_CosineSimilarity(Metric):
    preds: List[Tensor]
    target: List[Tensor]

    def __init__(
        self,
        reduction: str = "sum",
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        allowed_reduction = ("sum", "mean", "none", "abs", None)
        if reduction not in allowed_reduction:
            raise ValueError(
                f"Expected argument `reduction` to be one of "
                f"{allowed_reduction} but got {reduction}"
            )
        self.reduction = reduction

        self.add_state("preds", [], dist_reduce_fx="cat")
        self.add_state("target", [], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update metric states with predictions and targets.
        Args:
            preds: Predicted tensor with shape ``(N,d)``
            target: Ground truth tensor with shape ``(N,d)``
        """
        preds, target = _cosine_similarity_update(preds, target)

        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> Tensor:
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        return self.abs_cosine_similarity_compute(
            preds, target, self.reduction
        )

    def abs_cosine_similarity_compute(
        self, preds: Tensor, target: Tensor, reduction: str = "abs"
    ) -> Tensor:
        """Computes Cosine Similarity.
        torch.mean(1 - torch.abs(cosine similarity))
        Args:
            preds: Predicted tensor
            target: Ground truth tensor
            reduction:
                The method of reducing along the batch dimension using sum,
                mean or taking the individual scores
        """

        dot_product = (preds * target).sum(dim=-1)
        preds_norm = preds.norm(dim=-1)
        target_norm = target.norm(dim=-1)
        similarity = dot_product / (preds_norm * target_norm)
        reduction_mapping = {
            "abs": lambda x: (torch.mean(1 - torch.abs(x))),
            "sum": torch.sum,
            "mean": torch.mean,
            "none": lambda x: x,
            None: lambda x: x,
        }
        return reduction_mapping[reduction](similarity)
