from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.distributions import (
    Distribution,
    Poisson,
    NegativeBinomial
)

from gluonts.torch.distributions import DistributionOutput

class PoissonOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"rate": 1}
    distr_cls: type = Poisson

    @classmethod
    def domain_map(cls, rate: torch.Tensor):  # type: ignore
        rate_pos = F.softplus(rate).clone()
        return (rate_pos.squeeze(-1),)

    # Overwrites the parent class method. We cannot scale using the affine
    # transformation since Poisson should return integers. Instead we scale
    # the parameters.
    def distribution(
        self,
        distr_args,
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> Distribution:
        (rate,) = distr_args

        if scale is not None:
            rate *= scale

        return Poisson(rate=rate, validate_args=False)

    @property
    def event_shape(self) -> Tuple:
        return ()

class NegativeBinomialOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"total_count": 1, "logits": 1}
    distr_cls: type = NegativeBinomial

    @classmethod
    def domain_map(cls, total_count: torch.Tensor, logits: torch.Tensor):  # type: ignore
        total_count = F.softplus(total_count)
        return total_count.squeeze(-1), logits.squeeze(-1)

    def _base_distribution(self, distr_args) -> Distribution:
        total_count, logits = distr_args
        return self.distr_cls(total_count=total_count, logits=logits)

    # Overwrites the parent class method. We cannot scale using the affine
    # transformation since negative binomial should return integers. Instead
    # we scale the parameters.
    def distribution(
        self,
        distr_args,
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> Distribution:
        total_count, logits = distr_args

        if scale is not None:
            logits += scale.log()

        return NegativeBinomial(total_count=total_count, logits=logits, validate_args=False)

    @property
    def event_shape(self) -> Tuple:
        return ()