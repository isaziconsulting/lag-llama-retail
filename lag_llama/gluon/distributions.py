from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.distributions import (
    Distribution,
    Poisson,
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

        # Disable argument validation so that we can allow evaluating loss for zeros (will break if rate is ever negative)
        return Poisson(rate=rate, validate_args=False)

    @property
    def event_shape(self) -> Tuple:
        return ()
