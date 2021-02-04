from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic
import pulp
from pulp_lparray import lparray

from libgrid.palette import TP


class LocalGridConstraint(Generic[TP]):
    """
    Represents a required mutual relationship between local grid occupants.
    """

    def __init__(self, name: str):
        self.name = name

    # TODO half-baked API
    # I want to keep constraints as objects but there is currently no clear
    # understanding of the control flow leading to crossed concerns ¯\_(ツ)_/¯
    @abstractmethod
    def constrain(
        self,
        prob: pulp.LpProblem,
        lp_grid: lparray,
        **kwargs: Any,
    ) -> None:
        ...
