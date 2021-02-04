from typing import Any, Generic, Literal

import pulp

from libgrid.constraint import LocalGridConstraint


# FIXME
class CrossGridTieConstriaint(LocalGridConstraint[GP], Generic[GP, GPx]):
    def __init__(self, name: str, cross_map: dict[GP, set[GPx]]):
        super().__init__(name)
        self.cross_map = cross_map

    def constrain(
        self,
        grid: GridProblem[GP],
        right_grid: GridProblem[GPx],
        mode: Literal["right_excludes", "right_implies", "same_as"],
        **kwargs: Any,
    ) -> None:

        assert (
            grid.prob is right_grid.prob
        ), "Passed grids have separate problems!"

        for left, rights in self.cross_map.items():

            left_elem = grid.lpa[..., left.value]
            right_expr = right_grid.lpa[
                ..., [right.value for right in rights]
            ].sum(axis=-1)

            if mode == "right_excludes":
                mode = pulp.LpConstraintLE
            elif mode == "right_implies":
                mode = pulp.LpConstraintGE
            else:
                mode = pulp.LpConstraintEQ

            grid.prob += pulp.LpConstraint(
                left_elem, rhs=right_expr, sense=mode, name=self.name
            )
