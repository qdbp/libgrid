from __future__ import annotations

from enum import auto
from typing import Type, Union

import numpy as np
import pulp
from pulp_lparray import lparray

from libgrid import Ix, iter_ndim
from libgrid.palette import TP, TilePalette
from libgrid.problem import GridProblem


class FlexSizeGridProblem(GridProblem[TP]):
    """
    An arbitrary-dimensional grid problem with variable bounds.

    The solution space is free to range independently in each dimension
    from some lower bound to some upper bound, forming a contiguous
    hyperrectangular domain.

    The solution includes a set of optimal bounds ob for each dimension d.

    Supports exclusion and periodic boundary imposition in a compatible manner:
        - exclusion applies only to the valid region; the grid is 0 outside.
        - periodicity is defined such that for each dimension d:
            forall d. grid[..., 0(d), ...] == grid[..., ob(d), ...]
    """

    def __init__(
        self,
        name: str,
        palette: Type[TP],
        lbs: tuple[int, ...],
        ubs: tuple[int, ...],
        **kwargs,
    ) -> None:

        assert len(lbs) == len(ubs)

        super().__init__(name=name, palette=palette, geom_shape=ubs, **kwargs)

        # for each dim, equals [1, 1, 1, ..., 1, 0, 0, ..., 0]
        # the outer product of these marks which portion of the grid is valid
        self.extent_mask = {}

        # for each dim, equals [0, 0, 0, ..., 1, 0, 0, ..., 0]
        # it is equal to one exactly where the last valid entry of that dim is
        self.dim_edge_flags = {}

        for dx in range(self.ndim):
            lb = lbs[dx]
            ub = ubs[dx]

            assert 1 <= lb <= ub

            df = self.extent_mask[dx] = lparray.create_anon(
                f"{name}_DimFlag({dx})", (ub,), cat=pulp.LpBinary
            )

            (df[:lb] == 1.0).constrain(self.prob, f"{name}_DimLb({dx})")
            (df[:-1] >= df[1:]).constrain(
                self.prob, f"{name}_DimFlagMonotone({dx}"
            )
            dfe = self.dim_edge_flags[dx] = lparray.create_anon(
                f"{name}_DimEdgeFlag({dx})", (ub,), cat=pulp.LpBinary
            )

            (dfe[:-1] == df[:-1] - df[1:]).constrain(
                self.prob, f"{name}_DefLeDf({dx})"
            )
            (dfe[-1:] == df[-1:]).constrain(self.prob, f"{name}_DefEdge({dx})")

        # variables for assigning loss
        self.used_size = [
            self.dim_edge_flags[dx] @ np.arange(ubs[dx])
            for dx in range(self.ndim)
        ]

    def impose_exclusive(self) -> None:

        occupancy_mask = self.lpa.sum(axis=-1)
        (occupancy_mask <= 1.0).constrain(self.prob, f"{self.name}_ExclUb")

        # TODO right now lparray has super super slow broadcasting thanks to
        # how sum is implemented... for now this is better.
        for ix in iter_ndim(self.geom_shape):
            dim_align = 0

            for dx in range(self.ndim):
                incl_along_dim = self.extent_mask[dx][ix[dx]]
                dim_align += incl_along_dim
                self.prob += occupancy_mask[ix] <= incl_along_dim

            self.prob += pulp.LpConstraint(
                name=f"f{self.name}_ExclLb{ix}",
                e=occupancy_mask[ix] + self.ndim - dim_align,
                sense=pulp.LpConstraintGE,
                rhs=1,
            )

    def impose_pbc(self, axis: int) -> None:
        # We want grid[..., 0, ...] == grid[..., argmax(edge_flag(axis)), ...]

        def mk_plane_indexer(plane_ix: int) -> Ix:
            return tuple(
                slice(None, None, None) if dx != axis else plane_ix
                for dx in range(self.ndim)
            )

        zero_row = self.lpa[mk_plane_indexer(0)]

        for ix_along_axis in range(1, self.geom_shape[axis]):
            cname = f"{self.name}_PBC_({axis};{ix_along_axis})"
            (
                zero_row
                >= self.lpa[mk_plane_indexer(ix_along_axis)]
                + self.dim_edge_flags[axis][ix_along_axis]
                - 1
            ).constrain(self.prob, f"{cname}_lb")

            (
                zero_row
                <= self.lpa[mk_plane_indexer(ix_along_axis)]
                - self.dim_edge_flags[axis][ix_along_axis]
                + 1
            ).constrain(self.prob, f"{cname}_ub")

    @property
    def valid_values(self) -> np.ndarray:
        if not self.is_solved:
            raise ValueError("Grid is not solved.")

        valid_dim_extents = tuple(
            int(self.extent_mask[dx].values.sum()) for dx in range(self.ndim)
        )

        return self.gix.each_dim(lambda dx: slice(0, valid_dim_extents[dx]))()

    def print_solution(self) -> None:
        flat_values: np.ndarray = self.valid_values.argmax(axis=-1)

        if self.ndim == 2:
            for row in flat_values:
                for col in row:
                    print(self.palette(col).draw(), end="")
                print("")
        else:
            raise NotImplementedError("Cannot print non-2D solutions.")


def test_sub_max_feasibility():
    class TP(TilePalette):
        ZERO = auto()
        ONE = auto()

    grid = FlexSizeGridProblem(
        "submax_test", palette=TP, lbs=(1, 1), ubs=(5, 5)
    )

    grid.impose_exclusive()

    prob = grid.prob

    (grid.extent_mask[0] == np.array([1, 1, 1, 0, 0])).constrain(
        prob, "fixedflags_h"
    )
    prob += grid.extent_mask[1][-3] == 0

    prob += -grid[...].sumit()

    grid.solve()
    print(grid.extent_mask[0].values)
    print(grid.extent_mask[1].values)
    print(grid.valid_values)
    print(grid.lpa.values.sum(axis=-1))


if __name__ == "__main__":
    test_submax_feasibility()
