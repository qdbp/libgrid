from __future__ import annotations

from typing import Any, Callable, Generic, Iterable, Optional, Type, Union
import numpy as np

import pulp
from pulp import LpConstraint
from pulp_lparray import lparray

from libgrid import Ix, Shape, fmt_ix
from libgrid.constraint.conv import (
    ConvConstraint,
    ConvTemplate,
    Key,
)
from libgrid.palette import TP, TPx


class GIx(Generic[TP]):
    """
    Grid indexer.
    """

    def __init__(self, grid: GridProblem[TP]):
        self.grid = grid
        self.indexers: list[Union[int, slice, list[int]]] = [
            slice(None, None, None) for _ in range(grid.ndim + 1)
        ]

    @property
    def ndim(self) -> int:
        return self.grid.ndim

    # noinspection PyMethodParameters
    def dim(gix, dim: int):
        if not (0 <= dim <= gix.ndim):
            raise ValueError(
                f"This indexer indexes geometric dimensions [0, {gix.ndim}]"
            )

        class DimIndexer:
            def __getitem__(self, item: Union[int, slice]) -> GIx:
                gix.indexers[dim] = item
                return gix

        return DimIndexer()

    def each_dim(self, op: Callable[[int], Union[int, slice, list[int]]]):
        """
        Applies a dynamic slicer to each dimension in order.
        """
        for dx in range(self.ndim):
            self.indexers[dx] = op(dx)
        return self

    # noinspection PyMethodParameters
    def ix(gix):
        class IxIndexer:
            def __getitem__(self, item: Ix):
                if len(item) != gix.ndim:
                    raise ValueError(
                        f"Indexer ndim mismatch {len(item) != gix.ndim}"
                    )
                for tx, ixer in enumerate(item):
                    gix.indexers[tx] = ixer
                return gix

        return IxIndexer()

    def tiles(self, *tiles: TP) -> GIx:
        self.indexers[-1] = [t.value for t in tiles]
        return self

    def __call__(self) -> lparray:
        return self.grid[self.__index__()]

    def __index__(self) -> Ix:
        return tuple(self.indexers)

    # TODO consider inlining fmt_ix
    def __str__(self) -> str:
        return fmt_ix(self.__index__())


class GridProblem(Generic[TP]):
    """
    Represents a problem that can be stated as follows:

    Given an N-dimensional grid of tiles, find some assignment of distinct tiles
    from a finite set T into each cell of this grid that obeys some constraints
    and maximized some objective, where these constraints can be formulated as
    a mixed integer linear program.
    """

    def __init__(
        self,
        name: str,
        palette: Type[TP],
        geom_shape: Shape,
        prob: pulp.LpProblem = None,
    ):
        """
        Args:
            name: name of the grid. Used to disambiguate Lp expressions in pulp.
            palette: the TilePalette class object from which possible tile
                values are drawn
            geom_shape: the shape of the tile space. This excludes the last
                dimension of the underlying array, which represents tile
                identity.
            prob: the LpProblem associated with this grid. All constraints will
                be added to this problem. If None, a new problem will be
                created.

        """

        self.name = name
        self.palette = palette
        self.geom_shape = geom_shape

        if prob is None:
            self.prob = pulp.LpProblem(name=f"{self.name}Problem")
        else:
            self.prob = prob

        index_sets = tuple(
            [tuple(range(geom_shape[d])) for d in range(len(geom_shape))]
            + [[p.name for p in self.palette]]
        )
        self.lpa = lparray.create(
            f"{name}Grid",
            index_sets=index_sets,
            cat=pulp.LpBinary,
        )

        self.exclusive = False
        self.pbc = False

        self._status: Optional[int] = None

    @property
    def gix(self) -> GIx[TP]:
        """
        Create a new GIx indexer, initialized to select all data.
        """
        return GIx(self)

    @property
    def ndim(self) -> int:
        return len(self.geom_shape)

    @property
    def is_solved(self) -> bool:
        return self._status == pulp.LpStatusOptimal

    @property
    def status(self) -> str:
        if self._status is None:
            return "Unsolved"
        return pulp.LpStatus[self._status]

    @property
    def value(self) -> np.ndarray:
        if not self.is_solved:
            raise ValueError("Problem is not solved.")
        return self.lpa.values

    # TODO singledispatchmethod is great for this... except it bugs out
    # https://bugs.python.org/issue39679
    def impose(
        self,
        name: str,
        constraint: Union[LpConstraint, lparray[LpConstraint]],
    ):
        """
        Imposes an arbitrary constraint on the problem.
        """
        if isinstance(constraint, lparray):
            constraint.constrain(prob=self.prob, name=name)
        elif isinstance(constraint, LpConstraint):
            constraint.name = name
            self.prob += constraint
        else:
            raise TypeError(f"Can't constrain with {type(constraint)}")

    def pin_tile(self, ix: Ix, tile: TP) -> None:
        self.gix.dim(0)[0].tiles(tile)

        self.impose(
            f"{self.name}::Pin:{fmt_ix(ix)}={tile.name}:",
            self.lpa[ix, tile.value] == 1.0,
        )

    # TODO add support for exclusion sets.
    def impose_exclusive(self) -> None:
        """
        Imposes that tiles are mutually exclusive.
        """
        (self.lpa.sum(axis=-1) == 1).constrain(self.prob, f"{self.name}_Excl")
        self.exclusive = True

    def impose_pbc(self, axis: int) -> None:
        """
        Imposes periodic boundary conclusions along an axis.

        Args:
            axis: the axis to impose conditions along
        """

        start = self.gix.dim(axis)[0]()
        end = self.gix.dim(axis)[-1]()
        (start == end).constrain(self.prob, f"{self.name}_PBC[{axis}]")

        self.pbc = True

    def initialize(
        self, vals: np.ndarray, indexer: Optional[tuple[slice, ...]] = None
    ) -> None:
        """
        Sets the initial values of the underlying grid variables to some values.

        Args:
            vals: the values to initialize the grid to.
            indexer: if not None, the subset of the grid defined by
                grid[indexer] will be initialized instead of the whole grid.
        """
        if self.is_solved:
            raise ValueError("Grid is already solved.")

        assert vals.shape == self.lpa.shape

        def init_one(a: pulp.LpVariable, b: float) -> None:
            a.setInitialValue(b)

        vec_init = np.vectorize(init_one)

        if indexer is None:
            vec_init(self.lpa, vals)
        else:
            vec_init(self.lpa[indexer], vals)

    # TODO CBC is hardcoded
    def solve(self, **kwargs: Any) -> None:
        self.prob.solve(pulp.PULP_CBC_CMD(**kwargs))
        self._status = self.prob.status
        if self.status != pulp.LpStatus[pulp.LpStatusOptimal]:
            print(f"BAD STATUS {self.status}")

    def to_tiles(self) -> np.ndarray:
        """
        Converts the underlying lparray to a numpy array of tile objects.

        Returns:
            ndarray of shape self.geom_shape of tile objects from the palette.
        """

        def to_tile(x: int) -> TP:
            return self.palette(x)

        vec_to_tile: Callable[[np.ndarray], np.ndarray] = np.vectorize(to_tile)
        return vec_to_tile(self.lpa.values.argmax(axis=-1))

    def print_solution(self) -> None:
        flat = self.lpa.values.argmax(axis=-1)
        if self.ndim == 2:
            for row in flat:
                for col in row:
                    print(self.palette(col).draw(), end="")
                print("")

    def make_aux(
        self, aux_name: str, aux_palette: Type[TPx]
    ) -> GridProblem[TPx]:
        """
        Creates an auxiliary grid.

        The aux grid has the same shape and shares the LpProblem, but can
        have a different tile palette.

        Args:
            aux_palette: palette to use for the aux grid
            aux_name:

        Returns:

        """
        return GridProblem(
            name=aux_name,
            palette=aux_palette,
            geom_shape=self.geom_shape,
            prob=self.prob,
        )

    def constrain_conv(
        self,
        name: str,
        template_shape: Shape,
        template_spec: dict[tuple[int, ...], Union[TP, Key[TP], list[TP]]],
        *,
        min_conv: int = None,
        reflect_axis: int = None,
        boundary_tiles: Iterable[TP] = (),
    ) -> None:

        boundary_tiles = list(boundary_tiles)

        template = ConvTemplate(
            self.palette,
            shape=template_shape,
            spec=template_spec,
            min_conv=min_conv,
        )

        ConvConstraint(name, boundary_tiles=boundary_tiles).constrain(
            self.prob, self.lpa, template
        )

        if reflect_axis is not None:
            refl_name = f"{name}_r{reflect_axis}"
            ConvConstraint(
                refl_name,
                boundary_tiles=[
                    t.reflect(reflect_axis) for t in boundary_tiles
                ],
            ).constrain(self.prob, self.lpa, template.reflect(reflect_axis))

    def __getitem__(self, item: Any) -> Any:
        return self.lpa[item]
