"""
Constraints that can be represented as inequalities between a convolution of
the grid with some fixed template and some local value.
"""
from __future__ import annotations
from typing import Any, Generic, Iterable, Type, Union

import numpy as np
import pulp
from pulp_lparray import lparray

from libgrid import Shape, T, iter_ndim
from libgrid.palette import TP
from libgrid.constraint import LocalGridConstraint
from libgrid.problem import GridProblem


class Key(Generic[T]):
    """
    Wrapper around a value marking it as the defining tile of a convolution.
    """

    def __init__(self, val: T):
        self.val = val

    def __str__(self) -> str:
        return f"Key[{self.val}]"

    __repr__ = __str__


class ConvTemplate(Generic[TP]):
    def __init__(
        self,
        palette: Type[TP],
        shape: Shape,
        spec: dict[tuple[int, ...], Union[TP, Key[TP], list[TP]]],
        *,
        min_conv: int = None,
    ):

        palette.verify()
        self.palette = palette

        self.geom_shape = shape
        self.keys: dict[Shape, int] = {}
        self.arr = np.zeros(shape + (len(palette),), dtype=np.uint8)

        self.spec = spec

        self.key_ix: Shape
        self.key_idx: int

        for ix, item in spec.items():

            elems: list[TP]

            if isinstance(item, list):
                elems = item
            elif isinstance(item, Key):
                self.key_ix = ix
                self.key_idx = item.val.value
                elems = [item.val]
            else:
                # noinspection PyTypeChecker
                elems = [item]

            for c in elems:
                self.arr[ix + (c.value,)] = 1.0

        if min_conv is None:
            self.min_conv = int(self.arr.max(axis=-1).sum())
        else:
            self.min_conv = min_conv

        # assert self.min_conv <= len(spec.keys())
        assert self.key_idx is not None
        assert self.key_ix is not None

    @property
    def shape(self) -> Shape:
        return self.arr.shape  # type: ignore

    def calc_segment_excess(
        self, grid_segment: lparray
    ) -> pulp.LpAffineExpression:
        """
        Returns the difference between the obtained and expected convolution
        value for a single, matched-shape grid segment. Can be used as a
        condition directly, or combined more fancily.
        """

        assert grid_segment.shape == self.arr.shape

        return (
            (self.arr * grid_segment).sum()
            - self.min_conv * grid_segment[self.key_ix + (self.key_idx,)]
        ).item()

    def reflect(self, axis: int) -> ConvTemplate[TP]:

        reflected_dict = {
            tuple(
                (self.geom_shape[ix] - key[ix] - 1 if ix == axis else key[ix])
                for ix in range(len(self.geom_shape))
            ): (
                Key(spec_val.val.reflect(axis))
                if isinstance(spec_val, Key)
                else (
                    spec_val.reflect(axis)
                    if not isinstance(spec_val, list)
                    else [v.reflect(axis) for v in spec_val]
                )
            )
            for key, spec_val in self.spec.items()
        }

        return ConvTemplate(
            palette=self.palette,
            shape=self.geom_shape,
            spec=reflected_dict,
            min_conv=self.min_conv,
        )


class ConvConstraint:
    """
    Represents a local grid constraint that can be expressed as a convolution:

        G ⊛ T >= k * T[jx..., jdx]

    Where:
        G is the binary occupancy map
        T is some template with:
            T.ndim == G.ndim
            T.shape[-1] == G.shape[-1]
            T.shape[:-1] < G.shape[:-1]
        jx is the relative position, in the template, of the keynode
        jdx is the occupant-identity of the keynode, defined by the grid

    In human terms, this is to test for the presence of an occupant type X,
    and for every such occupant make sure that its surroundings, defined by T
    match what is found in G to tolerance k.

    For example, suppose we are laying out a 2D garden and declare that every
    pinwheel should be surrounded on adjacent sides by gnomes. Suppose the
    occupant index of a pinwheel is 5, and that of gnomes is 7. Suppose there
    are 10 total possible garden blocks.

    Then our template would have shape (3, 3, 10), and would be zero everywhere
    except for:
        T[0, 1, 7] = T[1, 0, 7] = T[1, 2, 7] = T[2, 1, 7]

    which defines the pattern of gnomes we are looking for. Since we want
    all the gnomes, we have k = 4, jx = (1, 1) and jdx = 5.
    This would result in the following inequality being defined at each valid
    spot (i, j) in the garden.

    G[i,j+1,7] + G[i+1,j,7] + G[i+1,j+2,7] + G[i+2,j+1,7] >= 4 * G[i+1,j+1,5]

    Which can be read as simply: if there is a pinwheel, we want the sum of
    (this tile is a gnome) on the adjacent sides to equal 4.
    """

    # noinspection PyPep8Naming
    # FIXME API should have GridProblem -- to use full API such as indexing,
    # etc
    def constrain(
        self,
        template: ConvTemplate[TP],
        gp: GridProblem,
        name: str,
        *,
        boundary_tiles: Iterable[TP] = (),
        **kwargs: Any,
    ) -> None:
        """
        Perform the convolution constraint.

        The grid is expanded in each dimension by the shape of the template
        in that dimension less one; that is, the convolution shape is full.
        However, only overlaps where the template's key overlaps the true
        grid are considered.

        By default, the expanded grid is zero padded: this assumes a strict
        requirement condition. This can be relaxed by adding elements to
        `boundary_tiles` -- the value of the tile plane for each of these
        will be set to 1 in the border. They will not be mutually exclusive.

        For the pinwheel example: by default, it could not go on the edge of the
        garden, because "off the edge" is assumed to have no tile. We could
        pass "gnome" as a boundary tile, in which case we could place pinwheels!

        Args:
            lp_grid: the lparray grid to apply the constraint template to.
                Assumes it is a binary array with the last dimension summing to
                1 for all coordinates.

        Returns:

        """

        N = gp.ndim
        assert N == len(template.geom_shape)

        expanded_lpa = np.zeros(
            tuple(
                gp.geom_shape[d] + 2 * (template.geom_shape[d] - 1)
                for d in range(N)
            )
            + (len(gp.palette),),
            dtype=object,
        ).view(lparray)

        for tile in boundary_tiles:
            expanded_lpa[..., tile.value] = 1.0

        insert_gix = gp.gix.each_dim(
            lambda d: slice(
                template.geom_shape[d], -template.geom_shape[d] + 1 or None
            )
        )
        expanded_lpa[insert_gix] = gp.lpa

        for coords in iter_ndim(
            # this is the coordinate into the expanded grid of the farthest
            # real (i.e. non-padding) tile
            tuple(
                gp.geom_shape[d] + template.geom_shape[d] - 2 for d in range(N)
            )
        ):

            # only calculate the convolution if the key tile is in the real grid
            real_key_ix = (
                coords + template.key_ix - np.array(template.geom_shape) + 1
            )
            if not any(real_key_ix < 0) and not any(
                real_key_ix >= np.array(gp.geom_shape)
            ):

                segment = expanded_lpa[
                    gp.gix.each_dim(
                        lambda d: slice(
                            coords[d], int(coords[d] + template.shape[d])
                        )
                    )
                ]

                excess = template.calc_segment_excess(segment)
                const_name = f"{name}@key={tuple(map(int,real_key_ix))}"
                gp.impose(const_name, excess >= 0)
