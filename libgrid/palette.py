from __future__ import annotations
from enum import Enum, Flag, auto
from typing import Type, TypeVar, Union

TP = TypeVar("GP", bound="GridPalette")
TPx = TypeVar("GPx", bound="GridPalette")
T = TypeVar("T")


class TileMeta(Flag):
    NO_REFLECT = auto()


# TODO python enums are incredibly hacky and clunky. I hate them.
# should find an enum-like solution that supports richer objects as values
class TilePalette(Enum):
    """
    Represents the possible tile values a grid can have.
    """

    # noinspection PyMethodParameters
    def _generate_next_value_(name, start, count, last_values):
        return count

    @classmethod
    def verify(cls: Type[TP]) -> None:
        elem_order: list[TP] = list(cls)
        for elem in cls:
            if elem.value != (ix := elem_order.index(elem)):
                raise ValueError(
                    f"Element {elem} has value {elem.value} but index {ix}"
                )

    @classmethod
    def n_cats(cls) -> int:
        return len(cls)

    @classmethod
    def register_reflection(
        cls: Type[TP],
        axis: int,
        pos: TP,
        neg: Union[TP, TileMeta],
    ) -> None:
        """
        Defines how to handle a tile object on reflection.

        A reflection along an axis is defined as reordering the elements along
        that axis.

        For common array layouts, a reflection along axis 0 is a vertical
        reflection, and along 1 it is a horizontal one. For 3D, it depends
        on the chosen convention -- take care!
        """

        reflect_map: dict[int, dict[TP, Union[TP, TileMeta]]]

        if (reflect_map := getattr(cls, "__reflect_map", None)) is None:
            reflect_map = {}
            setattr(cls, "__reflect_map", reflect_map)

        if axis not in reflect_map:
            reflect_map[axis] = {}

        reflect_map[axis][pos] = neg
        if not isinstance(neg, TileMeta):
            reflect_map[axis][neg] = pos

    def reflect(self: T, axis: int) -> T:
        cls = self.__class__
        if (
            (rmap := getattr(cls, "__reflect_map", None))
            and (axmap := rmap.get(axis))
            and self in axmap
        ):
            out: Union[T, TileMeta] = axmap[self]
            if isinstance(out, TileMeta):
                if out | TileMeta.NO_REFLECT:
                    raise ValueError(
                        f"Trying to reflect a NoReflect tile {self}."
                    )
                else:
                    raise NotImplementedError()
            return out
        return self

    # noinspection PyMethodMayBeStatic
    def draw(self) -> str:
        """
        Returns a canonical character representation of the tile.

        Used to represent solutions graphically.
        """
        return "?"
