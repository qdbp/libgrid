from __future__ import annotations

from enum import Enum
from itertools import product
from typing import Iterable, TypeVar, Union

E = TypeVar("E", bound=Enum)
T = TypeVar("T")

Shape = tuple[int, ...]  # type: ignore
Ix = tuple[Union[slice, int], ...]


def fmt_ix(ix: Ix) -> str:
    out = "("
    for elem in ix:
        if isinstance(elem, int):
            out += f"{elem}"
        else:

            out += f"{elem.start or ''}:{elem.stop or ''}"
            if elem.step:
                out += f":{elem.step}"
        out += ","
    return out[:-1] + ")"


def iter_ndim(shape: tuple[int, ...]) -> Iterable[tuple[int, ...]]:
    """
    Iterate an arbitrary-dimensional shape in C-column order.
    """
    yield from product(*(range(d) for d in shape))
