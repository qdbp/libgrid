from enum import auto
import numpy as np

from libgrid.palette import TilePalette
from libgrid.problem.flex import FlexSizeGridProblem


class TP(TilePalette):
    ZERO = auto()
    ONE = auto()


def test_sub_max_feasibility():
    """
    Checks a flex grid can find solutions strictly between the min and max
    sizes.
    """

    ubs = (5, 5)
    grid = FlexSizeGridProblem("submax_test", palette=TP, lbs=(1, 1), ubs=ubs)

    grid.impose_exclusive()

    prob = grid.prob

    (grid.extent_mask[0] == np.array([1, 1, 1, 0, 0])).constrain(
        prob, "fixedflags_h"
    )
    prob += grid.extent_mask[1][-3] == 0
    prob += -grid[...].sumit()
    grid.solve(msg=False)

    assert prob.objective.value() == -6

    expect = np.zeros(ubs)
    expect[:3, :2] = 1.0

    assert np.allclose(grid.value.sum(axis=-1), expect)

    assert np.allclose(grid.extent_mask[0].values, np.array([1, 1, 1, 0, 0]))
    assert np.allclose(grid.extent_mask[1].values, np.array([1, 1, 0, 0, 0]))


def test_sub_max_pbc():
    """
    Tests that periodic boundary conditions work with submax-bound grids.

    The idea is to check the model chooses a suboptimal value at the boundary,
    imposed in turn by a pin on the extent mask, caused by a pin at the zero
    line.
    """

    ubs = (4, 4)
    grid = FlexSizeGridProblem("submax_test", palette=TP, lbs=(1, 1), ubs=ubs)

    grid.impose_exclusive()
    grid.impose_pbc(0)
    grid.impose_pbc(1)

    grid.prob += grid[..., TP.ONE.value].sumit()

    grid.impose(
        "submax_h", grid.extent_mask[0] == np.array([1, 1, 1, 0])
    )
    grid.pin_tile((0, 3), TP.ZERO)

    print(grid.prob)

    grid.solve()
    print(grid.value)


if __name__ == "__main__":
    test_sub_max_pbc()
