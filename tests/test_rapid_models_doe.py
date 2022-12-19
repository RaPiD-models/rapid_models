#!/usr/bin/env python

"""Tests for `rapid_models.doe` module."""

import pytest

import rapid_models.doe as doe

import numpy as np

# 1. Arrange
# 2. Act
# 3. Assert
# 4. Cleanup

# Tests related to rapid_models.doe.fullfact_with_bounds()


def test_fullfact_with_bounds_pos_dim():
    """
    Test ValueError raised for negative dimensions
    """
    with pytest.raises(ValueError):
        doe.fullfact_with_bounds([0.1, 0.1], [0.9, 0.9], [-1, 5])


def test_fullfact_with_bounds_zero_dim():
    """
    Test ValueError raised for zero dimension
    """
    with pytest.raises(ValueError):
        doe.fullfact_with_bounds([0.1, 0.1], [0.9, 0.9], [3, 0])


def test_fullfact_with_bounds_input_shape1():
    """
    Test ValueError raised for different first dimension of LB
    """
    with pytest.raises(ValueError):
        doe.fullfact_with_bounds([0.1], [0.9, 0.9], [3, 3])


def test_fullfact_with_bounds_input_shape2():
    """
    Test ValueError raised for different first dimension of UB
    """
    with pytest.raises(ValueError):
        doe.fullfact_with_bounds([0.1, 0.1], [0.9, 0.9, 1.2], [3, 3])


def test_fullfact_with_bounds_input_shape3():
    """
    Test ValueError raised for different first dimension of N_xi
    """
    with pytest.raises(ValueError):
        doe.fullfact_with_bounds([0.1, 0.1], [0.9, 0.9], [3, 1, 4])


def test_fullfact_with_bounds_input_shape4():
    """
    Test ValueError raised for input dimensions > 1
    """
    with pytest.raises(ValueError):
        doe.fullfact_with_bounds([[0.1, 0.9]], [[0.1, 0.9]], [[3, 1]])


def test_fullfact_with_bounds_output_shape1():
    """
    Test output shape 1
    """
    assert (3 * 5, 2) == doe.fullfact_with_bounds([0.1, 0.1], [0.9, 0.9], [3, 5]).shape


def test_fullfact_with_bounds_1():
    """
    Test output 1
    """
    # 1. + 2.
    X = np.asarray(doe.fullfact_with_bounds([0.1, 0.1], [0.9, 0.9], [3, 5]))
    X = X[X[:, 0].argsort()]
    # 3.
    assert np.allclose(
        X,
        np.array(
            [
                [0.1, 0.1],
                [0.1, 0.3],
                [0.1, 0.5],
                [0.1, 0.7],
                [0.1, 0.9],
                [0.5, 0.1],
                [0.5, 0.3],
                [0.5, 0.5],
                [0.5, 0.7],
                [0.5, 0.9],
                [0.9, 0.1],
                [0.9, 0.3],
                [0.9, 0.5],
                [0.9, 0.7],
                [0.9, 0.9],
            ]
        ),
    )


def test_fullfact_with_bounds_2():
    """
    Test output 2
    """
    # 1. + 2.
    X = np.asarray(doe.fullfact_with_bounds([-3, 4], [5, 12], [5, 3]))
    X = X[X[:, 0].argsort()]
    # 3.
    assert np.allclose(
        X,
        np.array(
            [
                [-3.0, 4.0],
                [-3.0, 8.0],
                [-3.0, 12.0],
                [-1.0, 4.0],
                [-1.0, 8.0],
                [-1.0, 12.0],
                [1.0, 4.0],
                [1.0, 8.0],
                [1.0, 12.0],
                [3.0, 4.0],
                [3.0, 8.0],
                [3.0, 12.0],
                [5.0, 4.0],
                [5.0, 8.0],
                [5.0, 12.0],
            ]
        ),
    )


def test_fullfact_with_bounds_3():
    """
    Test output 3
    """
    # 1. + 2.
    X = np.asarray(
        doe.fullfact_with_bounds([0.1, 0.1, 0.1], [0.9, 0.9, 1.2], [3, 3, 2])
    )
    # X = X[X[:, 0].argsort()]
    # 3.
    assert np.allclose(
        X,
        np.array(
            [
                [0.1, 0.1, 0.1],
                [0.5, 0.1, 0.1],
                [0.9, 0.1, 0.1],
                [0.1, 0.5, 0.1],
                [0.5, 0.5, 0.1],
                [0.9, 0.5, 0.1],
                [0.1, 0.9, 0.1],
                [0.5, 0.9, 0.1],
                [0.9, 0.9, 0.1],
                [0.1, 0.1, 1.2],
                [0.5, 0.1, 1.2],
                [0.9, 0.1, 1.2],
                [0.1, 0.5, 1.2],
                [0.5, 0.5, 1.2],
                [0.9, 0.5, 1.2],
                [0.1, 0.9, 1.2],
                [0.5, 0.9, 1.2],
                [0.9, 0.9, 1.2],
            ]
        ),
    )


# Tests related to rapid_models.doe.lhs_with_bounds()


def test_lhs_with_bounds_input_shape1():
    """
    Test ValueError raised for different first dimension of LB
    """
    with pytest.raises(ValueError):
        doe.lhs_with_bounds(2, 15, [0.1], [0.9, 0.9])


def test_lhs_with_bounds_input_shape2():
    """
    Test ValueError raised for different first dimension of UB
    """
    with pytest.raises(ValueError):
        doe.lhs_with_bounds(2, 15, [0.1, 0.1], [0.9, 0.9, 1.2])


def test_lhs_with_bounds_input_shape3():
    """
    Test ValueError raised for different nDim than dimension of LB and UB
    """
    with pytest.raises(ValueError):
        doe.lhs_with_bounds(3, 15, [0.1, 0.1], [0.9, 0.9])


def test_lhs_with_bounds_output_shape1():
    """
    Test output shape
    """
    assert (15, 2) == doe.lhs_with_bounds(2, 15, [0.1, 0.1], [0.9, 0.9]).shape


def test_lhs_with_bounds_1():
    """
    Test output 1
    """
    X = doe.lhs_with_bounds(2, 15, [0.1, 0.1], [0.9, 0.9], 42)
    assert np.allclose(
        X,
        np.array(
            [
                [0.87826211, 0.64077301],
                [0.66596549, 0.55465368],
                [0.48303066, 0.84914402],
                [0.76432373, 0.59553222],
                [0.60303707, 0.82075917],
                [0.36776451, 0.78187605],
                [0.70224771, 0.15070476],
                [0.11997547, 0.41839519],
                [0.26309779, 0.21498637],
                [0.19237301, 0.48311491],
                [0.8039826, 0.43132475],
                [0.54289292, 0.70620596],
                [0.3453928, 0.30619606],
                [0.46439694, 0.18526179],
                [0.21498766, 0.3510972],
            ]
        ),
    )


# TEST rapid_models.doe.in_hull()
@pytest.fixture
def lhs_2d_n20():
    return np.array(
        [
            [7.80377243, 0.53899863],
            [8.03252580, 0.23767858],
            [2.30055751, 1.49247746],
            [4.71597251, 3.51161260],
            [4.15212112, 4.52441803],
            [9.84211651, 4.86003812],
            [5.30592645, 2.53487347],
            [2.51029225, 4.23722138],
            [9.15230688, 1.17701814],
            [6.22803499, 3.19629399],
            [8.98281602, 2.13118911],
            [0.86599697, 1.55308478],
            [1.52904181, 1.79585113],
            [0.18727006, 3.79263103],
            [6.59983689, 0.96654404],
            [1.07800932, 3.37855861],
            [7.29620728, 2.32280729],
            [3.59091248, 0.39966462],
            [5.64607232, 2.84159046],
            [3.41622132, 4.45209934],
        ]
    )


@pytest.fixture
def lhs_2d_n50():
    return np.array(
        [
            [3.10737316, 0.09744831],
            [8.19559363, 0.50240114],
            [0.34574693, 0.42466494],
            [5.23193495, 2.75679661],
            [2.19088254, 2.39688269],
            [2.71060514, 4.65892837],
            [0.54152103, 1.1005238],
            [0.72911237, 0.89171935],
            [0.9412062, 3.73029007],
            [8.36705919, 3.10590245],
            [5.91239984, 4.10853758],
            [9.12838444, 4.06020399],
            [3.20212041, 1.68838007],
            [4.69437767, 2.23890787],
            [2.34945297, 0.92735479],
            [7.68858557, 2.13607103],
            [6.39082179, 3.9525245],
            [5.42074576, 3.39270905],
            [2.90252471, 4.54736305],
            [9.7714333, 4.84764979],
            [1.85418416, 2.65969115],
            [6.65049548, 1.01271146],
            [5.10553037, 4.32168835],
            [2.57135812, 2.45142547],
            [8.78341118, 3.85281328],
            [4.10937699, 3.57540539],
            [1.63709883, 0.73222079],
            [6.88895012, 0.34145996],
            [9.46271766, 3.27837582],
            [4.42803645, 1.95264039],
            [6.13238679, 1.82672104],
            [1.05119849, 2.02586703],
            [3.87237258, 1.7942256],
            [4.23492727, 1.26959562],
            [7.49629646, 2.96480717],
            [7.3548916, 4.77580186],
            [7.1787647, 4.45451722],
            [3.74898906, 1.39352358],
            [8.88012425, 4.96609144],
            [9.31271483, 2.8084722],
            [3.53128213, 0.13514678],
            [5.67726128, 2.53052425],
            [1.52817114, 3.68790485],
            [4.82911349, 1.52934598],
            [7.92162892, 0.27996046],
            [0.02269769, 1.41776121],
            [9.97262743, 4.2900313],
            [8.49765816, 3.43771513],
            [6.57604125, 0.63004364],
            [1.21974519, 3.01892428],
        ]
    )


@pytest.fixture
def pts_def_hull():
    return np.array(
        [
            [2.30055751, 1.49247746],
            [4.71597251, 3.5116126],
            [5.30592645, 2.53487347],
            [9.15230688, 1.17701814],
            [6.59983689, 0.96654404],
            [7.29620728, 2.32280729],
            [3.59091248, 0.39966462],
            [5.64607232, 2.84159046],
        ]
    )


@pytest.fixture
def pts_def_circ():
    return np.array(
        [
            [2.30055751, 1.49247746],
            [4.71597251, 3.5116126],
            [5.30592645, 2.53487347],
            [2.51029225, 4.23722138],
            [1.52904181, 1.79585113],
            [3.59091248, 0.39966462],
            [3.41622132, 4.45209934],
        ]
    )


def test_in_hull_shape2():
    """
    Test ValueError raised for different second dimension of points and hull
    (i.e. different number of dimensions). The size of the first dimension is
    the sample size.
    """
    with pytest.raises(ValueError):
        doe.in_hull(np.ones((2, 3)), np.ones((2, 2)))


def test_in_hull_non_array1():
    """
    Test ValueError raised for non-array input
    """
    with pytest.raises(ValueError):
        doe.in_hull(2, np.ones((2, 2)))


def test_in_hull_non_array2():
    """
    Test ValueError raised for non-array input
    """
    with pytest.raises(ValueError):
        doe.in_hull(np.ones((2, 3)), 15)


def test_in_hull_1(lhs_2d_n50, pts_def_hull):
    """
    Test output 1 of a polygon hull
    """
    # 2. Act
    b_inhull = doe.in_hull(lhs_2d_n50, pts_def_hull)
    # 3. Assert
    assert np.allclose(
        lhs_2d_n50[b_inhull],
        np.array(
            [
                [5.23193495, 2.75679661],
                [3.20212041, 1.68838007],
                [4.69437767, 2.23890787],
                [6.65049548, 1.01271146],
                [4.42803645, 1.95264039],
                [6.13238679, 1.82672104],
                [3.87237258, 1.7942256],
                [4.23492727, 1.26959562],
                [3.74898906, 1.39352358],
                [5.67726128, 2.53052425],
                [4.82911349, 1.52934598],
            ]
        ),
    )


def test_in_hull_2(lhs_2d_n50, pts_def_circ):
    """
    Test output 2 of a circle hull
    """
    # 2. Act
    b_inhull = doe.in_hull(lhs_2d_n50, pts_def_circ)
    # 3. Assert
    assert np.allclose(
        lhs_2d_n50[b_inhull],
        np.array(
            [
                [2.19088254, 2.39688269],
                [3.20212041, 1.68838007],
                [4.69437767, 2.23890787],
                [2.57135812, 2.45142547],
                [4.10937699, 3.57540539],
                [4.42803645, 1.95264039],
                [3.87237258, 1.7942256],
                [4.23492727, 1.26959562],
                [3.74898906, 1.39352358],
            ]
        ),
    )
