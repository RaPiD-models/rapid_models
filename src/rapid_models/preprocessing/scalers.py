"""Convenience scaling functions for rapid-models.
  For more scaling functions refer to e.g.
  https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing"""

# @TODO: The 4 functions defined in this module are nowhere called.
#        If we keep them, we should add type hints and some example code or unit tests.
#        @AGRE / @ELD: Delete?
#        CLAROS, 2022-11-01


def scale_x_to_box(x, bounds):
    """
    Input x = points in [0, 1]^n
    output scaled to lie in the box given by bounds
    """
    x_tmp = x.copy()
    for i in range(x.shape[1]):
        x_tmp[:, i] = x_tmp[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]

    return x_tmp


def scale_x_to_box_inv(x, bounds):
    """
    Inverse of scale_x_to_box
    """
    x_tmp = x.copy()
    for i in range(x.shape[1]):
        x_tmp[:, i] = (x_tmp[:, i] - bounds[i][0]) / (bounds[i][1] - bounds[i][0])
    return x_tmp


def standardScaler(
    x, mean=None, std=None, dim=0, tensorType="torch", bReturnParam=False
):
    """
    Standardize features by removing the mean and scaling to unit variance.
    The standard score of a sample $x$ is calculated as:

    $$ x_{out} = \frac{x - x.mean()}{x.std()} $$

    Args:
      x (array-like, ND): Array of features to be scaled.
      mean (float, default=None): Specify the mean that will be subtracted from
        the features in $x$. If _None_, the mean will be calculated from the
        features as mean=x.mean().
      std (float, default=None): Specify the std that the features in $x$ will
        be scaled by. If _None_, the std will be calculated from the features
        as std=x.std() (unbiased).
      dim (int or tuple of python:ints, defalt=0): The dimension or dimensions
        to reduce to establish the mean and std if these are _None_.
      tensorType (str, default=torch): Specify if torch. or numpy. functions
        are used.
      bReturnParam (bool, default=False): Specify if the function should return
        the mean and std used in the scaling. Should be _True_ if mean and std
        is _None_ to retain the parameters.

    Returns:
      x_out (array-like, ND): Scaled features.

        - if bReturnParam=True the function return a tuple with (x_out, mean, std)
    """

    if mean is None and std is None:
        if tensorType.lower() == "torch":
            mean = x.mean(dim, keepdim=True)
            std = x.std(dim, unbiased=False, keepdim=True)
        else:
            mean = x.mean(dim, keepdims=True)
            std = x.std(dim, ddof=0, keepdims=True)
    x_out = x - mean
    x_out = x_out / std
    if bReturnParam:
        return x_out, mean, std
    else:
        return x_out


def standardReScaler(x, mean, std):
    """
    Rescale features based on specified mean and std.

    $$ x_{out} = x*x.std() + x.mean()

    Args:
      x (array-like, ND): Array of features to be rescaled.
      mean (float, default=None): Specify the mean that will be added to the
        features in $x$ after rescaling.
      std (float, default=None): Specify the std that the features in $x$ will
        be rescaled by.

    Returns:
      x_out (array-like, ND): Re-scaled features.
    """
    x_out = x * std
    x_out = x_out + mean
    return x_out
