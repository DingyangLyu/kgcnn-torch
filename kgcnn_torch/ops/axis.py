"""Axis-related tensor operations for graph neural networks."""
from typing import List, Optional, Union


def broadcast_shapes(shape1: Union[tuple, list], shape2: Union[tuple, list]) -> list:
    """Broadcast two shapes to a unified shape.

    Follows NumPy broadcasting rules. Both inputs are converted to lists for
    mutability.

    Args:
        shape1: First shape as a tuple or list of integers (or None for dynamic).
        shape2: Second shape as a tuple or list of integers (or None for dynamic).

    Returns:
        list: The broadcasted output shape.

    Raises:
        ValueError: If shapes cannot be broadcast together.

    Example:
        >>> broadcast_shapes((5, 3), (1, 3))
        [5, 3]
        >>> broadcast_shapes((1, 4), (3, 1))
        [3, 4]
    """
    shape1 = list(shape1)
    shape2 = list(shape2)
    origin_shape1 = list(shape1)
    origin_shape2 = list(shape2)

    if len(shape1) > len(shape2):
        shape2 = [1] * (len(shape1) - len(shape2)) + shape2
    if len(shape1) < len(shape2):
        shape1 = [1] * (len(shape2) - len(shape1)) + shape1

    output_shape = list(shape1)
    for i in range(len(shape1)):
        if shape1[i] == 1:
            output_shape[i] = shape2[i]
        elif shape1[i] is None:
            output_shape[i] = None if shape2[i] == 1 else shape2[i]
        else:
            if shape2[i] == 1 or shape2[i] is None or shape2[i] == shape1[i]:
                output_shape[i] = shape1[i]
            else:
                raise ValueError(
                    "Cannot broadcast shape, the failure dim has value "
                    "%s, which cannot be broadcasted to %s. "
                    "Input shapes are: %s and %s."
                    % (shape1[i], shape2[i], origin_shape1, origin_shape2)
                )

    return output_shape


def get_positive_axis(axis: int, ndims: Optional[int],
                      axis_name: str = "axis",
                      ndims_name: str = "ndims") -> int:
    """Validate an axis parameter and normalize it to be positive.

    If ndims is known (not None), checks that axis is in the range
    ``-ndims <= axis < ndims`` and returns the positive equivalent.

    Args:
        axis (int): The axis value to validate.
        ndims (int or None): Number of dimensions.
        axis_name (str): Name of the axis parameter for error messages.
        ndims_name (str): Name of the ndims parameter for error messages.

    Returns:
        int: The normalized positive axis value.

    Raises:
        TypeError: If axis is not an integer.
        ValueError: If axis is out of bounds or negative with unknown ndims.
    """
    if not isinstance(axis, int):
        raise TypeError(
            "%s must be an int; got %s" % (axis_name, type(axis).__name__)
        )
    if ndims is not None:
        if 0 <= axis < ndims:
            return axis
        elif -ndims <= axis < 0:
            return axis + ndims
        else:
            raise ValueError(
                "%s=%s out of bounds: expected %s<=%s<%s"
                % (axis_name, axis, -ndims, axis_name, ndims)
            )
    elif axis < 0:
        raise ValueError(
            "%s may only be negative if %s is statically known."
            % (axis_name, ndims_name)
        )
    return axis
