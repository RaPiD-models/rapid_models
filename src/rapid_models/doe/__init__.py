"""Basic DOE package for rapid-models based on pyDOE2."""

from .utils import fullfact_with_bounds, lhs_with_bounds, in_hull, kmeans_sample

__all__ = ["fullfact_with_bounds", "lhs_with_bounds", "in_hull", "kmeans_sample"]
