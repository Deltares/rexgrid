"""
1. structured -> structured
2. structured -> unstructured
3. unstructured -> structured
4. unstructured -> unstructured

Note: i -> j can be inverted to j -> v.

1a. Overlap: easy, compute and broadcast-multiply.
1b. Interpolation: easy, bilinear between four neighbors.

2a. Overlap: search unstructured bounding boxes on structured. More efficient than
conversion to unstructured, building celltree, then searching.
2b. Interpolation: easy, bilinear between four neighbors.

3a. Overlap: inverse of 2a.
3b. Interpolation: requires natural neighbor, IDW.

4a. Overlap: search celltree. Compute overlap.
4b. Interpolation: requires natural neighbor, IDW.

Conclusion: 2a, 3a could be optimized by using a grid search than celltree search.
Others require different classes?
Not all logic differs. Probably better to create GridWrapper classes which implement
weighting procedures?

*   *
  *

Xugrid explicitly targets spatial data.

While arbitrary dimensionality could be supported with sufficient (obtuse?)
abstraction, two to three spatial dimension is much easier to comprehend.

Wrapper or Adapter?

Based on dimension to regrid, decide on a regridder.
"""


class BaseGridWrapper(abc.ABC):
    """
    Abstract base class for GridWrappers.
    """

    def length(self, dimension: str):
        return np.diff(self.coord_bounds[dimension], axis=1)


class BaseGridWrapper2d(BaseGridWrapper):
    @abc.abstractmethod
    def area(self):
        """
        Compute area
        """


class BaseGridWrapper3d(BaseGridWrapper):
    @abc.abstractmethod
    def area(self):
        """
        Compute area
        """

    @abc.abstractmethod
    def volume(self):
        """
        Compute volume
        """
