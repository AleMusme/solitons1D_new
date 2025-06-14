import numpy as np 
from typing import Callable

def create_profile(num_grid_points : int) -> np.array:
    """
    Creates a profile function for a grid with 
    `num_grid_points` points.

    Parameters
    ----------
    num_grid_points: int
        Number of grid points

    Returns
    -------
    profile: np.array
        Generated profile function 
    """
    profile = np.zeros(num_grid_points)
    return profile

class Soliton():

    def __init__(self, grid_spacing:float, num_grid_points:int):
        self.ls = grid_spacing
        self.lp = num_grid_points
        self.profile = create_profile(num_grid_points)

    def compute_energy(self):
        """
        Computes the total energy of the profile of the soliton.
        For now, takes the profile, sums it up and multiplies it by the grid spacing.
        """
        total_energy = np.sum(self.profile)
        total_energy *= self.ls
        self.energy = total_energy

class Grid:
    """
    A 1D grid.

    Parameters
    ----------
    grid_spacing : float
        Spacing between grid points.
    num_grid_points : int
        Number of grid points used in grid.
    grid_start, grid_end : int
        Alternative to num_grid_points, specifies start and end of grid.

    Attributes
    ----------
    grid_length : float
        Total length of grid.
    grid_points : np.array[float]
        An array of grid points, of the grid.

    """

    def __init__(
        self,
        grid_spacing: float,
        num_grid_points: int = None,
        grid_start: float = None,
        grid_end: float = None
    ):
        self.grid_spacing = grid_spacing

        if num_grid_points is None:

            assert (grid_start is not None and grid_end is not None), "You must provide either num_grid_points or both grid_start and grid_end"
            assert grid_start < grid_end, "grid_start should be less than grid_end"

            self.grid_length = end - start 
            self.num_grid_points = int(np.floor(self.grid_length / grid_spacing))
            self.grid_points = np.arange(
                grid_start, grid_end, grid_spacing  
            )
        
        else:

            self.num_grid_points = num_grid_points        
            self.grid_length = (num_grid_points) * grid_spacing
            self.grid_points = np.arange(
                -self.grid_length / 2, self.grid_length / 2, grid_spacing
            )