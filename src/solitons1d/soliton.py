import numpy as np 
from typing import Callable

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

class Lagrangian:
    """
    Used to represent Lagrangians of the form:
        L = - 1/2(dx_phi)^2 - V(phi)

    Parameters
    ----------
    V : function
        The potential energy function, must be a map from R -> R
    dV : function
        The derivative of the potential energy function, must be a map from R -> R
    vacua : list-like or None
        List of vacua of the potential energy.
    """ 

    def __init__(
        self,
        V: Callable[[float], float], #this is how to pass functions as arguments in python
        dV: Callable[[float], float],
        vacua: list | np.ndarray | None = None,  # np.ndarray is the type of a numpy array
    ):
        self.V = V
        self.dV = dV
        self.vacua = vacua 

        if vacua is not None:
            for vacuum in vacua:
                assert np.isclose(dV(vacuum), 0), (
                    f"The given vacua do not satisfy dV({vacuum}) = 0"
                )

class Soliton():

    """
    A class describing a Soliton.

    Parameters
    ----------
    grid : Grid
        The grid underpinning the soliton.
    lagrangian : Lagrangian
        The Lagrangian of the theory supporting the soliton.
    initial_profile_function : None | function
        The initial profile function, must be from R -> R. Optional.
    initial_profile : None | array-like
        The initial profile function as an array. Optional.
    """

    def __init__(self, 
                 grid : Grid,
                 lagrangian : Lagrangian,
                 initial_profile_function: Callable[[float], float] | None = None,      #why this none=none?
                 initial_profile: np.ndarray | None = None
    ):
        self.grid = grid 
        self.lagrangian = lagrangian
        self.profile = np.zeros(grid.num_grid_points)

        assert (initial_profile_function is None) or (initial_profile is None), (
            "Please only specify `initial_profile_function` or `profile_function`"
        )

        if initial_profile_function is not None:
            self.profile = create_profile(self.grid.grid_points, initial_profile_function)
        else:
            self.profile = initial_profile
        
        self.compute_energy = self.compute_energy()

    def compute_energy(self):
        """
        Computes the total energy of the profile of the soliton, and stores it in `Soliton.energy`
        """
        energy = compute_energy_fast(
            self.lagrangian.V,
            self.profile,
            self.grid.num_grid_points,
            self.grid.grid_spacing,
        )
        self.energy = energy

def compute_energy_fast(V, profile, num_grid_points, grid_spacing):

    total_energy = 0
    return total_energy

def create_profile(
        grid_points : np.array,
        initial_profile_function : Callable[[float], float] | None = None
) -> np.array:
    """
    Creates a profile function on a grid, from profile function `initial_profile_function`.

    Parameters
    ----------
    grid_points: Grid
        The x-values of a grid.
    initial_profile_function: function
        A function which accepts and returns a 1D numpy array

    Returns
    -------
    profile: np.array
        Generated profile function
    """
    profile = initial_profile_function(grid_points)
    return profile