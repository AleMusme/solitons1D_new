import numpy as np 
import matplotlib.pyplot as plt
from typing import Callable

import json
from pathlib import Path
import pickle as pkl

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

            self.grid_length = grid_end - grid_start 
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

    def save(
        self,
        folder_name: str | Path
    ):
        """
        Saves a `Grid` object at `folder_name`.
        """
        metadata = {
            "num_grid_points": self.num_grid_points,
            "grid_spacing": self.grid_spacing
        }     #this is a dictionary object. needed to add to json

        # make the folder a Path if it is a string
        folder = Path(folder_name)
        folder.mkdir(exist_ok = True)

        # this overwrites any existing metadata.json file
        with open(folder / "metadata.json", "w") as f:
            json.dump(metadata, f)
        #here "w" means write mode, with automatically closes the file when it's done

def load_grid(folder_name: str | Path):
    """
    Loads the `Grid` object at `folder_name`.
    """       
    folder = Path(folder_name)
    metadata_path = folder / "metadata.json"

    #control file is there
    assert metadata_path.is_file(), f"Could not find Grid `metadata.json` file in {folder}."

    with open(metadata_path , "r") as f:
        grid_metadata = json.load(f)

    # the ** "unpacks" the dictionary into a series of arguments
    grid = Grid(**grid_metadata)
    return grid


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

    def save(
        self,
        folder_name: str | Path,
    ):
        """
        Saves a `Lagrangian` object at `folder_name`.
        """
        metadata = {
            "V": self.V,
            "dV": self.dV,
            "vacua": self.vacua,
        }

        # make the folder a Path if it is a string
        folder = Path(folder_name)
        folder.mkdir(exist_ok = True)

        with open(folder / "metadata.pkl", "wb") as f:
            pkl.dump(metadata, f)

def load_lagrangian(folder_name: str | Path):
    """
    Loads the `Lagrangian` object at `folder_name`.
    """
    folder = Path(folder_name)
    metadata_path = folder / "metadata.pkl"

    assert metadata_path.is_file(), f"Could not find Lagrangian `metadata.pkl` file in {folder}."
    
    with open(metadata_path, "rb") as f:
        lagrangian_metadata = pkl.load(f)
    
    # the ** "unpacks" the dictionary into a series of arguments
    lagrangian = Lagrangian(**lagrangian_metadata)
    return lagrangian

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
        
        #self.compute_energy()

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
    
    def plot_soliton(self):
        """Makes a plot of the profile function of your soliton"""

        fig, ax = plt.subplots()
        ax.plot(self.grid.grid_points, self.profile)
        ax.set_title(f"Profile function. Energy = {self.energy:.4f}")
        ax.grid()

        return fig
    
    def save(
        self,
        folder_name: str | Path,
    ):
        """
        Saves a `Soliton` object at `folder_name`.
        """

        folder = Path(folder_name)
        folder.mkdir(exist_ok = True)

        grid_folder =  "./grid"
        lagrangian_folder = "./lagrangian"

        metadata = {
            'grid_folder': str(grid_folder),
            'lagrangian_folder': str(lagrangian_folder),
        }

        properties = {
            'energy': self.energy
        }

        self.grid.save(grid_folder)
        self.lagrangian.save(lagrangian_folder)

        with open(folder / "metadata.json", "w") as f:
            json.dump(metadata, f)

        with open(folder / "properties.json", "w") as f:
            json.dump(properties, f)
        
        # use `numpy`s save function to save the profile array
        np.save("profile", self.profile)

def load_soliton(folder_name):
    """
    Loads the `Lagrangian` object at `folder_name`.
    """
    folder = Path(folder_name)
    metadata_path = folder / "metadata.json"

    assert metadata_path.is_file(), f"Could not find Grid `metadata.json` file in {folder}."
    
    with open(metadata_path, "r") as f:
        soliton_metadata = json.load(f)

    grid_folder = soliton_metadata.get("grid_folder")
    grid = load_grid(grid_folder)

    lagrangian_folder = soliton_metadata.get("lagrangian_folder")
    lagrangian = load_lagrangian(lagrangian_folder)

    profile = np.load("profile.npy")
    
    soliton = Soliton(grid = grid, lagrangian=lagrangian, initial_profile=profile)

    return soliton

def compute_energy_fast(
    V: Callable[[float], float], 
    profile: np.ndarray, 
    num_grid_points: int, 
    grid_spacing: float
) -> float:
    """
    Computes the energy of a Lagrangian of the form
        E = 1/2 (d_phi)^2 + V(phi)

    Parameters
    ----------
    V: function
        The potential energy function
    profile: np.ndarray
        The profile function of the soliton
    num_grid_points: int
        Length of `profile`
    grid_spacing: float
        Grid spacing of underlying grid
    """
    d_profile = get_first_derivative(profile, num_grid_points, grid_spacing)

    kin_eng = 0.5 * np.pow(d_profile, 2)
    pot_eng = V(profile)

    tot_eng = np.sum(kin_eng + pot_eng) * grid_spacing

    return tot_eng

def get_first_derivative(
    phi: np.ndarray, 
    num_grid_points: int, 
    grid_spacing: float,
) -> np.ndarray:
    """
    For a given array, computes the first derivative of that array.

    Parameters
    ----------
    phi: np.ndarray
        Array to get the first derivative of
    num_grid_points: int
        Length of the array
    grid_spacing: float
        Grid spacing of underlying grid

    Returns
    -------
    d_phi: np.ndarray
        The first derivative of `phi`.

    """
    d_phi = np.zeros(num_grid_points)
    for i in np.arange(num_grid_points)[2:-2]:
        d_phi[i] = (phi[i - 2] - 8 * phi[i - 1] + 8 * phi[i + 1] - phi[i + 2]) / (
            12.0 * grid_spacing
        )

    return d_phi

def get_second_derivative(
    phi: np.ndarray,
    num_grid_points: int,
    grid_spacing: float
) -> np.ndarray:
    """
    For a given array, computes the second derivative of that array.

    Parameters
    ----------
    phi: np.ndarray
        Array to get the first derivative of
    num_grid_points: int
        Length of the array
    grid_spacing: float
        Grid spacing of underlying grid

    Returns
    -------
    dd_phi: np.ndarray
        The second derivative of `phi`.

    """
    d_phi = get_first_derivative(phi, num_grid_points, grid_spacing)
    dd_phi = np.zeros(num_grid_points)

    for i in np.arange(num_grid_points)[2:-2]:
        dd_phi[i] = (d_phi[i - 2] - 8 * d_phi[i - 1] + 8 * d_phi[i + 1] - d_phi[i + 2]) / (
            12.0 * grid_spacing
        )
    
    return dd_phi

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