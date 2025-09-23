"""
Configuration classes for simulator setup
"""

from dataclasses import dataclass, field, MISSING
from typing import List, Optional, Union


@dataclass
class MaterialConfig:
    """Configuration for material properties."""
    density: float = 1e3  # kg/m^3
    material_model: str = "NeoHookean"
    youngs: float = 1e6  # Young's modulus (in Pa)
    poissons: float = 0.4  # Poisson's ratio
    thickness: float = 0.01  # Thickness in meters (for thin shells only)
    
    def __post_init__(self):
        """Validate material parameters after initialization."""
        if self.density <= 0:
            raise ValueError("Density must be positive")
        if self.youngs <= 0:
            raise ValueError("Young's modulus must be positive")
        if not (0 <= self.poissons < 0.5):
            raise ValueError("Poisson's ratio must be in range [0, 0.5)")
        if self.thickness <= 0:
            raise ValueError("Thickness must be positive")


@dataclass
class SolverConfig:
    """Configuration for Newton solver."""
    max_iterations: int = 300
    tolerance: float = 1e-3
    
    def __post_init__(self):
        """Validate solver parameters after initialization."""
        if self.max_iterations <= 0:
            raise ValueError("Max iterations must be positive")
        if self.tolerance <= 0:
            raise ValueError("Tolerance must be positive")


@dataclass
class ObjectConfig:
    """Configuration for a simulation object."""
    geometry_type: str = "solid"  # "solid" or "cloth"
    mesh: str = ""  # Path to mesh file
    initial_velocity: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    transform: List[List[float]] = field(default_factory=lambda: [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    material: MaterialConfig = field(default_factory=MaterialConfig)
    pinned_vertices: Optional[List[int]] = None  # List of pinned vertex indices
    
    def __post_init__(self):
        """Validate object configuration after initialization."""
        if not self.mesh:
            raise ValueError("Mesh path must be specified")
        
        # Validate transform matrix (should be 4x4)
        if len(self.transform) != 4:
            raise ValueError("Transform matrix must have 4 rows")
        for row in self.transform:
            if len(row) != 4:
                raise ValueError("Transform matrix must have 4 columns")
        
        # Validate initial velocity (should be 3D)
        if len(self.initial_velocity) != 3:
            raise ValueError("Initial velocity must be 3D")
        
        # Validate geometry type
        if self.geometry_type not in ["solid", "shell"]:
            raise ValueError("Geometry type must be 'solid' or 'shell'")


@dataclass
class SimulationConfig:
    """Main configuration class for the simulation."""
    objects: List[ObjectConfig] = MISSING
    timestep: float = 0.01
    gravity: List[float] = field(default_factory=lambda: [0.0, -9.8, 0.0])
    solver_settings: SolverConfig = field(default_factory=SolverConfig)
    
    def __post_init__(self):
        """Validate simulation configuration after initialization."""
        if self.timestep <= 0:
            raise ValueError("Timestep must be positive")
        
        if len(self.gravity) != 3:
            raise ValueError("Gravity must be 3D")

        if not self.objects:
            raise ValueError("At least one object must be specified")