"""
Two tetrahedra scene configuration.
Two tets with one moving and one pinned for testing.
"""

from data import get_data_directory
from utils import SimulationConfig, ObjectConfig, MaterialConfig, SolverConfig

class TwoTetsConfig(SimulationConfig):
    """Two tetrahedra scene configuration."""
    
    def __init__(self):
        # Material for both tets
        tet_material = MaterialConfig(
            density=1000.0,
            material_model="NeoHookean",
            youngs=1e3,  # 100 kPa
            poissons=0.3,
        )
        
        # Solver settings
        solver_settings = SolverConfig(
            max_iterations=200,
            tolerance=1e-3
        )
        
        # First tet - moving
        tet1 = ObjectConfig(
            geometry_type="solid",
            mesh=str(get_data_directory() / "two_tets.mesh"),
            initial_velocity=[1.0, 0.0, 0.0],  # Moving to the right
            transform=[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ],
            material=tet_material,
            pinned_vertices=[0]
        )
        
        # Initialize simulation configuration
        super().__init__(
            objects=[tet1],
            timestep=0.01,
            gravity=[0.0, -9.8, 0.0],
            solver_settings=solver_settings
        )
