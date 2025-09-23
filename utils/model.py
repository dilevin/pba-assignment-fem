"""
Model class for loading simulation data into Warp arrays.
Handles both solid meshes (tetrahedra) and cloth meshes (triangles).
"""

import numpy as np
import warp as wp
import igl
from pathlib import Path
from typing import List, Tuple, Optional

from .config import SimulationConfig

class State:
    """State storing solver variables that are updated each timestep."""
    
    def __init__(self):
        self.qt = None
        self.qdot = None
        self.q = None

class Model:
    """Model class that loads simulation configuration."""
    
    def __init__(self, config: SimulationConfig, device: str = "cpu"):
        """
        Initialize model from simulation configuration.
        
        Args:
            config: Simulation configuration
            device: Warp device ("cpu" or "cuda")
        """
        self.config = config
        self.device = device
        
        # Device arrays (will be converted to Warp arrays in finalize)
        self.vertices = []  # List of vertex arrays from all objects
        self.tri_indices = []  # List of triangle indices from all objects
        self.tet_indices = []  # List of tetrahedra indices from all objects
        self.initial_velocities = []  # List of initial velocity arrays from all objects
        # TODO: also store the per-element material parameters.
        
        # Host arrays
        self.object_vertex_ranges = []  # (start, end) vertex indices for each object
        self.object_tri_ranges = []  # (start, end) triangle indices for each object
        self.object_tet_ranges = []  # (start, end) tetrahedra indices for each object
        self.object_types = []  # "solid" or "shell" for each object

        # Load all objects
        self._load_objects()
        
        # Convert to Warp arrays
        self._finalize()
    
    def _load_objects(self):
        """Load all objects from the configuration."""
        vertex_offset = 0
        tri_offset = 0
        tet_offset = 0
        
        for obj_idx, obj_config in enumerate(self.config.objects):
            print(f"Loading object {obj_idx}: {obj_config.mesh} ({obj_config.geometry_type})")
            
            # Track vertex range for this object
            vertex_start = vertex_offset
            
            # Load mesh based on geometry type
            if obj_config.geometry_type == "solid":
                vertices, tets = self._load_solid_mesh(obj_config.mesh)
                
                # Apply transform to vertices
                vertices = self._apply_transform(vertices, obj_config.transform)
                
                # Create initial velocities array
                initial_velocities = np.tile(obj_config.initial_velocity, (len(vertices), 1)).astype(np.float32)
                
                # Append data (solid objects only have tetrahedra)
                self.vertices.append(vertices)
                self.tet_indices.append(tets + vertex_offset)
                self.initial_velocities.append(initial_velocities)
                self.object_types.append("solid")
                
                # Track ranges
                vertex_end = vertex_offset + len(vertices)
                tri_start = tri_offset  # No triangles for solids
                tri_end = tri_offset
                tet_start = tet_offset
                tet_end = tet_offset + len(tets)
                
                # Update offsets
                vertex_offset = vertex_end
                tet_offset = tet_end
                
            elif obj_config.geometry_type == "shell":
                vertices, triangles = self._load_shell_mesh(obj_config.mesh)
                
                # Apply transform to vertices
                vertices = self._apply_transform(vertices, obj_config.transform)
                
                # Create initial velocities array
                initial_velocities = np.tile(obj_config.initial_velocity, (len(vertices), 1)).astype(np.float32)
                
                # Append data (shell objects only have triangles)
                self.vertices.append(vertices)
                self.tri_indices.append(triangles + vertex_offset)
                self.initial_velocities.append(initial_velocities)
                self.object_types.append("shell")
                
                # Track ranges
                vertex_end = vertex_offset + len(vertices)
                tri_start = tri_offset
                tri_end = tri_offset + len(triangles)
                tet_start = tet_offset  # No tetrahedra for shells
                tet_end = tet_offset
                
                # Update offsets
                vertex_offset = vertex_end
                tri_offset = tri_end
            else:
                raise ValueError(f"Unknown geometry type: {obj_config.geometry_type}")
            
            # Store ranges for this object
            self.object_vertex_ranges.append((vertex_start, vertex_end))
            self.object_tri_ranges.append((tri_start, tri_end))
            self.object_tet_ranges.append((tet_start, tet_end))
    
    def _load_solid_mesh(self, mesh_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load solid mesh (tetrahedra) using igl.readMESH."""
        vertices, tets, _ = igl.readMESH(mesh_path)
        return vertices.astype(np.float32), tets.astype(np.int32)
    
    def _load_shell_mesh(self, mesh_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load shell mesh (triangles) using igl.read_triangle_mesh."""
        vertices, triangles = igl.read_triangle_mesh(mesh_path)
        return vertices.astype(np.float32), triangles.astype(np.int32)
    
    def _apply_transform(self, vertices: np.ndarray, transform_matrix: List[List[float]]) -> np.ndarray:
        """Apply transform matrix to vertices."""
        # Convert 4x4 matrix to Warp transform
        A = np.array(transform_matrix)
        R = A[:3, :3]
        t = A[:3, 3]
        return vertices @ R + t
    
    def _finalize(self):
        """Convert loaded data to Warp arrays."""
        # Combine all vertices into a single array
        all_vertices = np.vstack(self.vertices)

        # Combine all triangle indices
        if self.tri_indices:
            all_tri_indices = np.vstack(self.tri_indices)
        else:
            all_tri_indices = np.array([], dtype=np.int32).reshape(0, 3)
        
        # Combine all tetrahedra indices
        if self.tet_indices:
            all_tet_indices = np.vstack(self.tet_indices)
        else:
            all_tet_indices = np.array([], dtype=np.int32).reshape(0, 4)
        
        # Combine all initial velocities
        all_initial_velocities = np.vstack(self.initial_velocities)
        
        # Overwrite original arrays with Warp arrays
        self.vertices = wp.array(all_vertices, dtype=wp.vec3, device=self.device)
        self.tri_indices = wp.array(all_tri_indices, dtype=wp.int32, device=self.device)
        self.tet_indices = wp.array(all_tet_indices, dtype=wp.int32, device=self.device)
        self.initial_velocities = wp.array(all_initial_velocities, dtype=wp.vec3, device=self.device)
    
    def state(self) -> State:
        """Clone current model data to a new state."""
        # qt and q are the same initially (transformed positions)
        state = State()
        state.q = wp.clone(self.vertices)
        state.qdot = wp.clone(self.initial_velocities)
        state.qt = wp.clone(self.vertices)
        return state
            
    def print_summary(self):
        """Print a summary of the loaded model."""
        print(f"\n=== Model Summary ===")
        print(f"Device: {self.device}")
        print(f"Total vertices: {len(self.vertices)}")
        print(f"Total triangles: {len(self.tri_indices)}")
        print(f"Total tetrahedra: {len(self.tet_indices)}")
        print(f"Total initial velocities: {len(self.initial_velocities)}")
        print(f"Number of objects: {len(self.object_vertex_ranges)}")
        
        for i, obj_config in enumerate(self.config.objects):
            vertex_start, vertex_end = self.object_vertex_ranges[i]
            print(f"\nObject {i}: {obj_config.mesh}")
            print(f"  Type: {obj_config.geometry_type}")
            print(f"  Vertex range: {vertex_start}-{vertex_end} ({vertex_end - vertex_start} vertices)")
            print(f"  Material: {obj_config.material.material_model}")
            print(f"  Density: {obj_config.material.density} kg/m³")
            print(f"  Thickness: {obj_config.material.thickness} m")
            print(f"  Initial velocity: {obj_config.initial_velocity}")
            print(f"  Transform: {obj_config.transform}")
            
            if obj_config.pinned_vertices is not None:
                print(f"  Pinned vertices: {len(obj_config.pinned_vertices)}")
