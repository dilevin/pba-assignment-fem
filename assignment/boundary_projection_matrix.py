import warp as wp
import warp.sparse as wps
import torch

def boundary_projection_matrix(P_out: wps.BsrMatrix, vertices: torch.Tensor, fixed_indices: list[int]) -> wps.BsrMatrix:
    """
    Creates a projection matrix to eliminate fixed boundary vertices from the solution.
    
    This Python function constructs a block sparse matrix that projects out fixed boundary
    vertices from the finite element system. The resulting matrix can be used to reduce
    the system size by removing degrees of freedom corresponding to fixed vertices.
    
    Parameters
    ----------
    P_out : wps.BsrMatrix
        Output block sparse matrix that will store the projection matrix. If None,
        a new matrix will be created with appropriate dimensions. Each block is a 3x3
        identity matrix.
    vertices : torch.Tensor
        Tensor containing vertex coordinates of the mesh. Shape: (n_vertices, 3).
        Used to determine the total number of vertices in the system.
    fixed_indices : List[int]
        Tensor containing the indices of vertices that are fixed (have zero displacement
        boundary conditions). Shape: (n_fixed_vertices,). These vertices will be
        projected out of the system.
    
    Returns
    -------
    wps.BsrMatrix
        Block sparse projection matrix of shape ((n_vertices - n_fixed_vertices) × n_vertices)
        with 3x3 identity blocks. This matrix maps from the full system to the reduced
        system by selecting only the non-fixed vertices.
    
    Notes
    -----
    This is a Python function that operates on PyTorch tensors and Warp sparse matrices.
    The projection matrix P has the property that P * q_full = q_reduced, where q_full
    is the full vector of generalized coordinates and q_reduced contains only the generalized coordinates of
    non-fixed vertices.
    
    The matrix structure:
    - Each row corresponds to a non-fixed vertex
    - Each column corresponds to a vertex in the full system
    - 3x3 identity blocks are placed at positions (non_fixed_vertex, full_vertex)
    - This allows extraction of non-fixed vertex displacements from the full system
    
    Examples
    --------
    >>> vertices = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    >>> fixed_indices = torch.tensor([0])  # Fix first vertex
    >>> P = boundary_projection_matrix(None, vertices, fixed_indices)
    >>> # P can now be used to project out fixed vertices from the system
    """
    if P_out is None:
        P_out = wps.bsr_zeros(cols_of_blocks=vertices.shape[0], rows_of_blocks = (vertices.shape[0]-len(fixed_indices)), block_type=wp.mat33d, device=wp.device_from_torch(vertices.device))


    #YOUR CODE HERE
   
    return P_out


