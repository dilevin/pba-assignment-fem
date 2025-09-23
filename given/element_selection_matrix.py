import warp as wp
import warp.sparse as wps 
import torch 

def element_selection_matrix(vertices: torch.Tensor, elements: torch.Tensor) -> wps.BsrMatrix:
    """
    Constructs an element selection matrix for finite element analysis.
    
    This function creates a block sparse matrix (BSR format) that maps from global vertex 
    coordinates to element-local coordinates. Each element is represented by a 3x3 identity 
    block in the matrix, allowing for efficient selection and transformation of vertex data 
    for individual elements.
    
    Parameters
    ----------
    vertices : torch.Tensor
        Global vertex coordinates tensor of shape (n_vertices, 3) where n_vertices is the 
        total number of vertices in the mesh.
    elements : torch.Tensor
        Element connectivity tensor of shape (n_elements, vertices_per_element) containing 
        vertex indices that define each element. Typically vertices_per_element=4 for 
        tetrahedral elements.
    
    Returns
    -------
    wps.BsrMatrix
        Block sparse matrix of shape (n_elements * vertices_per_element, n_vertices) 
        with 3x3 identity blocks. This matrix can be used to select and transform 
        vertex data for individual elements in finite element computations.
    
    Notes
    -----
    The resulting matrix has the structure:
    - Each row corresponds to a vertex within an element
    - Each column corresponds to a global vertex
    - 3x3 identity blocks are placed at positions (element_vertex, global_vertex)
    - This allows efficient extraction of element-local vertex data from global arrays
    
    Examples
    --------
    >>> vertices = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    >>> elements = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    >>> E = element_selection_matrix(vertices, elements)
    >>> # E can now be used to select vertex data for element 0
    """

    # Ensure consistent device usage - use vertices.device for consistency
    wp_device = wp.device_from_torch(vertices.device)
    dtype = vertices.dtype
    device = vertices.device

    # Create BSR matrix with consistent dtype and device
    wp_dtype = wp.dtype_from_torch(dtype)
    E_mat = wps.bsr_zeros(
        rows_of_blocks=elements.shape[0]*elements.shape[1], 
        cols_of_blocks=vertices.shape[0], 
        block_type=wp.mat((3,3), dtype=wp_dtype), 
        device=wp_device
    )
    
    # Create block values with consistent dtype and device
    block_values = torch.tile(torch.eye(3, dtype=dtype, device=device), (elements.shape[0]*elements.shape[1], 1, 1))
    block_row_indices = torch.arange(0, elements.shape[0]*elements.shape[1], device=device, dtype=torch.int32)
    block_col_indices = elements.reshape(-1).to(torch.int32)

    # Convert to warp tensors with consistent device and dtype
    wp_row_indices = wp.from_torch(block_row_indices, dtype=wp.int32)
    wp_col_indices = wp.from_torch(block_col_indices, dtype=wp.int32)
    wp_block_values = wp.from_torch(block_values, dtype=wp.mat((3,3), dtype=wp_dtype))

    wps.bsr_set_from_triplets(E_mat, wp_row_indices, wp_col_indices, wp_block_values)

    return E_mat
       
    