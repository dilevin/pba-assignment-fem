## Some diagnostic functions to make my life easier 
import torch
import warp as wp
import warp.sparse as wsp 

def to_torch_tensor(bsp_mat: wsp.BsrMatrix) -> torch.Tensor:
    """
    Converts a Warp block sparse row (BSR) matrix to a dense PyTorch tensor.
    
    This Python function converts a Warp BSR matrix to a dense PyTorch tensor for
    debugging, visualization, or operations that require dense matrix representation.
    The conversion reconstructs the full dense matrix from the sparse BSR format.
    
    Parameters
    ----------
    bsp_mat : wsp.BsrMatrix
        The Warp block sparse row matrix to convert. This matrix is stored in BSR
        format with block structure defined by the matrix's block type.
    
    Returns
    -------
    torch.Tensor
        A dense PyTorch tensor representing the full matrix. The tensor has shape
        (nrow * block_size_rows, ncol * block_size_cols) where nrow and ncol are
        the number of block rows and columns, and block_size_rows/cols are the
        dimensions of each block.
    
    Notes
    -----
    This is a Python function that operates on Warp sparse matrices and PyTorch tensors.
    The conversion process:
    1. Extracts the block values, row offsets, and column indices from the BSR matrix
    2. Reconstructs the dense matrix by placing each block at its correct position
    3. Returns the result as a PyTorch tensor on the same device as the input matrix
    
    This function is primarily used for debugging and visualization purposes, as
    dense matrix operations are generally less efficient than sparse operations
    for large finite element systems.
    
    Warning: This function can consume significant memory for large matrices,
    as it creates a full dense representation of the sparse matrix.
    
    Examples
    --------
    >>> # Convert a BSR matrix to dense tensor for debugging
    >>> bsr_matrix = create_bsr_matrix(...)
    >>> dense_tensor = to_torch_tensor(bsr_matrix)
    >>> print(f"Dense matrix shape: {dense_tensor.shape}")
    """
    #convert a bsr matrix to a dense torch tenser and return
    block_size_rows = bsp_mat.values.dtype._shape_[0]
    block_size_cols = bsp_mat.values.dtype._shape_[1]
    tensor = torch.zeros((bsp_mat.nrow*block_size_rows, bsp_mat.ncol*block_size_cols), dtype=wp.dtype_to_torch(bsp_mat.dtype._wp_scalar_type_), device=wp.device_to_torch(bsp_mat.device))
    
    values = wp.to_torch(bsp_mat.values)
    row_offsets = wp.to_torch(bsp_mat.offsets)
    col_indices = wp.to_torch(bsp_mat.columns)                   
   
    for i in range(row_offsets.shape[0]-1):
        for j in range(row_offsets[i],row_offsets[i+1]):
            tensor[i*block_size_rows:(i+1)*block_size_rows, col_indices[j]*block_size_cols:(col_indices[j]+1)*block_size_cols] = values[j,:,:]

    return tensor
