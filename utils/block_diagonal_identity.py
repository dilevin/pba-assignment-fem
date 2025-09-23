import warp as wp
import warp.sparse as wps
import torch
def block_diagonal_identity(warp_subblock_type, row_subblocks, col_subblocks, num_blocks, torch_device, torch_dtype):
    """
    Creates a block diagonal matrix with identity sub-blocks.
    
    This Python function constructs a block diagonal matrix where each block is an identity
    matrix of the specified sub-block type. The matrix is stored in Warp's block sparse
    row (BSR) format for efficient sparse matrix operations.
    
    Parameters
    ----------
    warp_subblock_type : type
        The Warp matrix type for the sub-blocks (e.g., wp.mat((3,3), dtype=wp.float64)).
        This defines the size and data type of each block in the matrix.
    row_subblocks : int
        The number of sub-blocks per row in each block of the matrix.
    col_subblocks : int
        The number of sub-blocks per column in each block of the matrix.
    num_blocks : int
        The number of blocks along the diagonal of the matrix.
    torch_device : str or torch.device
        The PyTorch device where the matrix data will be stored (e.g., "cpu", "cuda").
    torch_dtype : torch.dtype
        The PyTorch data type for the matrix elements (e.g., torch.float64).
    
    Returns
    -------
    wps.BsrMatrix
        A block sparse matrix in BSR format with identity sub-blocks along the diagonal.
        The matrix has dimensions (row_subblocks * num_blocks) × (col_subblocks * num_blocks)
        with each sub-block being an identity matrix of the specified type.
    
    Notes
    -----
    This is a Python function that operates on PyTorch tensors and Warp sparse matrices.
    The resulting matrix has the structure:
    
    [I  0  0  ...]
    [0  I  0  ...]
    [0  0  I  ...]
    [... ... ...]
    
    where I represents an identity sub-block of the specified type.
    
    The matrix is constructed by:
    1. Creating identity matrices for each sub-block
    2. Setting up row and column indices for the BSR format
    3. Using Warp's BSR matrix construction functions
    
    This function is commonly used to initialize mass matrices, stiffness matrices,
    and other block-structured matrices in finite element analysis.
    
    Examples
    --------
    >>> # Create a 3x3 block diagonal matrix with 2x2 identity blocks
    >>> M = block_diagonal_identity(wp.mat((2,2), dtype=wp.float64), 2, 2, 3, "cpu", torch.float64)
    >>> # M is now a 6x6 matrix with 2x2 identity blocks along the diagonal
    """

    #block sizes and total matrix size
    rows = row_subblocks * num_blocks
    cols = col_subblocks * num_blocks
    total_subblocks = row_subblocks*col_subblocks*num_blocks
    
    wp_device = wp.device_from_torch(torch_device)
    subblock_rows = warp_subblock_type._shape_[0]
    subblock_cols = warp_subblock_type._shape_[1]

    
    #value, row and column indices for all the subblocks
    block_values = torch.eye(subblock_rows, dtype=torch_dtype, device=torch_device).unsqueeze(0).repeat(total_subblocks, 1, 1)
    block_row_indices = torch.arange(0, rows, dtype=torch.int32, device=torch_device).repeat(col_subblocks,1).transpose(0,1).reshape(-1,1)
    block_col_indices = torch.arange(0, cols, dtype=torch.int32, device=torch_device).reshape(-1, col_subblocks).repeat(1,row_subblocks).reshape(-1,1)
    
    #create bsr matrix
    A = wps.bsr_zeros(rows_of_blocks=rows, cols_of_blocks=cols, block_type=warp_subblock_type, device=wp_device)
    
    #set from subblock triplets
    wps.bsr_set_from_triplets(A, wp.from_torch(block_row_indices,dtype=wp.int32), wp.from_torch(block_col_indices,dtype=wp.int32), wp.from_torch(block_values, dtype=warp_subblock_type))
    
    #return matrix
    return A 
