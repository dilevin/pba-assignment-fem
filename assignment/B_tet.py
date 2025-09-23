import warp as wp

@wp.func 
def B_tet(dXinv: wp.mat((4,3), dtype=wp.float64)) -> wp.mat((9,12), dtype=wp.float64):
    """
    Constructs the B-matrix for tetrahedral finite elements.
    
    This Warp function creates a 9x12 matrix that transforms element vertex displacements
    to the row-flattened deformation gradient. The B-matrix is a fundamental component
    in finite element analysis for computing strain-displacement relationships.
    
    Parameters
    ----------
    dXinv : wp.mat((4,3), dtype=wp.float64)
        This is the D matrix described in class
    
    Returns
    -------
    wp.mat((9,12), dtype=wp.float64)
        The B-matrix of shape (9,12) that transforms 12 vertex displacement components
        (4 vertices × 3 DOFs each) to 9 deformation gradient components (3×3 matrix
        flattened row-wise). This matrix is used to compute the deformation gradient
        from vertex displacements: F = B * q where q is the per-tetrahedron vector of generalized coordinates
    
    Notes
    -----
    This is a Warp function (@wp.func) that can be called from within Warp kernels.
    
    The matrix is constructed by restructuring the dXinv matrix elements with appropriate
    zero padding to create the proper finite element B-matrix structure.
    
    """
    
    #this just retructures dXinv into a 9x12 matrix that can be used to compute the row floattened deformation gradient
    #matrix
    
    #YOUR CODE HERE

    return B
