import warp as wp

@wp.func
def dphi_tetdq(X0: wp.vec3d, X1: wp.vec3d, X2: wp.vec3d, X3: wp.vec3d) -> wp.mat((4,3), dtype=wp.float64):
    """
    Computes the gradient of tetrahedral shape functions with respect to reference coordinates.
    
    This Warp function calculates the gradient matrix of the linear tetrahedral shape functions
    with respect to the reference coordinates. This matrix is used to compute the inverse
    reference gradient matrix dXinv, which is essential for finite element computations.
    
    Parameters
    ----------
    X0 : wp.vec3d
        Reference position of the first vertex of the tetrahedron.
    X1 : wp.vec3d
        Reference position of the second vertex of the tetrahedron.
    X2 : wp.vec3d
        Reference position of the third vertex of the tetrahedron.
    X3 : wp.vec3d
        Reference position of the fourth vertex of the tetrahedron.
    
    Returns
    -------
    wp.mat((4,3), dtype=wp.float64)
        The gradient matrix of shape (4,3) representing the derivatives of the tetrahedral
        shape functions with respect to the reference coordinates.  The D matrix from the notes.
    
    Notes
    -----
    This is a Warp function (@wp.func) that can be called from within Warp kernels.
    
    Examples
    --------
    >>> # Within a Warp kernel
    >>> X0, X1, X2, X3 = ....
    >>> dphi = dphi_tetdq(X0, X1, X2, X3)
    >>> # dphi can now be used to compute dXinv and other finite element quantities
    """
     

    #YOUR CODE HERE

    return build_D@Tinv
