import warp as wp

@wp.func
def mass_matrix_tet(rho: wp.float64, volume: wp.float64) -> wp.mat((12,12), dtype=wp.float64):
    """
    Computes the consistent mass matrix for a tetrahedral element.
    
    This function calculates the 12x12 mass matrix for a tetrahedral element using 
    the consistent mass matrix formulation. The matrix is structured as a 12x12 matrix 
    with specific coefficients for diagonal and off-diagonal elements. The coefficients 
    are derived from analytical integration of the shape functions over the tetrahedral domain.
    
    Parameters
    ----------
    rho : wp.float64
        Material density of the tetrahedral element in units of mass per unit volume.
    volume : wp.float64
        Volume of the tetrahedral element. This should be precomputed using the
        tetrahedron volume formula.
    
    Returns
    -------
    wp.mat((12,12), dtype=wp.float64)
       
    Notes
    -----
    This matrix is used in finite element analysis for dynamic simulations where
    inertial effects need to be properly accounted for.
    
    Examples
    --------
    >>> # Assuming rho and volume are properly defined
    >>> M = mass_matrix_tet(rho, volume)
    >>> # M is now a 12x12 mass matrix for the tetrahedral element
    """

    #YOUR CODE HERE

    return M_tet