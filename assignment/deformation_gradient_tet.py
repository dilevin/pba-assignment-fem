import warp as wp

@wp.func
def deformation_gradient_tet(q0: wp.vec3d, q1: wp.vec3d, q2: wp.vec3d, q3: wp.vec3d, dXinv: wp.mat((4,3), dtype=wp.float64)) -> wp.mat33d:
    """
    Computes the 3x3 deformation gradient matrix for a tetrahedral element.
    
    This Warp function calculates the deformation gradient tensor F (3x3 matrix) for a tetrahedral
    element given its current vertex positions and the inverse dphi matrix (D in the notes) matrix.
    The deformation gradient describes the local deformation of the material element.
    
    Parameters
    ----------
    q0 : wp.vec3d
        Current position of the first vertex of the tetrahedron.
    q1 : wp.vec3d
        Current position of the second vertex of the tetrahedron.
    q2 : wp.vec3d
        Current position of the third vertex of the tetrahedron.
    q3 : wp.vec3d
        Current position of the fourth vertex of the tetrahedron.
    dXinv : wp.mat((4,3), dtype=wp.float64)
        D matrix in the notes
    
    Returns
    -------
    wp.mat33d
        The deformation gradient tensor F of shape (3,3). This matrix describes how
        the tetrahedral element has deformed from its reference configuration to its
        current configuration. F maps infinitesimal vectors from reference to current
        configuration: dx = F * dX.
    
    Notes
    -----
    This is a Warp function (@wp.func) that can be called from within Warp kernels.
    The deformation gradient is a fundamental quantity in continuum mechanics and
    finite element analysis, used to compute strains, stresses, and energy densities.
    
    Examples
    --------
    >>> # Within a Warp kernel
    >>> q0, q1, q2, q3 = ...
    >>> dXinv = ...
    >>> F = deformation_gradient_tet(q0, q1, q2, q3, dXinv)
    >>> # F can now be used to compute material response
    """
    
    #YOUR CODE HERE

    return F 