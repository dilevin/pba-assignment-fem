import warp as wp

@wp.func
def d2neohookean_energy_dF2_tet(F: wp.mat((3,3), dtype=wp.float64), params: wp.vec2d) -> wp.mat((9,9), dtype=wp.float64):
    """
    Computes the second derivative (Hessian) of Neo-Hookean energy with respect to 9x1 vectorized deformation gradient.
    
    This Warp function calculates the 9x9 Hessian matrix of the Neo-Hookean hyperelastic energy
    density function with respect to the vectorized deformation gradient F. The Hessian is used in Newton's
    method for solving nonlinear finite element systems.
    
    Parameters
    ----------
    F : wp.mat((3,3), dtype=wp.float64)
        The deformation gradient tensor of shape (3,3). This matrix describes the local
        deformation of the material element and maps from reference to current configuration.
    params : wp.vec2d
        Material parameters vector containing [mu, lambda] where:
        - mu: Second Lame parameter (shear modulus)
        - lambda: First Lame parameter (bulk modulus)
        These parameters define the Neo-Hookean material model.
    
    Returns
    -------
    wp.mat((9,9), dtype=wp.float64)
        The Hessian matrix of shape (9,9) representing the second derivative of the
        Neo-Hookean energy density with respect to the deformation gradient components.
        The matrix is organized with deformation gradient components flattened row-wise:
        [F_00, F_01, F_02, F_10, F_11, F_12, F_20, F_21, F_22].
    
    Notes
    -----
    This is a Warp function (@wp.func) that can be called from within Warp kernels.
    The Neo-Hookean energy density is given by:
    W(F) = (μ/2) * (tr(F^T F) - 3) - μ * (det(F) - 1) + (λ/2) * (det(F) - 1)^2
    
    The Hessian matrix is symmetric and positive definite for physically reasonable
    deformations. 
    
    The implementation uses analytically derived expressions for the second derivatives,
    which are more accurate and efficient than numerical differentiation.
    
    Examples
    --------
    >>> # Within a Warp kernel
    >>> F = ...
    >>> params = ...
    >>> H = d2neohookean_energy_dF2_tet(F, params)
    >>> # H can now be used in Newton's method for nonlinear solving
    """

    #YOUR CODE HERE
    
    return H