import warp as wp

@wp.func
def dneohookean_energy_dF_tet(F: wp.mat((3,3), dtype=wp.float64), params: wp.vec2d) -> wp.vec(length=9, dtype=wp.float64):
    """
    Computes the first derivative (gradient) of Neo-Hookean energy with respect to the vectorized deformation gradient.
    
    This Warp function calculates the gradient of the Neo-Hookean hyperelastic energy density
    function with respect to the vectorizeddeformation gradient F. The gradient is used in the computation
    of internal forces in finite element analysis.
    
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
    wp.vec(length=9, dtype=wp.float64)
        The gradient vector of length 9 representing the first derivative of the
        Neo-Hookean energy density with respect to the deformation gradient components.
        The vector is organized with deformation gradient components flattened row-wise:
        [∂W/∂F_00, ∂W/∂F_01, ∂W/∂F_02, ∂W/∂F_10, ∂W/∂F_11, ∂W/∂F_12, ∂W/∂F_20, ∂W/∂F_21, ∂W/∂F_22].
    
    Notes
    -----
    This is a Warp function (@wp.func) that can be called from within Warp kernels.
    The Neo-Hookean energy density is given by:
    W(F) = (μ/2) * (tr(F^T F) - 3) - μ * (det(F) - 1) + (λ/2) * (det(F) - 1)^2
    
    The gradient represents the first Piola-Kirchhoff stress tensor components and is
    used to compute internal forces in the finite element system.
    
    The implementation uses analytically derived expressions for the first derivatives,
    which are more accurate and efficient than numerical differentiation.
    
    Examples
    --------
    >>> # Within a Warp kernel
    >>> F = ...
    >>> params = ...
    >>> grad = dneohookean_energy_dF_tet(F, params)
    >>> # grad can now be used to compute internal forces
    """

    #YOUR CODE HERE

    return grad
