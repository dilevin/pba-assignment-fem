import warp as wp
import torch

@wp.func
def neohookean_energy_tet(F: wp.mat((3,3), dtype=wp.float64), params: wp.vec2d) -> wp.float64:
    """
    Computes the Neo-Hookean hyperelastic energy density for a tetrahedral element.
    
    This Warp function calculates the energy density of the Neo-Hookean hyperelastic
    material model given the deformation gradient. The Neo-Hookean model is commonly
    used for modeling rubber-like materials and soft tissues.
    
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
    wp.float64
        The Neo-Hookean energy density value. This scalar represents the strain energy
        per unit reference volume stored in the material due to deformation.
    
    Notes
    -----
    This is a Warp function (@wp.func) that can be called from within Warp kernels.
    The Neo-Hookean energy density is given by:
    W(F) = (μ/2) * (tr(F^T F) - 3) - μ * (det(F) - 1) + (λ/2) * (det(F) - 1)^2
    
    Where:
    - tr(F^T F) is the trace of F^T F (first invariant of the right Cauchy-Green tensor)
    - det(F) is the determinant of F (volume change)
    - μ is the shear modulus
    - λ is the bulk modulus
    
    The energy density is used in finite element analysis to compute the total potential
    energy of the system, which is minimized to find the equilibrium configuration.
    
    Examples
    --------
    >>> # Within a Warp kernel
    >>> F = ...
    >>> params = ...
    >>> energy = neohookean_energy_tet(F, params)
    >>> # energy can now be used to compute total system energy
    """

    #YOUR CODE HERE

    return E
