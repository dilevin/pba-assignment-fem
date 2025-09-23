import torch
#convert young's modulus and poissons ration to mu and lambda
def emu2lame(E: torch.Tensor, mu: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    Converts Young's modulus and Poisson's ratio to Lame parameters.
    
    This Python function converts the standard engineering material parameters
    (Young's modulus and Poisson's ratio) to the Lame parameters (lambda and mu)
    used in continuum mechanics and finite element analysis.
    
    Parameters
    ----------
    E : torch.Tensor
        Young's modulus (elastic modulus) of the material. This represents the
        material's stiffness in tension/compression. Units: Pa (Pascals).
    mu : torch.Tensor
        Poisson's ratio of the material. This represents the ratio of lateral
        strain to axial strain under uniaxial stress. Dimensionless, typically
        in range [0, 0.5) for most materials.
    
    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple containing (lambda, mu) where:
        - lambda: First Lame parameter (bulk modulus component)
        - mu: Second Lame parameter (shear modulus)
        Both have the same shape and device as the input tensors.
    
    Notes
    -----
    This is a Python function that operates on PyTorch tensors.
    The conversion formulas are:
    - lambda = (E * mu) / ((1 + mu) * (1 - 2 * mu))
    - mu = E / (2 * (1 + mu))
    
    The Lame parameters are fundamental in continuum mechanics and are used
    in constitutive equations for elastic materials. They are particularly
    important in finite element analysis for hyperelastic materials like
    the Neo-Hookean model.
    
    Examples
    --------
    >>> E = torch.tensor(1e6)  # 1 MPa Young's modulus
    >>> nu = torch.tensor(0.4)  # Poisson's ratio
    >>> lambda_val, mu_val = emu2lame(E, nu)
    >>> print(f"Lambda: {lambda_val}, Mu: {mu_val}")
    """
    return (E*mu)/((1.0+mu)*(1.0-2.0*mu)),  E/(2.0*(1.0+mu))