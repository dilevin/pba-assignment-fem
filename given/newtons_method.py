import warp as wp
import warp.optim.linear as wpol
import warp.sparse as wps
import torch

def newtons_method(q: torch.Tensor, energy_func, gradient_func, hessian_func, P: wps.BsrMatrix = None):

    #make a diagonal preconditioner
    H = hessian_func(q)
    g = gradient_func(q)
   

    if P is None:
        #linearly implicit for now just to save sometime and get things going 
        M = wpol.preconditioner(H, 'diag')
        dq = 0*q.detach().clone()
        wpol.cg(H, g, wp.from_torch(dq.reshape((-1,3)), dtype=wp.vec3d), tol=1e-5, maxiter=10000, M = M)
        q -= dq.reshape((-1,))
    else:
        #crush everything down, solve re-expand to full size
        H_proj = P@H@P.transpose()
        M = wpol.preconditioner(H_proj, 'diag')
        dq = wp.zeros((P.nrow,), dtype=wp.vec3d, device=P.device)
        wpol.cg(H_proj, P@g, dq, tol=1e-5, maxiter=10000, M = M)
        q -= wp.to_torch(P.transpose()@dq).reshape((-1,))
    
