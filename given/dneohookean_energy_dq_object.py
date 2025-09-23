import warp as wp
import warp.sparse as wps
import torch
from assignment import *

def dneohookean_energy_dq_object(grad_out: torch.Tensor, q: torch.Tensor, tets: torch.Tensor, dXinv: torch.Tensor, params: torch.Tensor, volumes: torch.Tensor):

    grad_out.zero_()
    
    @wp.kernel
    def dneohookean_energy_dq_object_kernel(grad_out: wp.array(dtype=wp.vec3d), q: wp.array(dtype=wp.vec3d), tets: wp.array2d(dtype=wp.int32), dXinv: wp.array(dtype=wp.mat((4,3), dtype=wp.float64)), params: wp.array(dtype=wp.vec2d), volumes: wp.array(dtype=wp.float64)):
        tid = wp.tid()
        
        v0 = q[tets[tid,0]]
        v1 = q[tets[tid,1]]
        v2 = q[tets[tid,2]]
        v3 = q[tets[tid,3]]

        F = deformation_gradient_tet(v0,v1,v2,v3, dXinv[tid])

        B = B_tet(dXinv[tid])

        grad_verts = volumes[tid]*wp.transpose(B)@dneohookean_energy_dF_tet(F, params[tid])

        wp.atomic_add(grad_out, tets[tid,0],wp.vector(grad_verts[0], grad_verts[1], grad_verts[2], length=3, dtype=wp.float64))
        wp.atomic_add(grad_out, tets[tid,1],wp.vector(grad_verts[3], grad_verts[4], grad_verts[5], length=3, dtype=wp.float64))
        wp.atomic_add(grad_out, tets[tid,2],wp.vector(grad_verts[6], grad_verts[7], grad_verts[8], length=3, dtype=wp.float64))
        wp.atomic_add(grad_out, tets[tid,3],wp.vector(grad_verts[9], grad_verts[10], grad_verts[11], length=3, dtype=wp.float64))


    wp.launch(dneohookean_energy_dq_object_kernel, dim=tets.shape[0], \
         inputs=[wp.from_torch(grad_out.reshape((-1,3)), dtype=wp.vec3d), wp.from_torch(q.reshape((-1,3)), dtype=wp.vec3d), \
                 wp.from_torch(tets, dtype=wp.int32), wp.from_torch(dXinv, dtype=wp.mat((4,3), dtype=wp.float64)), \
                 wp.from_torch(params, dtype=wp.vec2d), wp.from_torch(volumes, dtype=wp.float64)], \
         device=wp.device_from_torch(q.device))
    
    return grad_out
