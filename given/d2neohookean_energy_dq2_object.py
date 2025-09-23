import warp as wp
import warp.sparse as wps
import torch
from assignment import *

def d2neohookean_energy_dq2_object(H_out: wps.BsrMatrix, q: torch.Tensor, tets: torch.Tensor, dXinv: torch.Tensor, params: torch.Tensor, volumes: torch.Tensor, E_mat: wps.BsrMatrix, H_blk: wps.BsrMatrix = None):

    @wp.kernel
    def dneohookean_energy_dq_object_kernel(H_blk: wp.array(dtype=wp.mat33d), q: wp.array(dtype=wp.vec3d), tets: wp.array2d(dtype=wp.int32), dXinv: wp.array(dtype=wp.mat((4,3), dtype=wp.float64)), params: wp.array(dtype=wp.vec2d), volumes: wp.array(dtype=wp.float64)):
        tid = wp.tid()
        
        v0 = q[tets[tid,0]]
        v1 = q[tets[tid,1]]
        v2 = q[tets[tid,2]]
        v3 = q[tets[tid,3]]

        F = deformation_gradient_tet(v0,v1,v2,v3, dXinv[tid])

        B = B_tet(dXinv[tid])

        H_tet = volumes[tid]*wp.transpose(B)@d2neohookean_energy_dF2_tet(F, params[tid])@B

        for i in range(4):
            for j in range(4): #somewhat annoying conversion from 12x12 scalar to block 4x4 stored as list of 3x3 matrices 
                H_blk[16*tid + j + i*4] = wp.matrix(H_tet[3*i,3*j],H_tet[3*i,3*j+1],H_tet[3*i,3*j+2],H_tet[3*i+1,3*j],H_tet[3*i+1,3*j+1],H_tet[3*i+1,3*j+2],H_tet[3*i+2,3*j],H_tet[3*i+2,3*j+1],H_tet[3*i+2,3*j+2], shape=(3,3), dtype=wp.float64) #setting scalar entries doesn't work in warp 
    
    wp.launch(dneohookean_energy_dq_object_kernel, dim=tets.shape[0], \
         inputs=[H_blk.values, wp.from_torch(q.reshape((-1,3)), dtype=wp.vec3d), \
                 wp.from_torch(tets, dtype=wp.int32), wp.from_torch(dXinv, dtype=wp.mat((4,3), dtype=wp.float64)), \
                 wp.from_torch(params, dtype=wp.vec2d), wp.from_torch(volumes, dtype=wp.float64)], \
         device=wp.device_from_torch(q.device))
    
    if H_out is None:
        H_out = wps.bsr_zeros(rows_of_blocks=q.shape[0]//3, cols_of_blocks=q.shape[0]//3, block_type=wp.mat((3,3),dtype=wp.float64), device=wp.device_from_torch(q.device))

    H_out = E_mat.transpose()@H_blk@E_mat
    
    return H_out
