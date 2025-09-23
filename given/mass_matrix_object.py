import warp as wp
import warp.sparse as wps
import torch
from assignment import *

def mass_matrix_object(M_out: wps.BsrMatrix, vertices: torch.Tensor, tets: torch.Tensor , rho: torch.float64, volumes: torch.Tensor, M_blk: wps.BsrMatrix, E_mat: wps.BsrMatrix):
    

    #step 1:
    #call kernel tofill in block diagonal mass matrix 
    @wp.kernel
    def fill_block_diagonal_mass_matrix(M_blk: wp.array(dtype=wp.mat((3,3), dtype=wp.float64)), rho: wp.float64, volumes: wp.array(dtype=wp.float64)):

        tid = wp.tid()
       
        M_tet = mass_matrix_tet(rho, volumes[tid])
        for i in range(4):
            for j in range(4): #somewhat annoying conversion from 12x12 scalar to block 4x4 stored as list of 3x3 matrices 
                M_blk[16*tid + j + i*4] = wp.matrix(M_tet[3*i,3*j],M_tet[3*i,3*j+1],M_tet[3*i,3*j+2],M_tet[3*i+1,3*j],M_tet[3*i+1,3*j+1],M_tet[3*i+1,3*j+2],M_tet[3*i+2,3*j],M_tet[3*i+2,3*j+1],M_tet[3*i+2,3*j+2], shape=(3,3), dtype=wp.float64) #setting scalar entries doesn't work in warp 
    
    
    wp.launch(fill_block_diagonal_mass_matrix, dim=tets.shape[0], inputs=[M_blk.values, rho, wp.from_torch(volumes)], device=wp.device_from_torch(vertices.device))        

    #step 2: 
    #assembly via multiplication with selection matrices 
    if M_out is None:
        M_out = wps.bsr_zeros(rows_of_blocks=vertices.shape[0], cols_of_blocks=vertices.shape[0], block_type=wp.mat((3,3),dtype=wp.float64), device=wp.device_from_torch(vertices.device))

    M_out = E_mat.transpose()@M_blk@E_mat
    return M_out
