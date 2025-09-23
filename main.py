# This file constains the main application code for the assignment
# DO NOT MODIFY THIS FILE, to complete the assignment you need
# only correctly modify the files in the ./assignment directory
import polyscope as ps
import polyscope.imgui as psim
import warp as wp
import torch
import argparse 
import igl
from assignment import *
from given import *
from utils import *


wp.init() #initialize warp

#GUI global variables
simulating = False

#global variables for state of simulation 
sim_thin_shell = False

vertices = None
tets = None
triangles = None
tet_volumes = None

#simulation variables
E_mat = None #global selection matrix 
q = None #global vertex position vector at time t
qm1 = None #global vertex position vector at time t-1
q_pred = None #predictor position for implicit integration 
time = 0.0 #current simulation time
dt = 0.1 #simulation time step
gravity = None #torch.tensor([0.0, -9.8, 0.0], dtype=sim_dtype, device=sim_device) #default gravitational acceleration
rho = 1000.0 #default density in kg/m^3
Mass_matrix = None
H_energy = None
a_gravity = None 
dXinv = None
grad_energy = None
params = None
H_blk = None #temporary storage for hessian computation 
Pinned_matrix = None
obj_transform = None
obj_initial_velocity = None

def simulation_init(vertices: torch.Tensor, tets: torch.Tensor):

    print("Initializing simulation")
    #init quantities needed for simulation
    global q, qm1,E_mat, a_gravity,tet_volumes, Mass_matrix, dXinv, grad_energy, H_blk, H_energy, obj_transform, obj_initial_velocity

    print("Initializing selection matrix")
    E_mat = element_selection_matrix(vertices, tets)

    #compute volumes 
    print("Computing volumes")
    tet_volumes = torch.from_numpy(igl.volume(vertices.detach().cpu().numpy(), tets.detach().cpu().numpy())).to(sim_device)

    print("Initializing simulation")

    #Build mass matrix
    M_blk = block_diagonal_identity(wp.mat((3,3),dtype=wp.dtype_from_torch(sim_dtype)), 4, 4, tets.shape[0], sim_device, sim_dtype)
    H_blk = block_diagonal_identity(wp.mat((3,3),dtype=wp.dtype_from_torch(sim_dtype)), 4, 4, tets.shape[0], sim_device, sim_dtype)

    print("Initializing simulation")

    Mass_matrix = mass_matrix_object(Mass_matrix, vertices, tets, rho, tet_volumes, M_blk, E_mat)

    print("Initializing simulation")

    #make global gravity vector for the simulation
    a_gravity = gravity.tile((vertices.shape[0],)).to(sim_device)

    #initialize warp discrete gradient operators using a kernel
    dXinv = torch.zeros((tets.shape[0], 4, 3), dtype=sim_dtype, device=sim_device)
    tet_verts = wp.indexedarray(data=wp.from_torch(vertices, dtype=wp.vec3d), indices=wp.from_torch(tets.reshape((-1,))))
    grad_energy = torch.zeros((3*vertices.shape[0], ), dtype=sim_dtype, device=sim_device)
    
    @wp.kernel
    def compute_dXinv(dXinv: wp.array(dtype=wp.mat((4,3), dtype=wp.float64)), tet_verts: wp.indexedarray(dtype=wp.vec3d)):
        tid = wp.tid()
        vert_id = wp.int32(4)*tid
        dXinv[tid] = dphi_tetdq(tet_verts[vert_id], tet_verts[vert_id+1], tet_verts[vert_id+2], tet_verts[vert_id+3])
    
    wp.launch(compute_dXinv, dim=tets.shape[0], inputs=[wp.from_torch(dXinv, dtype=wp.mat((4,3), dtype=wp.float64)), tet_verts], device=wp.device_from_torch(vertices.device))

#callback to run one simulation step
def simulation_step():
    
    global q, qm1, q_pred, a_gravity,Mass_matrix,  grad_energy, params, H_blk, H_energy, dt, Pinned_matrix

    q_pred = (q + (q-qm1) + dt*dt*a_gravity).detach().clone()

    #define lambda functions for  energy, gradient and hessian
    energy_func = lambda q: 0.0
    gradient_func = lambda q: Mass_matrix@wp.from_torch(q - q_pred) + wp.from_torch(dneohookean_energy_dq_object(grad_energy,q, tets, dXinv, params, tet_volumes).reshape(-1,3),dtype=wp.vec3d)*wp.float64(dt)*wp.float64(dt)
    hessian_func = lambda q: Mass_matrix + d2neohookean_energy_dq2_object(H_energy,q, tets, dXinv, params, tet_volumes, E_mat,H_blk)*wp.float64(dt)*wp.float64(dt)

    qm1 = q.detach().clone()

    #compute new position
    newtons_method(q, energy_func, gradient_func, hessian_func, Pinned_matrix)
    
    
   

def ui_callback():
    global simulating,q, qm1
    # start top simulation button with text box for end time and dt
    #checkbox for write to USD
    changed_sim, simulating = psim.Checkbox("Start Simulation", simulating)

    #reset button
    if psim.Button("Reset Simulation"):
        print("Resetting Simulation")
        transformed_vertices = vertices.detach().clone()@obj_transform[:3,:3].T + obj_transform[:3,3]
        q = transformed_vertices.reshape((-1,))    
        qm1 = (transformed_vertices.detach().clone() - dt*obj_initial_velocity).reshape((-1,))

    
    if simulating:
        simulation_step()
        #update the vertices in the mesh
        ps.get_surface_mesh("mesh").update_vertex_positions(q.detach().cpu().reshape((-1,3)).numpy())

if __name__ == "__main__":
    #check arguments, load approriate model and test configuration
    parser = argparse.ArgumentParser(description="Physics-Based Animation Assignment program")
    parser.add_argument("--mesh", help="Path to the input file")
    parser.add_argument("--scene", help="Path to the scene file")
    parser.add_argument("--usd_output", help="Path to usd output directory, requires num_steps parameters")
    parser.add_argument("--num_steps", help="Number of steps to simulate", type=int)
    parser.add_argument("--device", help="Device to use", type=str, default="cpu")
    args = parser.parse_args()

    if args.device == "cpu":
        sim_dtype = torch.float64
        sim_device = "cpu"
    elif args.device == "cuda":
        sim_dtype = torch.float64
        sim_device = "cuda"

    sim_device_wp = wp.device_from_torch(sim_device)
    wp.set_device(sim_device_wp)

    if args.scene:
        config = load_config(args.scene)
        dt = config.timestep
        sim_thin_shell = True

        if config.objects[0].geometry_type == "solid":
            vertices, tets, _ = igl.readMESH(config.objects[0].mesh)
            triangles,_,_ = igl.boundary_facets(tets)
            sim_thin_shell = False
        else:
            vertices, _, _, triangles, _, _ = igl.read_obj(args.mesh)

        #load transform matrix if it exists, otherwise use identity
        if config.objects[0].transform:
            obj_transform = torch.Tensor(config.objects[0].transform).to(sim_device).to(sim_dtype)
        else:
            transform = torch.eye(4,4, dtype=sim_dtype, device=sim_device)

        if config.objects[0].initial_velocity:
            obj_initial_velocity = torch.Tensor(config.objects[0].initial_velocity).to(sim_device).to(sim_dtype)
        else:
            obj_initial_velocity = torch.zeros(3, dtype=sim_dtype, device=sim_device)

        #convert everything to torch tensors
        vertices = torch.tensor(vertices, dtype=sim_dtype, device=sim_device)
        tets = torch.tensor(tets, dtype=torch.int32, device=sim_device)
        triangles = torch.tensor(triangles, dtype=torch.int32, device=sim_device)

        rho = config.objects[0].material.density
        gravity = torch.Tensor(config.gravity)
        #mu and lambda for neohookean energy
        params = torch.tensor([config.objects[0].material.youngs, config.objects[0].material.poissons], dtype=sim_dtype, device=sim_device).tile((tets.shape[0],1))
        params[:,1], params[:,0] = emu2lame(params[:,0], params[:,1])

        #check for pinned vertices and build BC projection matrix
        if config.objects[0].pinned_vertices:

            if isinstance(config.objects[0].pinned_vertices, list):
                pinned_vertices = config.objects[0].pinned_vertices
            else:
                pinned_vertices = torch.arange(0,vertices.shape[0],device=sim_device)[config.objects[0].pinned_vertices(vertices)]
            
            Pinned_matrix = boundary_projection_matrix(Pinned_matrix, vertices, pinned_vertices)
        else:
            Pinned_matrix = None 
            
    elif args.mesh:
        #add command line argument to take in usd file and write sim to file without starting polyscope 
        if args.mesh.endswith(".obj"):
            sim_thin_shell = True
            vertices, _, _, triangles, _, _ = igl.read_obj(args.mesh)
        else:
            sim_thin_shell = False
            vertices, tets, _ = igl.readMESH(args.mesh)
            
            print(f"Loaded {len(vertices)} vertices and {len(tets)} tets")
            #surface mesh for rendering
            triangles,_,_ = igl.boundary_facets(tets)
            
            #convert everything to torch tensors
            vertices = torch.tensor(vertices, dtype=sim_dtype, device=sim_device)
            tets = torch.tensor(tets, dtype=torch.int32, device=sim_device)
            triangles = torch.tensor(triangles, dtype=torch.int32, device=sim_device)

            # Initialize material parameters for mesh-only case
            params = torch.tensor([1e6, 0.4], dtype=sim_dtype, device=sim_device).repeat(tets.shape[0], 1)
            params[:,1], params[:,0] = emu2lame(params[:,0], params[:,1])

            
       
    else:
        print("No scene or mesh file provided")
        exit()
        


    #if aabb of mesh is bigger than 10m rescale to 1m
    if vertices.max() - vertices.min() > 10.0:
        vertices = vertices / (vertices.max() - vertices.min())

    transformed_vertices = vertices.detach().clone()@obj_transform[:3,:3].T + obj_transform[:3,3]
    q = transformed_vertices.reshape((-1,))    
    qm1 = (transformed_vertices.detach().clone() - dt*obj_initial_velocity).reshape((-1,))

    simulation_init(vertices, tets)

    if args.usd_output:
        if args.num_steps: 
            print("Writing to USD")
            
            face_counts = 3*torch.ones(triangles.shape[0], dtype=torch.int32, device=sim_device)
            writer = USDMeshWriter(args.usd_output, fps=1.0/dt, up_axis="Y", write_velocities=False)
            writer.open(face_counts=face_counts.detach().cpu().numpy(), face_indices=triangles.detach().reshape((-1,)).cpu().numpy(), num_points=vertices.shape[0])

            for k in range(args.num_steps):
                print(f"Simulating step {k}")
                writer.write_frame(q.detach().clone().cpu().reshape((-1,3)).numpy(), sim_up="Y")
                simulation_step()
                

            writer.close()

                

            exit()
        else:
            print("Num steps not provided, skipping USD output")
            exit()

    #load mesh into polyscope
    ps.init()
   
    #launch polyscope and display surface mesh
    ps.register_surface_mesh("mesh", vertices.detach().cpu().numpy(), triangles.detach().cpu().numpy())
    ps.set_user_callback(ui_callback)

    #turn off polyscope ground plane
    ps.set_ground_plane_mode("none")
    ps.show()

