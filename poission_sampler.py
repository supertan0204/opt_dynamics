import taichi as ti
import taichi.math as tm
import numpy as np
import h5py
import os
import trimesh
from datetime import datetime

from math import sqrt

ti.init(arch=ti.cpu)


@ti.func
def erfinv(x):
        sgn = -1 if x < 0 else 1
        x = (1 - x) * (1 + x)
        lnx = tm.log(x)
        tt1 = 2 / (tm.pi * 0.147) + 0.5 * lnx
        tt2 = 1 / 0.147 * lnx
        return sgn * ti.sqrt(-tt1 + ti.sqrt(tt1**2 - tt2))

@ti.func
def normalrandom():
    samp = ti.random()
    return ti.sqrt(2) * erfinv(samp * 2 - 1)

@ti.data_oriented
class PoissonDiskSampler():
    def __init__(self, radius, sample_ratio, total_mass):
        self.radius = radius
        self.sample_ratio = sample_ratio
        self.total_mass = total_mass

        self.dx = radius / sqrt(2)
        self.inv_dx = 1.0 / self.dx
        self.bounding_box_width = None
        self.res = None
        self.desired_samples = None
        # grid_size = tuple(self.res.tolist())
        self.grid = None
        self.samples = None 
        
        self.mesh = None
        self.bounding_box_lower_left = None


        self.sampled = False # whether the mesh has been sampled
        self.num_samples = ti.field(dtype=int, shape=())

        self.mesh_query = None

    def load_mesh(self, path):
        self.mesh = trimesh.load_mesh(path, use_embree=False)
        print(f"mesh loaded successfully from {path}")
        bounding_box_lower_left = self.mesh.bounding_box.bounds[0,:]
        bounding_box_width = self.mesh.bounding_box.extents
        self.bounding_box_lower_left = tm.vec3(bounding_box_lower_left)
        bounding_box_width = tm.vec3(bounding_box_width)
        print("lower left: ", bounding_box_lower_left)
        print("width: ", bounding_box_width)

        # initialize value from mesh
        self.bounding_box_width = bounding_box_width
        self.res = ti.ceil(self.bounding_box_width / self.dx)
        self.desired_samples = int(ti.ceil(self.res[0]*self.res[1]*self.res[2]*self.sample_ratio))
        self.grid = ti.field(dtype=int, shape=self.res)
        self.samples = ti.Vector.field(3, float, shape=int(self.desired_samples))
        self.grid.fill(-1)

        self.mesh_query = trimesh.proximity.ProximityQuery(self.mesh)

    @ti.func
    def check_collision(self, p, index):
        x, y, z = index[0], index[1], index[2]
        collision = False
        for i in range(max(0, x - 2), min(self.res[0], x + 3)):
            for j in range(max(0, y - 2), min(self.res[1], y + 3)):
                for k in range(max(0, z - 2), min(self.res[2], z + 3)):
                    if self.grid[i,j,k] != -1:
                        q = self.samples[self.grid[i,j,k]]
                        
                        if (q - p).norm() < self.radius - 1e-6:
                            collision = True
        return collision


    @ti.kernel
    def sample(self) -> ti.int32:
        assert self.mesh != None, "mesh doesn't exist, exiting..."

        self.samples[0] = self.bounding_box_width/2.0 # default is center
        self.grid[int(self.res[0] * 0.5), int(self.res[1] * 0.5), int(self.res[2] * 0.5)] = 0
        head, tail = 0, 1
        while head < tail and head < self.desired_samples:
            print(f"head: {head}; tail: {tail}")
            source_x = self.samples[head]
            head += 1
            for _ in range(200):
                rho = (1 + ti.random()) * self.radius
                x1 = normalrandom()
                x2 = normalrandom()
                x3 = normalrandom()
                denominator = 1/ti.sqrt(x1*x1+x2*x2+x3*x3+1e-6)
                offset = tm.vec3(rho*denominator*x1, rho*denominator*x2, rho*denominator*x3)
                new_x = source_x + offset
                new_index = int(new_x * self.inv_dx)
                # check if new_x inside the mesh
                # if self.mesh.contains([new_x.tolist()])[0]:
                if 0 <= new_x[0] < self.bounding_box_width[0] and 0 <= new_x[1] < self.bounding_box_width[1] and 0 <= new_x[2] < self.bounding_box_width[2]:
                    collision = self.check_collision(new_x, new_index)
                    if not collision and tail < self.desired_samples:
                        self.samples[tail] = new_x
                        self.grid[new_index] = tail
                        tail += 1

        for i in range(tail):
            self.samples[i] += self.bounding_box_lower_left

        self.num_samples[None] = tail
        return tail
    
    def save_sample_to_file(self, fullfilename, flag_p2d=True, save_to_obj=True, shuffle=True):
        '''
        default would be a .h5 file, you can choose whether to save to a .obj file 
        '''
        samples_np = self.samples.to_numpy() # (n, 3)
        samples_np = samples_np[:self.num_samples[None], :] # (n, 3)
        bool_all_samples = np.full((self.num_samples[None]), True)
        chunk_size = 10000
        num_chunk = ti.ceil(self.num_samples[None]/chunk_size)

        chunks_ok_points = np.array_split(samples_np, num_chunk)
        current_count = 0
        for k in range(int(num_chunk)):
            chunk_k = chunks_ok_points[k]
            dist = self.mesh_query.signed_distance(chunk_k)
            dist = dist > 0 # inside mesh
            bool_all_samples[current_count:current_count+dist.shape[0]] = dist
            current_count = current_count + dist.shape[0]
            print(current_count, "points have been filtered")

        assert(current_count == self.num_samples[None])
        samples_np = samples_np[bool_all_samples, :]
        print(f"{samples_np.shape[0]} particles are inside mesh. They are saved")

        samples_np[:,2] = -samples_np[:,2]
        samples_np[:,[1,2]] = samples_np[:,[2,1]]

        if shuffle:
            np.random.seed(1031)
            np.random.shuffle(samples_np) # shuffle the order of points
        
        num_inside = samples_np.shape[0]
        samples_np = samples_np.transpose() # (3, n)
    
        if os.path.exists(fullfilename):
            os.remove(fullfilename)

        newFile = h5py.File(fullfilename, "w")
        newFile.create_dataset("x", data=samples_np) # initial position

        if flag_p2d:
            zero_disp = np.zeros_like(samples_np)
            newFile.create_dataset("q", data=zero_disp) # displacement
        else:
            newFile.create_dataset("q", data=samples_np) # deformed position = original position

        p_mass_np = np.ones((1, samples_np.shape[1])) * (self.total_mass/samples_np.shape[1])
        newFile.create_dataset("masses", data=p_mass_np) # particle mass
        p_volume_np = np.ones((1, samples_np.shape[1])) * (self.bounding_box_width[0]*self.bounding_box_width[1]*self.bounding_box_width[2]) / self.num_samples[None]
        newFile.create_dataset("particle_volume", data=p_volume_np)
        print(f"density is {p_mass_np[0,0]/p_volume_np[0,0]}")
        newFile.create_dataset("particle_density", data=p_mass_np/p_volume_np) # particle density

        ###################### Time ###########################
        currentTime = np.array([0])
        currentTime = currentTime.reshape(1,1)
        newFile.create_dataset("time", data=currentTime)
        #######################################################

        if flag_p2d:
            f_tensor_np = np.full((samples_np.shape[1], 3, 3), np.zeros((3,3)))
        else:
            f_tensor_np = np.full((samples_np.sahpoe[1], 3, 3), np.eye(3))

        newFile.create_dataset("f_tensor", data=f_tensor_np) # deforamtion gradient
        print(f"save poisson sampling data: {fullfilename}")

        if save_to_obj:
            fullfilename_obj = fullfilename[:-2] + 'obj'
            if os.path.exists(fullfilename_obj):
                os.remove(fullfilename_obj)
            objfile = open(fullfilename_obj, 'w')
            for k in range(samples_np.shape[1]):
                line = "v " + str(samples_np[0, k]) + " " + str(samples_np[1, k]) + " " + str(samples_np[2, k]) + "0 0 0"
                objfile.write(line)
                objfile.write('\n')

            print("save poisson sampling data: ", fullfilename_obj)
        
        return num_inside, p_mass_np[0,0]/p_volume_np[0,0]

        