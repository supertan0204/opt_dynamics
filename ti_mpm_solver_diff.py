import taichi as ti
import argparse
import os
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import torch
from poission_sampler import PoissonDiskSampler


dim = 3

real = ti.f32
ti.init(default_fp=real, arch=ti.gpu, flatten_if=True, device_memory_GB=3.5)
scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(dim, dtype=real)
mat = lambda: ti.Matrix.field(dim, dim, dtype=real)


# helper functions
@ti.func
def zero_vec():
    return [0.0, 0.0, 0.0]

@ti.func
def zero_matrix():
    return [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]


@ti.kernel
def assign_value_to_scalar_field(field: ti.template(), value: ti.float32):
    for i in field:
        field[i] = value

@ti.kernel
def assign_value_to_vec_field(field: ti.template(), value: ti.types.ndarray()):
    """
    field: (num_frame, num_particle) vector field
    value: (num_frame, num_particle, 3) numpy or torch array
    """
    for i in range(value.shape[0]):
        for j in range(value.shape[1]):
                field[i, j] = [value[i, j, 0], value[i, j, 1], value[i, j, 2]]

@ti.kernel
def learn(field: ti.template(), lr: ti.float32):
    for i in field:
        field[i] -= lr*field.grad[i]
        if field[i] > 55.0:
            field[i] = 55.0
        if field[i] < 0.1:
            field[i] = 0.1
        # ti.math.clamp(field[i], 1.0, 40.0)

@ti.kernel
def learn_iso(field: ti.template(), lr: ti.float32, grad_value:float):
    for i in field:
        field[i] -= lr*grad_value
        ti.math.clamp(field[i], 1.0, 40)


# splat 3d points to 2d screen for ggui visualization
def T(a):
    phi, theta = np.radians(28), np.radians(32)

    a = a - 0.5
    x, y, z = a[:, 0], a[:, 1], a[:, 2]
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    x, z = x * cp + z * sp, z * cp - x * sp
    u, v = x, y * ct + z * st
    return np.array([u, v]).swapaxes(0, 1) + 0.5



@ti.data_oriented
class Diff_MPM_Taichi():
    def __init__(self, n_particles=1000, steps = 512, dt=2e-3, n_grid = 64):
        self.dim = 3
        self.n_particles = n_particles
        self.n_solid_particles = n_particles
        self.n_grid = n_grid
        self.dx = 1/self.n_grid
        self.dt = dt
        self.inv_dx = 1/self.dx
        self.p_vol = 1
        self.max_steps = 512
        self.steps = steps
        self.gravity = 9.8



        self.E = scalar()
        self.nu = scalar()
        self.mu = scalar()
        self.lam = scalar()

        self.particle_type = ti.field(ti.i32)
        self.x, self.v = vec(), vec()
        self.x_avg = vec()
        self.pair_wise = scalar()
        self.grid_v_in, self.grid_m_in = vec(), scalar()
        self.grid_v_out = vec()
        self.C = mat()
        self.F = mat()

        # boundary conditions
        self.bound = 3
        self.coeff = 1.5


    def allocate_fields(self, n):
        self.n_particles = n
        ti.root.dense(ti.k, self.max_steps).dense(ti.l, self.n_particles).place(self.x, self.v, self.C, self.F)
        ti.root.dense(ti.ijk, self.n_grid).place(self.grid_v_in, self.grid_m_in, self.grid_v_out)
        ti.root.dense(ti.i, self.n_particles).place(self.particle_type)
        ti.root.dense(ti.i, self.n_particles).place(self.E, self.nu, self.mu, self.lam)
        ti.root.place(self.x_avg, self.pair_wise)
        # TODO: add loss outside Diff_MPM_Taichi
        # ti.root.place(loss, x_avg)
        # ti.root.dense(ti.ij,
        #               (visualize_resolution, visualize_resolution)).place(screen)

        ti.root.lazy_grad()

    @ti.kernel
    def compute_mu_lam_from_E_nu(self):
        for i in range(self.n_particles):
            self.mu[i] = self.E[i]/(2.*(1. + self.nu[i]))
            self.lam[i] = self.E[i]*self.nu[i]/((1.+self.nu[i])*(1. - 2.*self.nu[i]))


    def read_params_from_dict(self, dict):
        assign_value_to_scalar_field(self.E, dict["E"])
        assign_value_to_scalar_field(self.nu, dict["nu"])
        self.p_vol = dict["vol"]
        self.gravity = dict["g"]
        # self.E_value = dict["E"]
        # self.nu_value = dict["nu"]
        # self.mu_value = self.E_value/(2.*(1.+self.nu_value))
        # self.lam_value = self.E_value*self.nu_value/((1.+self.nu_value)*(1.-2.*self.nu_value))
        self.compute_mu_lam_from_E_nu()



    @ti.kernel
    def init_from_np(self, x_: ti.types.ndarray(element_dim=1), particle_type_arr: ti.types.ndarray()):
        """
        here x_, particle_type_arr should be numpy arrays
        """
        # n = x_.shape[0]
        # self.allocate_fields(n)
        for i in range(self.n_particles):
            self.x[0, i] = x_[i]
            self.F[0, i] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            self.particle_type[i] = particle_type_arr[i]


    @ti.kernel
    def clear_grid(self):
        for i, j, k in self.grid_m_in:
            self.grid_v_in[i, j, k] = [0, 0, 0]
            self.grid_m_in[i, j, k] = 0
            self.grid_v_in.grad[i, j, k] = [0, 0, 0]
            self.grid_m_in.grad[i, j, k] = 0
            self.grid_v_out.grad[i, j, k] = [0, 0, 0]
    @ti.kernel
    def clear_particle_grad(self):
        # for all time steps and all particles
        for f, i in self.x:
            self.x.grad[f, i] = zero_vec()
            self.v.grad[f, i] = zero_vec()
            self.C.grad[f, i] = zero_matrix()
            self.F.grad[f, i] = zero_matrix()
    @ti.kernel
    def p2g(self, f: ti.i32):
        for p in range(0, self.n_particles):
            base = ti.cast(self.x[f, p] * self.inv_dx - 0.5, ti.i32)
            fx = self.x[f, p] * self.inv_dx - ti.cast(base, ti.i32)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            new_F = (ti.Matrix.diag(dim=self.dim, val=1) + self.dt * self.C[f, p]) @ self.F[f, p]
            J = (new_F).determinant()
            if self.particle_type[p] == 0:  # fluid
                sqrtJ = ti.sqrt(J)
                # TODO: need pow(x, 1/3)
                new_F = ti.Matrix([[sqrtJ, 0, 0], [0, sqrtJ, 0], [0, 0, 1]])

            self.F[f + 1, p] = new_F
            cauchy = ti.Matrix(zero_matrix())
            mass = 0.0
            ident = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            if self.particle_type[p] == 0:
                mass = 4
                cauchy = ti.Matrix(ident) * (J - 1) * self.E[p]
            else:
                mass = 1
                cauchy = self.mu[p] * (new_F @ new_F.transpose()) + ti.Matrix(ident) * (
                    self.lam[p] * ti.log(J) - self.mu[p])
            stress = -(self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx) * cauchy
            affine = stress + mass * self.C[f, p]
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for k in ti.static(range(3)):
                        offset = ti.Vector([i, j, k])
                        dpos = (ti.cast(ti.Vector([i, j, k]), real) - fx) * self.dx
                        weight = w[i][0] * w[j][1] * w[k][2]
                        ti.atomic_add(self.grid_v_in[base + offset], weight * (mass * self.v[f, p] + affine @ dpos))
                        ti.atomic_add(self.grid_m_in[base + offset], weight * mass)

    @ti.kernel
    def grid_op(self):
        for i, j, k in self.grid_m_in:
            inv_m = 1 / (self.grid_m_in[i, j, k] + 1e-10)
            v_out = inv_m * self.grid_v_in[i, j, k]
            v_out[1] -= self.dt * self.gravity

            if i < self.bound and v_out[0] < 0:
                v_out[0] = 0
                v_out[1] = 0
                v_out[2] = 0
            if i > self.n_grid - self.bound and v_out[0] > 0:
                v_out[0] = 0
                v_out[1] = 0
                v_out[2] = 0

            if k < self.bound and v_out[2] < 0:
                v_out[0] = 0
                v_out[1] = 0
                v_out[2] = 0
            if k > self.n_grid - self.bound and v_out[2] > 0:
                v_out[0] = 0
                v_out[1] = 0
                v_out[2] = 0

            if j < self.bound and v_out[1] < 0:
                v_out[0] = 0
                v_out[1] = 0
                v_out[2] = 0
                normal = ti.Vector([0.0, 1.0, 0.0])
                lsq = (normal**2).sum()
                if lsq > 0.5:
                    if ti.static(self.coeff < 0):
                        v_out[0] = 0
                        v_out[1] = 0
                        v_out[2] = 0
                    else:
                        lin = v_out.dot(normal)
                        if lin < 0:
                            vit = v_out - lin * normal
                            lit = vit.norm() + 1e-10
                            if lit + self.coeff * lin <= 0:
                                v_out[0] = 0
                                v_out[1] = 0
                                v_out[2] = 0
                            else:
                                v_out = (1 + self.coeff * lin / lit) * vit
            if j > self.n_grid - self.bound and v_out[1] > 0:
                v_out[0] = 0
                v_out[1] = 0
                v_out[2] = 0

            self.grid_v_out[i, j, k] = v_out
    @ti.kernel
    def compute_x_avg(self):
        # compute the average of x after several steps
        for i in range(self.n_particles):
            contrib = 1.0 / self.n_particles
            ti.atomic_add(self.x_avg[None], contrib*self.x[self.steps, i])
    
    @ti.kernel
    def compute_E_grad_sum(self, E_grad_value: ti.template()):
        for i in range(self.n_particles):
            contrib = self.E.grad[i]
            ti.atomic_add(E_grad_value[None], contrib)

    @ti.kernel
    def compute_pairwise_loss(self, lam: ti.float32):
        for i in range(self.n_particles):
            for j in range(self.n_particles):
                contrib = lam/(self.n_particles*self.n_particles)
                diff = (self.E[i] - self.E[j])*(self.E[i] - self.E[j]) 
                ti.atomic_add(self.pair_wise[None], contrib*diff)
        # return self.pair_wise[None]

    @ti.kernel
    def g2p(self, f: ti.i32):
        for p in range(0, self.n_particles):
            base = ti.cast(self.x[f, p] * self.inv_dx - 0.5, ti.i32)
            fx = self.x[f, p] * self.inv_dx - ti.cast(base, real)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
            new_v = ti.Vector(zero_vec())
            new_C = ti.Matrix(zero_matrix())

            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for k in ti.static(range(3)):
                        dpos = ti.cast(ti.Vector([i, j, k]), real) - fx
                        g_v = self.grid_v_out[base[0] + i, base[1] + j, base[2] + k]
                        weight = w[i][0] * w[j][1] * w[k][2]
                        new_v += weight * g_v
                        new_C += 4 * weight * g_v.outer_product(dpos) * self.inv_dx

            self.v[f + 1, p] = new_v
            self.x[f + 1, p] = self.x[f, p] + self.dt * self.v[f + 1, p]
            self.C[f + 1, p] = new_C

    def forward(self, n):
        self.compute_mu_lam_from_E_nu()
        for s in range(n):
            self.clear_grid()
            self.p2g(s)
            self.grid_op()
            self.g2p(s)
        # self.x_avg[None] = [0,0,0]
        self.pair_wise[None] = 0
        # self.compute_x_avg()
        # self.compute_loss()
        self.clear_particle_grad()

        return self.pair_wise[None]

    # this is the adjoint phase of forward
    def backward(self, n):
        """
        dt: time step
        n: steps to backward
        """
        self.pair_wise.grad[None] = 1
        # grad = np.zeros((self.steps, self.n_particles, 3), dtype=np.float32)
        # grad[self.steps-1] = np.ones((self.n_particles, 3), dtype=np.float32)
        # assign_value_to_vec_field(self.x.grad, grad)

        # self.compute_x_avg.grad()
        for s in reversed(range(n)):
            # since we do not store the grid history (to save space), we redo p2g and grid op
            self.clear_grid()
            self.p2g(s)
            self.grid_op()
            self.g2p.grad(s)
            self.grid_op.grad()
            self.p2g.grad(s)
        self.compute_mu_lam_from_E_nu.grad() # pass gradient to E and nu

    def visualize_in_gui(self):
        self.forward(self.steps)

        gui = ti.GUI("Diff_MPM_Taichi", background_color=0x112F41)
        current_f = 0
        while gui.running and not gui.get_event(gui.ESCAPE):
            # print(self.x.to_numpy()[2])
            pos = self.x.to_numpy()[current_f]
            pos = T(pos)
            gui.circles(pos, radius=1.5, color=0x66CCFF)
            gui.show()
            current_f+=1

    def write_particle_position_to_ply(self, filename, frame):
        # position is (n,3)
        if os.path.exists(filename):
            os.remove(filename)
        position = self.x.to_numpy()[frame-1] # position: (num_partiles, 3)
        num_particles = (position).shape[0]
        position = position.astype(np.float32)
        with open(filename, 'wb') as f: # write binary
            header = f"""ply
                        format binary_little_endian 1.0
                        element vertex {num_particles}
                        property float x
                        property float y
                        property float z
                        end_header
                        """
            f.write(str.encode(header))
            f.write(position.tobytes())
            print("write", filename)







# def main():
#     n_particles = 10000
#     mpm_solver = Diff_MPM_Taichi(steps=300, n_grid=64)
#     x_np = np.random.uniform(0.4, 0.7, (n_particles, 3)).astype(np.float32)
#     # print(x_np)
#     particle_type = torch.ones(n_particles,dtype=torch.int32).numpy()
#     mpm_solver.allocate_fields(x_np.shape[0])
#     mpm_solver.init_from_np(x_np,particle_type)
#     param_dict = {
#         "E": 0.8,
#         "nu": 0.2,
#         "vol": 1,
#         "g": 9.8,
#     }
#     mpm_solver.read_params_from_dict(param_dict)
#     # mpm_solver.visualize_in_gui()
#     ti.ad.clear_all_gradients()


#     print(mpm_solver.E.grad)
#     lm = 1e2
#     mpm_solver.compute_pairwise_loss(lm)
#     # print(l_pwise)\
#     print(mpm_solver.E)
#     mpm_solver.pair_wise.grad[None] = 1
#     mpm_solver.compute_pairwise_loss.grad(lm)
#     print(mpm_solver.E.grad)



#     mpm_solver.forward(mpm_solver.steps)
#     # print(mpm_solver.x.grad.to_numpy().shape)
#     mpm_solver.backward(mpm_solver.steps)
#     print(mpm_solver.E.grad)


    # loss = []
    # for _ in range(50):
    #     ti.ad.clear_all_gradients() # do not forget to clear gradients before gradient calculation
    #     l = mpm_solver.forward(mpm_solver.steps)
    #     # print(mpm_solver.x_avg[None])
    #     print(l)
    #     loss.append(l)
    #     mpm_solver.backward(mpm_solver.steps)
    #     # print(mpm_solver.E.grad)
    #     mpm_solver.update_params(1e3)
    #     # print(diff_ti_solver.x.to_numpy()[1])
    #     # print(diff_ti_solver.x.to_numpy()[2])

    # print("ok")


if __name__ == "__main__":
    main()