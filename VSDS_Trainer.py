import torch
from diffuser import GaussianDiffusion
from external.unet_sd import UNetSD
from external.autoencoder import AutoencoderKL
from external.text_to_video_synthesis_model import FrozenOpenCLIPEmbedder
from os import path
from torch.cuda.amp import custom_bwd, custom_fwd
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

from render_utils import *

from warp_mpm.mpm_solver_warp_diff import MPM_Simulator_WARP
from warp_mpm.engine_utils import *
from warp_mpm.warp_utils import *


from ti_mpm_solver_diff import Diff_MPM_Taichi, assign_value_to_vec_field, learn, learn_iso
import matplotlib.pyplot as plt

from sds_util import SDSVideoDiffuser, get_augmentations
from torchvision import transforms

import taichi as ti
from torch.utils.tensorboard import SummaryWriter

# initialize Nvidia Warp lang
wp.init()


CKPT_PATH = './check_points'
AE_MODEL = 'VQGAN_autoencoder.pth'
UNET_MODEL = 'text2video_pytorch_model.pth'
CLIP_MODEL = 'open_clip_pytorch_model.bin'
AE_CKPT_PATH = path.join(CKPT_PATH, AE_MODEL)
UNET_CKPT_PATH = path.join(CKPT_PATH, UNET_MODEL)
CLIP_CKPT_PATH = path.join(CKPT_PATH, CLIP_MODEL)


MODEL_CFG = {
            "unet_in_dim": 4,
            "unet_dim": 320,
            "unet_y_dim": 768,
            "unet_context_dim": 1024,
            "unet_out_dim": 4,
            "unet_dim_mult": [1, 2, 4, 4],
            "unet_num_heads": 8,
            "unet_head_dim": 64,
            "unet_res_blocks": 2,
            "unet_attn_scales": [1, 0.5, 0.25],
            "unet_dropout": 0.1,
            "temporal_attention": "True",
            "num_timesteps": 1000,
            "mean_type": "eps",
            "var_type": "fixed_small",
            "loss_type": "mse",
            "unet_device": "cuda"
        }

AE_CONFIG = {
            'double_z': True,
            'z_channels': 4,
            'resolution': 256,
            'in_channels': 3,
            'out_ch': 3,
            'ch': 128,
            'ch_mult': [1, 2, 4, 4],
            'num_res_blocks': 2,
            'attn_resolutions': [],
            'dropout': 0.0,
            'ae_device': "cpu"
        }

RENDER_CONFIG = {
    'dist': 10,
    'elev': 30,
    'znear': 0.01,
    'size': 256,
    'radius': 0.003,
    'points_per_pixel': 10,
    'background_color': (0, 0, 0)
}

TINY_GPU = 1
DEVICE = 'cpu'
if TINY_GPU==0 and torch.cuda.is_available():
    DEVICE = 'cuda'

def beta_schedule(schedule,
                  num_timesteps=1000,
                  init_beta=None,
                  last_beta=None):
    if schedule == 'linear_sd':
        return torch.linspace(
            init_beta**0.5, last_beta**0.5, num_timesteps,
            dtype=torch.float64)**2
    else:
        raise ValueError(f'Unsupported schedule: {schedule}')
    
class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        print("processing sds backward...")
        (gt_grad,) = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None

class TaichiSimulator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, trainer, params_tensor, num_frames, pre_steps=120):
        mpm_solver = trainer.setup_ti_solver_from_input(params_tensor)
        mpm_solver.forward(mpm_solver.steps)
        # read from solver
        frame_tensor = torch.from_numpy(mpm_solver.x.to_numpy())
        # adjust coordinate system
        frame_tensor[:,:,[1,2]] = frame_tensor[:,:,[2,1]]
        # fetch frames for renderer
        interpolate = (mpm_solver.steps - pre_steps) // num_frames
        frame_tensor = frame_tensor[pre_steps:mpm_solver.steps:interpolate, :, :]

        ctx.solver = mpm_solver
        ctx.target_params = trainer.target_params
        ctx.num_frames = num_frames
        ctx.pre_steps = pre_steps

        return frame_tensor
    
    @staticmethod
    def backward(ctx, grad_scale):
        print("processing mpm simulation backward...")
        mpm_solver = ctx.solver,
        target_params = ctx.target_params
        num_frames = ctx.num_frames
        pre_steps = ctx.pre_steps

        







class WarpSimulator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, trainer, params_tensor, num_frames, dt=2e-3, interpolate=3, pre_steps=60):
        # ensure Torch operations complete before running Warp
        wp.synchronize_device()

        mpm_solver = trainer.setup_solver_from_input(params_tensor)
        frame_tensor, tape = simulate_to_video_tensor(mpm_solver, num_frames, pre_steps, dt, interpolate)
        ctx.solver = mpm_solver
        ctx.target_params = trainer.target_params
        ctx.num_frames = num_frames
        ctx.pre_steps = pre_steps
        ctx.dt = dt
        ctx.interpolate = interpolate
        ctx.tape = tape

        return frame_tensor
    @staticmethod
    def backward(ctx, grad_scale):
        # grad_scale: (num_particles, 3, num_frames)
        wp.synchronize_device() # ensure Torch operations complete before running Warp


        print("Processing mpm simulation backward...")
        mpm_solver = ctx.solver
        target_params = ctx.target_params
        num_frames = ctx.num_frames
        pre_steps = ctx.pre_steps
        dt = ctx.dt
        tape = ctx.tape
        interpolate = ctx.interpolate

        params_label = target_params.keys()
        # set gradient initial value to all 1 to prevent zero gradient output
        # particle_x_grad_tensor = torch.full((mpm_solver.n_particles, 3), 1, dtype=float)
        # particle_x_grad_tensor_np = particle_x_grad_tensor.numpy()
        # x_grad_initial_value = [wp.vec3(*particle_x_grad_tensor_np[i]) for i in range(mpm_solver.n_particles)]

        jacobian_dict = {}
        grad_tensor = torch.zeros(10,dtype=float)
        for label in params_label:
            jacobian_dict[label] = np.empty((num_frames, mpm_solver.n_particles), dtype=float)
        
        grad_scale_acc = torch.sum(grad_scale, dim=1) 
        dV_dx = grad_scale_acc.permute(1,0) # (batch, num_frames, wp_solver.n_particles)
        # run adjoint process of mpm simulation
        particle_x_grad_tensor_np = grad_scale[:,:,num_frames-1].numpy()
        x_grad_initial_value = [wp.vec3(*particle_x_grad_tensor_np[i]) for i in range(mpm_solver.n_particles)]
        mpm_solver.mpm_state.particle_x.grad = wp.array(x_grad_initial_value, dtype=wp.vec3)
        l = wp.zeros(1, dtype=float, device="cuda:0", requires_grad=True)
        wp.launch(
            sum_vec3,
            dim = mpm_solver.n_particles,
            inputs = [mpm_solver.mpm_state.particle_x.grad, mpm_solver.mpm_state.particle_x, l],
            device="cuda:0"
        )
        # print(tape.scopes)
        tape.backward(l)
        # mpm_solver.clear_mpm_model_grad()
        # mpm_solver.clear_particle_grad()
        # mpm_solver.mpm_state.particle_x.grad = wp.array(x_grad_initial_value, dtype=wp.vec3)
        # # print(f"E grad is: {mpm_solver.E_array.grad.numpy()[0]}")
        # # print(f"particle E grad is: {mpm_solver.mpm_model.E.grad}")
        # mpm_solver.p2g2p_adjoint(dt)
        # # for frame in tqdm(range(num_frames)):
        # #     for step in range(interpolate):
        # #         mpm_solver.p2g2p_adjoint(dt)
        # print(f"mu grad is: {mpm_solver.mpm_model.mu.grad}")

        # # print(f"mu grad is: {mpm_solver.mpm_model.mu.grad}")
        # # print(f"E grad is: {mpm_solver.E_array.grad.numpy()[0]}")
        # # print(f"particle E grad is: {mpm_solver.mpm_model.E.grad}")

        # # for step in range(pre_steps):
        # #     mpm_solver.p2g2p_adjoint(dt)
        # #     print(f"mu grad is: {mpm_solver.mpm_model.mu.grad}")

        # mpm_solver.finalize_mu_lam_adjoint()

        # print(f"E grad is: {mpm_solver.E_array.grad.numpy()[0]}")
        # print(f"particle E grad is: {mpm_solver.mpm_model.E.grad}")

        # print(f"mu grad is: {mpm_solver.mpm_model.mu.grad}")
        # print(f"E grad is: {mpm_solver.E_array.grad.numpy()[0]}")
        # print(f"particle E grad is: {mpm_solver.mpm_model.E.grad}")
        if "E" in params_label:
            mpm_solver.E_array.grad.zero_()
            E_grad_np = mpm_solver.mpm_model.E.grad.numpy()
            wp.launch(
                   kernel=set_value_to_float_array_diff,
                   dim=mpm_solver.n_particles,
                   inputs=[mpm_solver.mpm_model.E, mpm_solver.E_array],
                   adjoint=True,
                   adj_inputs=[None, None],
                   device="cuda:0"
               )
            grad_tensor[0] = float(mpm_solver.E_array.grad.numpy()[0])
        # for frame in tqdm(range(num_frames)):
        #     particle_x_grad_tensor_np = grad_scale[:,:,frame].numpy()
        #     x_grad_initial_value = [wp.vec3(*particle_x_grad_tensor_np[i]) for i in range(mpm_solver.n_particles)]
        #     # zero all grad first
        #     mpm_solver.clear_mpm_model_grad()
        #     mpm_solver.clear_particle_grad()
        #     # set initial grad 
        #     mpm_solver.mpm_state.particle_x.grad = wp.array(x_grad_initial_value, dtype=wp.vec3)
        #     for _ in range(interpolate):
        #         mpm_solver.p2g2p_adjoint(dt)
            
        #     if mpm_solver.mpm_model.material != 6:
        #         mpm_solver.finalize_mu_lam_adjoint()

        #     if "E" in params_label:
        #         mpm_solver.E_array.grad.zero_()
        #         E_grad_np = mpm_solver.mpm_model.E.grad.numpy()
        #         wp.launch(
        #             kernel=set_value_to_float_array_diff,
        #             dim=mpm_solver.n_particles,
        #             inputs=[mpm_solver.mpm_model.E, mpm_solver.E_array],
        #             adjoint=True,
        #             adj_inputs=[None, None],
        #             device="cuda:0"
        #         )
        #         # print(E_grad_np.size)
        #         jacobian_dict["E"][frame, :] = E_grad_np
        #         grad_tensor[0] = grad_tensor[0] + mpm_solver.E_array.grad.numpy()[0]

        #     if "bulk_modulus" in params_label:
        #         bulk_grad_np = mpm_solver.mpm_model.bulk.grad.numpy()
        #         jacobian_dict["bulk_modulus"][frame, :] = bulk_grad_np
            
        #     if "nu" in params_label:
        #         nu_grad_np = mpm_solver.mpm_model.nu.grad.numpy()
        #         jacobian_dict["nu"][frame, :] = nu_grad_np
            
            
        # grad_tensor = grad_tensor / num_frames
        # grad_dict = {key: }
        # grad_dict = {key: torch.from_numpy(jacobian).sum(dim=0) for key, jacobian in jacobian_dict.items()}

        # dx_dp = {key: torch.from_numpy(jacobian) for key, jacobian in jacobian_dict.items()} # dict of (num_frames, wp_solver.n_particles) 
        # # grad_scale should be a tensor of shape (wp_solver.n_particles, 3, num_frames)
        
        # # Hadamard product here and sum up
        # grad_dict = {key: (dV_dx*dxdp).sum()/3.0 for key, dxdp in dx_dp.items()}
        # grad = (dV_dx*dx_dp).sum() / 3.0 # approximate weighted sum by assuming the same weight
        # if "E" in grad_dict.keys():
        #     grad_tensor[0] = grad_dict["E"]
        # if "nu" in grad_dict.keys():
        #     grad_tensor[1] = grad_dict["nu"]
        # if "bulk_modulus" in grad_dict.keys():
        #     grad_tensor[2] = grad_dict["bulk_modulus"]
        print(grad_tensor)
        return (None, grad_tensor, None, None, None, None)


def validity_check(user_input, supported, initial_values):
    error_status = 0
    difference = set(user_input) - (supported)
    if len(difference) != 0:
        error_status = 1
    elif len(difference) == 0 and len(initial_values) != len(user_input):
        error_status = 2
    return error_status, difference

class VSDS_Trainer:
    def __init__(self, trainer_config, sds_config):
        # diffusion model config
        self.num_timesteps = 50
        self.betas = beta_schedule(
            'linear_sd',
            num_timesteps=self.num_timesteps,
            init_beta=0.00085,
            last_beta=0.0120
        )
        self.sds_config = sds_config
        # self.alphas = torch.tensor([1], dtype=torch.float32) - self.betas
        # self.diffuser = GaussianDiffusion(self.betas)

        # mpm model config
        self.optimizable = {"E", "nu", "bulk_modulus"} # optimizable parameters currently supported

        self.mpm_solver_config = trainer_config['mpm_solver_config']
        self.initial_values_raw = trainer_config['initial_values']
        self.target_labels = trainer_config['target_parameters']
        self.use_pbr = trainer_config["use_pbr"]
        self.target_params = dict(zip(self.target_labels, self.initial_values_raw)) 

        
        initial_values_np = np.zeros(10, dtype=float)
        #############################################################################################
        # we use a tensor to store the gradients (dictionary is not supported in autograd.Function) #
        # 0: E                                                                                      #
        # 1: nu                                                                                     #
        # 2: bulk_modulus                                                                           #
        #############################################################################################
        if "E" in self.target_labels:
            initial_values_np[0] = self.target_params["E"]
        if "nu" in self.target_labels:
            initial_values_np[1] = self.target_params["nu"]
        if "bulk_modulus" in self.target_labels:
            initial_values_np[2] = self.target_params["bulk_modulus"]
        initial_values_tensor = torch.tensor(initial_values_np, dtype=float, requires_grad=True)

        self.initial_values_tensor = initial_values_tensor.clone().detach().requires_grad_()

        status, difference = validity_check(self.target_labels, self.optimizable, self.initial_values_raw)
        if status == 1:
            raise ValueError(f"User input {difference} are not supported for optimization!") # this will terminate the trainer
        elif status == 2:
            raise ValueError("Initial values must match target parameters in length!") # this will terminate the trainer
        
    
        # initialize mpm solver
        # self.wp_solver = MPM_Simulator_WARP(10) # initialize with whatever number is fine, it will be reinitialize
        self.ti_solver = Diff_MPM_Taichi(steps=300, n_grid=64)

        # setup diffusion utilities before training
        self.diffuser = SDSVideoDiffuser(sds_config, "cpu", reuse_pipe=False)
        # self.setup_unet(UNET_CKPT_PATH, MODEL_CFG)
        # self.setup_autoencoder(AE_CKPT_PATH, AE_CONFIG)
        # self.setup_clipencoder(CLIP_CKPT_PATH)

        
        self.tb_writer = SummaryWriter("./logs")
        self.iso_material = trainer_config["iso_material"] 

        
    def train(self, num_frames=60, max_steps=25, plot_target_value=True):
        torch.cuda.empty_cache()
        # empty_prompt = ''
        # txt_embedding = self.text_encoder.encode(text_prompt)
        # empt_embedding = self.text_encoder.encode(empty_prompt)
        # clip_device = 'cpu' if TINY_GPU==1 else 'cuda'
        # context = torch.cat([empt_embedding, txt_embedding], dim=0).to(clip_device)
        # txt_model_kwargs = [
        #     {'y': context[1].unsqueeze(0).repeat(1, 1, 1).to(clip_device)},
        #     {'y': context[0].unsqueeze(0).repeat(1, 1, 1).to(clip_device)}
        #     ]        
        
        initial_log_values = torch.log10(self.initial_values_tensor).detach().clone().requires_grad_()

        optimizer = optim.Adam([initial_log_values], lr=1e-2)
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, )
        
        target_values = [self.initial_values_tensor[0].item()]
        grad_values = []
        lr = 1e-1
        
        
        self.ti_solver = self.setup_ti_solver_from_input(self.initial_values_tensor)

        # Training loop
        for step in tqdm(range(max_steps + 1)):
            print(f"Starting epoch: {step}, {max_steps} in total")
            # print(torch.pow(10, initial_log_values))
            # learning rate adjust\
            # if (step+1) % 6 == 0:
            #     lr *= 1e-2
            ti.ad.clear_all_gradients()

            # Forward pass
            # frame_tensor = TaichiSimulator.apply(self,self.initial_values_tensor, 15)
            # frame_tensor = WarpSimulator.apply(self, 
            #                                         self.initial_values_tensor, 
            #                                         # torch.pow(10, initial_log_values),
            #                                         num_frames, 
            #                                         2e-3, 6, 120) # should be of size (N, 3, num_frames)
            
            pre_steps = 120
            self.ti_solver.forward(self.ti_solver.steps)
            frame_tensor = torch.from_numpy(self.ti_solver.x.to_numpy())
            frame_tensor[:,:,[1,2]] = frame_tensor[:,:,[2,1]]
            interpolate = (self.ti_solver.steps - pre_steps) // num_frames
            frame_tensor = frame_tensor[pre_steps:self.ti_solver.steps:interpolate,:,:].requires_grad_()

            write_video = True
            output_path = f"./results/jelly_epoch{step}.mp4"
            if step==0:
                write_video = True
                output_path = f"./results/jelly_epoch{step}.mp4"
            elif step==max_steps:
                write_video = True
                output_path = "./results/jelly_after.mp4"
            print("rendering video...")
            if self.use_pbr:
                mi.set_variant("cuda_ad_rgb")
                obj_dict = {"jelly": {"verts": frame_tensor, "material": "plastic", "scale": 4.0}}
                obj_pos = {"jelly": [0, 0, 0]}
                video = Mitsuba3RenderScene.apply(frame_tensor.permute(1,2,0), obj_dict, obj_pos, )
                if write_video:
                    video_tensor_out = (video*255.0/video.max()).byte()
                    rendered_video = video_tensor_out.cpu()
                    torchvision.io.write_video(output_path, rendered_video, fps=15)
                    print(f"successfully written video")
            else:
                video = wp_pt3d_render_point_cloud(RENDER_CONFIG, frame_tensor, device="cuda:0", output_path=output_path, fps=30, write_video=write_video)
            
            # log video to TensorBoard
            video_for_tb = video.unsqueeze(0).permute(0,1,4,2,3) # [batch_size, channels, frames, height, width]
            video_for_tb = video_for_tb.repeat(self.sds_config["batch_size"],1,1,1,1)
            self.tb_writer.add_video(f'epoch_{step}/video', video_for_tb, step, fps=15)

            x = video.unsqueeze(0).permute(0,1,4,2,3)
            x = x.repeat(self.sds_config["batch_size"],1,1,1,1)
            if self.sds_config["augment_frames"]:
                augmented_pair = get_augmentations(torch.cat([x.squeeze(0), x.squeeze(0)]))
                x_aug = augmented_pair[:self.sds_config["num_frames"]].unsqueeze(0)
            else: 
                x_aug = x

            # Compute the sds loss
            # vsds_loss = self.VSDS(video, txt_model_kwargs)
            vsds_loss = self.diffuser(x_aug)

            # log the loss to TensorBoard
            self.tb_writer.add_scalar("sds loss", vsds_loss.item(), step)

            self.tb_writer.close()
            # Backward pass
            vsds_loss.backward()  # Compute gradients   

            grad = torch.zeros_like(self.ti_solver.x.to_torch())
            grad[pre_steps:self.ti_solver.steps:interpolate,:,:] = frame_tensor.grad


            assign_value_to_vec_field(self.ti_solver.x.grad, grad)
            # print("ok")
            # print(self.ti_solver.x.grad)
            # assign grad to particle positions
            self.ti_solver.backward(self.ti_solver.steps)
            print(self.ti_solver.E.grad)
            lam = 1e3
            self.ti_solver.compute_pairwise_loss(lam)
            self.ti_solver.compute_pairwise_loss.grad(lam)
            print(self.ti_solver.E.grad)
            if not self.iso_material:
                print(f"lr: {lr}")
                learn(self.ti_solver.E, lr)
            else:
                grad_value = ti.field(ti.float32, shape=())
                self.ti_solver.compute_E_grad_sum(grad_value)
                learn_iso(self.ti_solver.E, 1e-2, grad_value[None])


            
            E_tensor = self.ti_solver.E.to_torch()

            print(torch.mean(E_tensor))
            print(torch.max(E_tensor))
            print(torch.min(E_tensor))





            # grad_values.append(abs(self.initial_values_tensor.grad[0].item()))
            # #################################################################################
            # # d(scale * loss)/d(loss) * grad * d(latents)/d(video) * d(video)/d(parameters) #
            # #   = scale * grad_scale * w * (pred_noise - noise) * d(video)/d(parameters)    #
            # #################################################################################
            # # update parameters
            # self.initial_values_tensor.grad = self.initial_values_tensor.grad
            # initial_log_values = initial_log_values - (lr*self.initial_values_tensor.grad).clamp(-0.25,0.25)
            # # ensure validity
            # # if initial_log_values[1] > 0.45:
            #     # initial_log_values[1] = 0.45
            # # optimizer.step()
            # # scheduler.step()
            # with torch.no_grad():
            #     self.initial_values_tensor.copy_(torch.pow(10, initial_log_values))
        
            # # Zero the gradients manually after updating
            # self.initial_values_tensor.grad.zero_()
            # # optimizer.zero_grad()
            # target_values.append(self.initial_values_tensor[0].item())

            # print(f"Updated value: {self.initial_values_tensor.tolist()} after epoch: {step}")

        # if plot_target_value:
        #     plt.figure(1)
        #     plt.plot(list(range(len(target_values))), target_values, marker='o')
            
        #     plt.figure(2)
        #     plt.plot(list(range(len(grad_values))), grad_values, marker='s')

        #     plt.show()


    def VSDS(
            self,
            video, # torch.Size([num_frames, H, W, 3])
            txt_model_kwargs, 
            guide_scale=200,
            grad_scale=1.
    ):
        
        # encoder into latent, grad is required
        print("Encodeing video to latent space...")
        torch.manual_seed(12)
        # switch dimensions of video to match the encoder
        video = video.permute(0, 3, 1, 2).to(AE_CONFIG['ae_device'])
        latents_distribution = self.encoder.encode(video)
        latents = latents_distribution.sample()
        
        # generate t
        t = torch.randint(
            1, self.num_timesteps, [1], dtype=torch.long, device=latents.device
        )
        

        # We do not want to calculate the derivative of pred_noise with respect to parameters
        with torch.no_grad(): 
            noise, noised_latents = self.diffuser.add_noise(latents, t) # noise: torch.Size([num_frames, 4, h, w]), noised_latents: torch.Size([num_frames, 4, h, w])
            txt_model_kwargs_gpu = [{'y': item['y'].cuda()} for item in txt_model_kwargs]
            pred_noise = self.predict_noise(noised_latents.to("cuda:0"), guide_scale, t.to("cuda:0"), txt_model_kwargs_gpu) 
        pred_noise = pred_noise.to("cpu")
    
        # w(t) == sigma_t^2 == beta_t
        w = 1 - self.alphas[t]
        grad = grad_scale * w * (pred_noise - noise) # torch.Size([num_frames, channels, h, w])
        grad = torch.nan_to_num(grad) 

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        loss = SpecifyGradient.apply(latents, grad)

        return loss
    
    def predict_noise(self, latent_video, guide_scale, t, model_kwargs):
        latent_video = latent_video.unsqueeze(0).transpose(1, 2)
        y_out = self.unet(latent_video, t, **model_kwargs[0])
        u_out = self.unet(latent_video, t, **model_kwargs[1])

        dim = y_out.size(1)
        a = u_out[:, :dim]
        b = guide_scale * (y_out[:, :dim] - u_out[:, :dim])
        c = y_out[:, dim:]
        out = torch.cat([a + b, c], dim=1)

        return out.squeeze(0).transpose(0,1)


    def setup_unet(self, ckpt_path, config=MODEL_CFG):
        print("Setting up UNet...")
        self.unet = UNetSD(
            in_dim=config['unet_in_dim'],
            dim=config['unet_dim'],
            y_dim=config['unet_y_dim'],
            context_dim=config['unet_context_dim'],
            out_dim=config['unet_out_dim'],
            dim_mult=config['unet_dim_mult'],
            num_heads=config['unet_num_heads'],
            head_dim=config['unet_head_dim'],
            num_res_blocks=config['unet_res_blocks'],
            attn_scales=config['unet_attn_scales'],
            dropout=config['unet_dropout'],
            temporal_attention=config['temporal_attention']
            )
        self.unet.load_state_dict(torch.load(ckpt_path), strict=True)
        self.unet.cpu() if config['unet_device']=='cpu' else self.unet.cuda()
        self.unet.eval()
        # unet_device = config["unet_device"]
        print("Successfully setup UNet on: "+ config['unet_device'])

    def setup_autoencoder(self, ckpt_path, config=AE_CONFIG):
        print("Setting up autoencoder...")
        self.encoder = AutoencoderKL(config, 4, ckpt_path)
        self.encoder.cpu() if config['ae_device']=='cpu' else self.encoder.cuda()
        self.encoder.eval()
        print("Successfully setup autoencoder on: " + config['ae_device'])
    
    def setup_clipencoder(self, ckpt_path):
        print("Setting up CLIP encoder...")
        device ='cpu' if TINY_GPU==1 else 'cuda'
        self.text_encoder = FrozenOpenCLIPEmbedder(version=ckpt_path, layer='penultimate', device=device)
        self.text_encoder.eval()
        print("Successffully setup CLIP encoder on: " + str(device))

    def setup_solver_from_input(self, params_tensor):
        self.wp_solver.load_from_sampling(self.mpm_solver_config['data'])
        volume_tensor = torch.ones(self.wp_solver.n_particles) * 2.5e-3
        position_tensor = self.wp_solver.export_particle_x_to_torch()
        self.wp_solver.load_initial_data_from_torch(position_tensor, volume_tensor)

        length = len(self.initial_values_raw)
        params_list = [params_tensor[i].item() for i in range(length)]
        target_params = dict(zip(self.target_labels, params_list))
        material_params = self.mpm_solver_config | target_params

        self.wp_solver.set_parameters_dict(material_params)
        
        # box_length = 0.2
        self.wp_solver.add_surface_collider((0.0, 0.0, 0.1), (0.0,0.0,1.0), 'cut', 0.0)
        # self.mpm_solver.add_surface_collider((0.5 - box_length/2, 0.0, 0.0), (1.0,0.0,0.0), 'cut', 0.0)
        # self.mpm_solver.add_surface_collider((0.5 + box_length/2, 0.0, 0.0), (-1.0,0.0,0.0), 'cut', 0.0)
        # self.mpm_solver.add_surface_collider((0.0, 0.5 + box_length/2, 0.0), (0.0,-1.0,0.0), 'cut', 0.0)
        # self.mpm_solver.add_surface_collider((0.0, 0.5 - box_length/2, 0.0), (0.0,1.0,0.0), 'cut', 0.0)

        return self.wp_solver
    
    def setup_ti_solver_from_input(self, params_tensor):
        x_data = self.mpm_solver_config['data']['x']
        particle_type = self.mpm_solver_config['data']['type']
        n_particles = x_data.shape[0]
        # always remember to allocate fields first
        self.ti_solver.allocate_fields(n_particles)
        self.ti_solver.init_from_np(x_data, particle_type)
        
        length = len(self.initial_values_raw)
        params_list = [params_tensor[i].item() for i in range(length)]
        target_params = dict(zip(self.target_labels, params_list))
        material_params = self.mpm_solver_config | target_params

        self.ti_solver.read_params_from_dict(material_params)

        return self.ti_solver