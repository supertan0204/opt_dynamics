from diffusers import DiffusionPipeline
import torch.nn as nn
import torch
from torch.cuda.amp import custom_bwd, custom_fwd

import einops

from torch.nn import functional as nnf
from torchvision import transforms, io


def get_augmentations():
    augemntations = []
    augemntations.append(transforms.RandomPerspective(
        fill=1, p=1.0, distortion_scale=0.5))
    augemntations.append(transforms.RandomResizedCrop(
        size=(256,256), scale=(0.4, 1), ratio=(1.0, 1.0)))
    augment_trans = transforms.Compose(augemntations)
    return augment_trans

data_augs = get_augmentations()

# =============================================
# ===== Helper function for SDS gradients =====
# =============================================
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
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None


# ========================================================
# ===== Basic class to extend with SDS loss variants =====
# ========================================================
class SDSDiffuserBase(nn.Module):

    _global_pipe = None

    def __init__(self, cfg, device, reuse_pipe=True):
        super(SDSDiffuserBase, self).__init__()

        self.cfg = cfg
        self.device = device

        # initiate a diffusion pipeline if we don't already have one / don't want to reuse it for both paths
        self.maybe_init_pipe(reuse_pipe) 

        self.alphas = self.pipe.scheduler.alphas_cumprod.to(self.device)
        self.sigmas = (1 - self.pipe.scheduler.alphas_cumprod).to(self.device)

        if cfg["use_xformers"]:
            self.pipe.enable_xformers_memory_efficient_attention()

        self.text_embeddings = self.embed_text(self.cfg["caption"])

        if self.cfg["del_text_encoders"]:
            del self.pipe.tokenizer
            del self.pipe.text_encoder

    def maybe_init_pipe(self, reuse_pipe):
        if reuse_pipe:
            if SDSDiffuserBase._global_pipe is None:
                SDSDiffuserBase._global_pipe = DiffusionPipeline.from_pretrained(self.cfg["model_name"], torch_dtype=torch.float16, variant="fp16")
                SDSDiffuserBase._global_pipe = SDSDiffuserBase._global_pipe.to(self.device)
            self.pipe = SDSDiffuserBase._global_pipe
        else:
            if self.device=="cpu":
                self.pipe = DiffusionPipeline.from_pretrained(self.cfg["model_name"], torch_dtype=torch.float32, variant="fp16")
            else: # use float 32 for cpu 
                self.pipe = DiffusionPipeline.from_pretrained(self.cfg["model_name"], torch_dtype=torch.float16, variant="fp16")
                
            self.pipe = self.pipe.to(self.device)

    def embed_text(self, caption):
        # tokenizer and embed text
        text_input = self.pipe.tokenizer(caption, padding="max_length",
                                         max_length=self.pipe.tokenizer.model_max_length,
                                         truncation=True, return_tensors="pt")
        uncond_input = self.pipe.tokenizer([""], padding="max_length",
                                         max_length=text_input.input_ids.shape[-1],
                                         return_tensors="pt")
        with torch.no_grad():
            text_embeddings = self.pipe.text_encoder(text_input.input_ids.to(self.device))[0]
            uncond_embeddings = self.pipe.text_encoder(uncond_input.input_ids.to(self.device))[0]
            
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        text_embeddings = text_embeddings.repeat_interleave(self.cfg["batch_size"], 0)

        return text_embeddings

        
    def prepare_latents(self, x_aug):
        x = x_aug * 2. - 1. # encode rendered image, values should be in [-1, 1]
        
        if self.device=="cpu":
            with torch.cuda.amp.autocast(False):
                batch_size, num_frames, channels, height, width = x.shape # [1, K, 3, 256, 256], for K frames
                x = x.reshape(batch_size * num_frames, channels, height, width) # [K, 3, 256, 256], for the VAE encoder
                init_latent_z = (self.pipe.vae.encode(x.to("cpu")).latent_dist.sample()) # [K, 4, 32, 32]
                frames, channel, h_, w_ = init_latent_z.shape
                init_latent_z = init_latent_z[None, :].reshape(batch_size, num_frames, channel, h_, w_).permute(0, 2, 1, 3, 4) # [1, 4, K, 32, 32] for the video model
        
        else:
            with torch.cuda.amp.autocast():
                batch_size, num_frames, channels, height, width = x.shape # [1, K, 3, 256, 256], for K frames
                x = x.reshape(batch_size * num_frames, channels, height, width) # [K, 3, 256, 256], for the VAE encoder
                init_latent_z = (self.pipe.vae.encode(x).latent_dist.sample()) # [K, 4, 32, 32]
                frames, channel, h_, w_ = init_latent_z.shape
                init_latent_z = init_latent_z[None, :].reshape(batch_size, num_frames, channel, h_, w_).permute(0, 2, 1, 3, 4) # [1, 4, K, 32, 32] for the video model
        latent_z = self.pipe.vae.config.scaling_factor * init_latent_z  # scaling_factor * init_latents

        return latent_z

    def add_noise_to_latents(self, latent_z, timestep, return_noise=True, eps=None):
        
        # sample noise if not given some as an input
        if eps is None:
            if self.cfg["same_noise_for_frames"]: # This works badly. Do not use.
                eps = torch.randn_like(latent_z[:, :, 0, :, :]) # create noise for single frame
                eps = einops.repeat(eps, 'b c h w -> b c f h w', f=latent_z.shape[2])
            else:
                eps = torch.randn_like(latent_z)

        # zt = alpha_t * latent_z + sigma_t * eps
        noised_latent_zt = self.pipe.scheduler.add_noise(latent_z, eps, timestep)

        if return_noise:
            return noised_latent_zt, eps

        return noised_latent_zt
    
    # overload this if inheriting for VSD etc.
    def get_sds_eps_to_subract(self, eps_orig, z_in, timestep_in):
        return eps_orig

    def drop_nans(self, grads):
        assert torch.isfinite(grads).all()
        return torch.nan_to_num(grads.detach().float(), 0.0, 0.0, 0.0)

    def get_grad_weights(self, timestep):
        return (1 - self.alphas[timestep])

    def sds_grads(self, latent_z, **sds_kwargs):

        with torch.no_grad():
            # sample timesteps
            timestep = torch.randint(
                low=self.cfg["sds_timestep_low"],
                high=min(950, self.cfg["timesteps"]) - 1,  # avoid highest timestep | diffusion.timesteps=1000
                size=(latent_z.shape[0],),
                device=self.device, dtype=torch.long)

            # add noise
            noised_latent_zt, eps = self.add_noise_to_latents(latent_z, timestep, return_noise=True)

            # denoise
            z_in = torch.cat([noised_latent_zt] * 2)  # expand latents for classifier free guidance
            timestep_in = torch.cat([timestep] * 2)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                eps_t_uncond, eps_t = self.pipe.unet(z_in, timestep_in, encoder_hidden_states=self.text_embeddings).sample.float().chunk(2)
            
            eps_t = eps_t_uncond + self.cfg["guidance_scale"] * (eps_t - eps_t_uncond)

            eps_to_subtract = self.get_sds_eps_to_subract(eps, z_in, timestep_in, **sds_kwargs)

            w = self.get_grad_weights(timestep)
            grad_z = w * (eps_t - eps_to_subtract)

            grad_z = self.drop_nans(grad_z)

        return grad_z


# =======================================
# =========== Basic SDS loss  ===========
# =======================================
class SDSVideoDiffuser(SDSDiffuserBase):
    def __init__(self, cfg, device, reuse_pipe=True):
        super(SDSVideoDiffuser, self).__init__(cfg, device, reuse_pipe=reuse_pipe)

    def forward(self, x_aug, grad_scale=1.0):
        latent_z = self.prepare_latents(x_aug)

        grad_z = grad_scale * self.sds_grads(latent_z)

        # this loss formulation is equivalent to the one below, but is more intuitive
        targets = (latent_z - grad_z).detach()
        sds_loss = 0.5 * nnf.mse_loss(latent_z.float(), targets, reduction='sum') / latent_z.shape[0]

        # sds_loss = SpecifyGradient.apply(latent_z, grad_z)

        return sds_loss    

if __name__ == "__main__":
    sds_config = {
    "use_xformers": False,
    "del_text_encoders": False,
    "model_name": "damo-vilab/text-to-video-ms-1.7b",
    "batch_size": 1,
    "same_noise_for_frames": False,
    "sds_timestep_low": 50,
    "timesteps": 1000,
    "guidance_scale": 50,
    "caption": "",
    "augment_frames": False
}
    sds = SDSVideoDiffuser(sds_config, device="cuda:0", )
    print("successfully initialized...")
    video = io.read_video("./results/jelly_epoch9.mp4", output_format="TCHW")[0]
    video = video.to(torch.float32) / 255.0 # b, K, 3, H, W
    x = video.unsqueeze(0).permute(0,1,2,3,4).to("cuda:0") #[batch_size, channels, frames, height, width]
    sds_loss = sds(x)
    print(sds_loss)
    print("ok")