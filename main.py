from VSDS_Trainer import VSDS_Trainer
import torch
import numpy as np
import argparse
from pytorch_lightning import seed_everything


seed_everything(42)

# read from a mesh object
# mesh = o3d.io.read_triangle_mesh("bunny.ply")
# pcd = mesh.sample_points_poisson_disk(20000)
# o3d.visualization.draw_geometries([pcd])

n_particles = 10000
x_np = np.random.uniform(0.4, 0.6, (n_particles, 3)).astype(np.float32)
particle_type = torch.ones(n_particles,dtype=torch.int32).numpy()
cubic_data = {"x": x_np, "type":particle_type}
# training config
# trainer_config_wp = {
#     "target_parameters": ['E'],
#     "initial_values": [0.8],
#     "mpm_solver_config": {
#         "material": "jelly",
#         'friction_angle': 35,
#         'g': [0.0, 0.0, -4.0],
#         'nu': 0.2,
#         'data': "sand_column.h5",
#         "density": 450.0
#     },
#     "use_pbr": False
# }
trainer_config_ti = {
    "target_parameters": ['E'],
    "initial_values": [0.8],
    "mpm_solver_config": {
        'g': 9.8,
        'nu': 0.2,
        "vol": 1,
        'data': cubic_data,
    },
    "use_pbr": True,
    "iso_material": False
}
txt_prompt_jelly = 'a box bouncing up from ground'

sds_config = {
    "use_xformers": False,
    "del_text_encoders": False,
    "model_name": "damo-vilab/text-to-video-ms-1.7b",
    "batch_size": 1,
    "same_noise_for_frames": False,
    "sds_timestep_low": 50,
    "timesteps": 1000,
    "guidance_scale": 50,
    "caption": txt_prompt_jelly,
    "augment_frames": False
}


trainer = VSDS_Trainer(trainer_config_ti, sds_config)
trainer.train(num_frames=15, max_steps=50)
