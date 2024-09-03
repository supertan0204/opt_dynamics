import torch
import numpy as np
import time
import torchvision
import mitsuba as mi
import drjit as dr
from scipy.spatial import KDTree
from tqdm import tqdm

from pytorch3d.renderer import (
    look_at_view_transform, 
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    NormWeightedCompositor,
    AlphaCompositor
    )

from pytorch3d.structures import Pointclouds
from pytorch3d.ops import estimate_pointcloud_normals
from mesh_utils import mc_from_psr, point_rasterize

from dpsr import DPSR
import open3d as o3d
import alphashape


# Here we define a function to transform PSR surface data to mesh data.
# PSR2Mesh codes are copied from https://github.com/autonomousvision/shape_as_points
class PSR2Mesh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, psr_grid):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        verts, faces, normals = mc_from_psr(psr_grid, pytorchify=True)
        verts = verts.unsqueeze(0)
        faces = faces.unsqueeze(0)
        normals = normals.unsqueeze(0)

        res = torch.tensor(psr_grid.shape[2])
        ctx.save_for_backward(verts, normals, res)

        return verts, faces, normals

    @staticmethod
    def backward(ctx, dL_dVertex, dL_dFace, dL_dNormals):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        vert_pts, normals, res = ctx.saved_tensors
        res = (res.item(), res.item(), res.item())
        # matrix multiplication between dL/dV and dV/dPSR
        # dV/dPSR = - normals
        grad_vert = torch.matmul(dL_dVertex.permute(1, 0, 2), -normals.permute(1, 2, 0))
        grad_grid = point_rasterize(vert_pts, grad_vert.permute(1, 0, 2), res) # b x 1 x res x res x res
        
        return grad_grid


RENDER_CONFIG = {
    'dist': 10,
    'elev': 30,
    'znear': 0.01,
    'size': 256,
    'radius': 0.003,
    'points_per_pixel': 10,
    'background_color': (1,1,1)
}



def ti_pt3d_render_point_cloud(RENDER_CONFIG, frame_tensor, device, rgb = [[0.1, 0.1, 0.9, 0.9]], output_path=None, fps=0, write_video=False):
    """
    point_cloud_data: dict {'verts': torch.tensor of size (N, 3, num_frames), 'rgb'}
    """

    frame_tensor.requires_grad_()

    torch.cuda.set_device(device)
    print("Start render using device:" + str(device) + "...")

    # Set up camera and renderer
    R, T = look_at_view_transform(dist=RENDER_CONFIG['dist'], elev=RENDER_CONFIG['elev'])
    cameras = FoVOrthographicCameras(device=device, R=R, T=T, znear=RENDER_CONFIG['znear'])
    raster_settings = PointsRasterizationSettings(
        image_size=RENDER_CONFIG['size'],
        radius=RENDER_CONFIG['radius'],
        points_per_pixel=RENDER_CONFIG['points_per_pixel']
    )
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor())

    # Convert data to tensors and move to device
    verts_tensor = frame_tensor.to(device)  # shape: (num_particles, 3, num_frames)
    num_particles = verts_tensor.shape[0]
    rgb_tensor = torch.tensor([[0.1, 0.1, 0.9, 0.9]] * num_particles, dtype=torch.float32).to(device)  # shape: (num_particles, 3)

    # Define bounds for normalization
    (x_max, x_min, y_max, y_min, z_max, z_min) = (6.3, 0.1, 6.3, 0.1, 6.3, 0.1)
    x_range = y_range = z_range = 6.2

    # Normalize and transform to PyTorch3D NDC
    x_ndc = 2 * (verts_tensor[:, 0, :] - x_min) / x_range - 1
    y_ndc = 2 * (verts_tensor[:, 1, :] - y_min) / y_range - 1
    z_ndc = 2 * (verts_tensor[:, 2, :] - z_min) / z_range - 1
    ndc_coords = torch.stack([x_ndc, z_ndc, y_ndc], dim=1)  # shape: (num_particles, 3, num_frames)

    # Reshape for batch processing
    ndc_coords = ndc_coords.permute(2, 0, 1)  # shape: (num_frames, num_particles, 3)
    rgb_tensor = rgb_tensor.unsqueeze(0).expand(ndc_coords.shape[0], -1, -1)  # shape: (num_frames, num_particles, 3)

    # Create point clouds for each frame in batch
    point_clouds = Pointclouds(points=ndc_coords, features=rgb_tensor)

    # Render all frames in batch
    images = renderer(point_clouds)  # shape: (num_frames, image_size, image_size, 4)

    # Process rendered images
    rendered_images_tensor = images[..., :3]
    rendered_images_tensor.requires_grad_(True)
    

    if write_video:
        assert output_path is not None
            
        assert fps != 0

        # Convert to bytes for visualization
        video_tensor_for_visualization = (rendered_images_tensor * 255.0 / rendered_images_tensor.max()).byte()
        # Convert to CPU to write to video
        rendered_video = video_tensor_for_visualization.cpu()

        torchvision.io.write_video(output_path, rendered_video, fps=fps)
        print("Successfully written to: " + output_path)

    # rendered_video_tensor = (rendered_images_tensor*255./rendered_images_tensor.max()).byte()
    return rendered_images_tensor

def wp_pt3d_render_point_cloud(render_config, verts, device, rgb=[[0.1, 0.1, 0.9, 0.9]], output_path=None, fps=15, write_video=False):
    print("Start render using device: " + str(device) + "...")

    # Set up camera and renderer
    R, T = look_at_view_transform(dist=render_config['dist'], elev=render_config['elev'])
    cameras = FoVOrthographicCameras(device=device, R=R, T=T, znear=render_config['znear'])
    raster_settings = PointsRasterizationSettings(
        image_size=render_config['size'],
        radius=render_config['radius'],
        points_per_pixel=render_config['points_per_pixel'],
        bin_size=0
    )
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(rasterizer=rasterizer, compositor=NormWeightedCompositor(background_color=render_config['background_color']))

    # Convert data to tensors and move to device
    verts = verts.permute(1,2,0)
    verts_tensor = verts.to(device)  # shape: (num_particles, 3, num_frames)
    num_particles = verts_tensor.shape[0]
    rgb_tensor = torch.tensor(rgb * num_particles, dtype=torch.float32).to(device)  # shape: (num_particles, 3)
    
    # Normalize vertex positions to NDC
    min_vals = verts_tensor.min(dim=0, keepdim=True)[0].min(dim=2, keepdim=True)[0]
    max_vals = verts_tensor.max(dim=0, keepdim=True)[0].max(dim=2, keepdim=True)[0]
    normalized = (verts_tensor - min_vals) / (max_vals - min_vals)
    ndc_coords = normalized * 2 - 1

    # Ensure we don't clip the top part of the scene
    ndc_coords = ndc_coords * 0.8
    ndc_coords[:, [1, 2], :] = ndc_coords[:, [2, 1], :]  # Switch y and z
    ndc_coords[:, 2, :] = ndc_coords[:, 2, :] * (0.1 - 1)  # Adjust z scaling to avoid clipping
    ndc_coords = ndc_coords.permute(2, 0, 1)  # shape: (num_frames, num_particles, 3)
    rgb_tensor = rgb_tensor.unsqueeze(0).expand(ndc_coords.shape[0], -1, -1)  # shape: (num_frames, num_particles, 3)

    # Create point clouds for each frame in batch
    point_clouds = Pointclouds(points=ndc_coords.float(), features=rgb_tensor.float())

    # Render all frames in batch
    images = renderer(point_clouds)  # shape: (num_frames, image_size, image_size, 4)

    # Process rendered images
    rendered_images_tensor = images[..., :3]
    rendered_images_tensor.requires_grad_(True)
    
    if write_video:
        assert output_path is not None
        assert fps != 0

        # Convert to bytes for visualization
        video_tensor_for_visualization = (rendered_images_tensor * 255.0 / rendered_images_tensor.max()).byte()
        rendered_video = video_tensor_for_visualization.cpu()
        torchvision.io.write_video(output_path, rendered_video, fps=fps)
        print("Successfully written to: " + output_path)

    return rendered_images_tensor

@dr.wrap_ad(source='torch', target='drjit')
def render_scene(scene, mesh_params_list):
    params = mi.traverse(scene)
    for obj_name, mesh_params in mesh_params_list:
        params.keep(f"{obj_name}.vertex_positions")
    # params.keep("jelly.vertex_positions")
    for _, mesh_params in mesh_params_list:
        dr.enable_grad(mesh_params["vertex_positions"])
    # dr.enable_grad(mesh_params["vertex_positions"])
    # render current frame
    return mi.render(scene, params=params)


def mitsuba_render_scene(obj_dict, obj_location, frame_res=512, dpsr_res=512, dpsr_sig=10, alpha=0.03, output_path=None, fps=15, write_video=False):
    '''
    :param dpsr_config: differentiable poisson surface reconstruction configurations
    :param obj_verts: dictionary filled with PyTorch tensor with shape (num_frames, num_particles, 3)
    :param output_path: path to output video, write_video must set to be true
    :param fps: frame per second
    :param write_video: write video or not
    example:
    >>> object_dict = {
    >>> "fluid": {"verts": fluid_verts_tensor, "material": "plastic"},
    >>> "jelly": {"verts": jelly_verts_tensor, "material": "metal"},
    >>> }
    >>> positions = {
    >>> "fluid": [0, 0, 0],  # Target position for the "fluid" object
    >>> "jelly": [1, 1, 1],  # Target position for the "jelly" object
    >>> }
    '''
    # record time
    time_start_render = time.time()
    imgs = []
    jacobian = []
    res = dpsr_res
    sig = dpsr_sig
    num_frames = next(iter(obj_dict.values()))['verts'].shape[0]
    scene_params_list = []
    for i in range(num_frames):
        print(f"rendering frame {i+1}...")
        scene_dict = {
            "type": "scene",
            "integrator": {
                "type": "prb",
                "max_depth": 8,
            },
            "light": {"type": "constant"},
            # "emitter_id":{
            #     "type": "envmap",
            #     "filename": "envmap.hdr"
            # },
            "floor_bsdf":{
                "type": "twosided",
                "material": {
                    "type": "diffuse",
                    "reflectance": {
                        "type": "checkerboard",
                        "color0": {
                            "type": "rgb",
                            "value": [0.325, 0.31, 0.25]
                        },
                        "color1": {
                            "type": "rgb",
                            "value": [0.725, 0.71, 0.68]
                        },
                    "to_uv": mi.ScalarTransform4f.scale([10.0, 10.0, 1.0])
                    }
                }
            },
            "sensor": {
                "type": "perspective",
                'fov': 45,
                "to_world": mi.ScalarTransform4f.look_at(
                    origin=[0, -4, 2], target=[0, 0, 0], up=[0, 0, 1]
                ),
                'film_id': {
                    'type': 'hdrfilm',
                    'width': frame_res,
                    'height': frame_res,
                },
                "sampler": {
                    "type": "stratified",
                    "sample_count": 81,
                },
            },
            "floor": {
                "type": "rectangle",
                "to_world": mi.ScalarTransform4f.translate([0,0,-0.5]).rotate(axis=[0,0,1], angle=20).scale(10),
                "bsdf":{
                    "type": "ref",
                    "id": "floor_bsdf"
                }
            }
        }
        mesh_params_list = []
        for obj_name, obj_data in obj_dict.items():
            frame_tensor = obj_data['verts']
            material_type = obj_data['material']
            verts = frame_tensor[i].unsqueeze(0)
            v, f, n = alpha2mesh(verts, (res, res, res), alpha, sig, 'o3d')

            material_bsdf = get_bsdf(material_type)

            props = mi.Properties()
            props["mesh_bsdf"] = material_bsdf

            mesh = mi.Mesh(
                obj_name,
                vertex_count=v.size()[1],
                face_count=f.size()[1],
                has_vertex_normals=False,
                has_vertex_texcoords=False,
                props=props
            )

            v_mi = mi.Point3f(v.squeeze().detach().numpy())
            f_mi = mi.Vector3u(f.squeeze().detach().numpy())
            n_mi = mi.Normal3f(n.squeeze().detach().numpy())

            mesh_params = mi.traverse(mesh)
            mesh_params["vertex_positions"] = dr.ravel(v_mi)
            mesh_params["faces"] = dr.ravel(f_mi)
            mesh_params["vertex_normals"] = dr.ravel(n_mi)
            mesh_params.update()

            center = mesh.bbox().center()
            position = obj_location.get(obj_name, [0, 0, 0])
            v_mi = transform_object_to_position(v_mi, center, position, scale=7.)

            mesh_params = mi.traverse(mesh)
            mesh_params["vertex_positions"] = dr.ravel(v_mi)
            mesh_params["faces"] = dr.ravel(f_mi)
            mesh_params["vertex_normals"] = dr.ravel(n_mi)
            mesh_params.update()

            scene_dict[obj_name] = mesh  
            mesh_params_list.append((obj_name, mesh_params))       
        # then render
        scene = mi.load_dict(scene_dict)
        img = render_scene(scene)
        # params = mi.traverse(scene)
        # for obj_name, mesh_params in mesh_params_list:
        #     params.keep(f"{obj_name}.vertex_positions")
        # # params.keep("jelly.vertex_positions")

        # for _, mesh_params in mesh_params_list:
        #     dr.enable_grad(mesh_params["vertex_positions"])
        # # dr.enable_grad(mesh_params["vertex_positions"])

        # scene_params_list.append(params)

        # # render current frame
        # img = mi.render(scene, params=params)

        # backward and store the jacobian
        # dr.backward(img)
        # grad = dr.grad(params)
        # jacobian.append(torch.tensor(grad["jelly.vertex_positions"].numpy()).reshape(-1,3))
        imgs.append(img)  

    imgs_converted = [torch.tensor(img.numpy()) for img in imgs]
    video_tensor = torch.stack(imgs_converted)[..., :3]
    jacobian_tensor = torch.stack(jacobian)
    if write_video:
        video_tensor_out = (video_tensor*255.0/video_tensor.max()).byte()
        rendered_video = video_tensor_out.cpu()
        torchvision.io.write_video(output_path, rendered_video, fps=fps)
        print(f"successfully written video to {output_path}")
    time_end_render = time.time()
    print(f"spend {time_end_render-time_start_render}s for rendering video with {num_frames} frames")
    return video_tensor, jacobian_tensor
    

def compute_alpha_shape(point_cloud, alpha, use_pipeline='o3d'):
    if use_pipeline=='o3d':
        points = point_cloud.cpu().squeeze(0).detach().numpy()
        pc_o3d = o3d.geometry.PointCloud()
        pc_o3d.points = o3d.utility.Vector3dVector(points)
        alpha_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pc_o3d, alpha)
        alpha_mesh.compute_vertex_normals()
        verts = torch.tensor(np.array(alpha_mesh.vertices), dtype=torch.float32).unsqueeze(0).to("cuda:0")
        normals = torch.tensor(np.array(alpha_mesh.vertex_normals), dtype=torch.float32).unsqueeze(0).to("cuda:0")
        # print("computing selection matrix")
        start = time.time()
        kd_tree = KDTree(points)
        _, indices = kd_tree.query(np.array(alpha_mesh.vertices))

        N = points.shape[0]
        n = len(indices)
        S = np.zeros((n,N))
        S[np.arange(n), indices] = 1

        S_tensor = torch.tensor(S, dtype=torch.float32)
        end = time.time()
    # this is 20x slower
    elif use_pipeline=='pt3d':
        normals = estimate_pointcloud_normals(point_cloud, 500)
    else: raise("Unsupported meshing pipeline, exiting...")
    return verts, normals, S_tensor
def alpha2mesh(verts, normals, dpsr_res, sig=15):
    '''
    dpsr_res: dpsr resolution, e.g. (128, 128, 128) 
    sig: degree of gaussian smoothing
    point_cloud: point cloud data in tensor with shape (minibatch, num_point, 3) 
    '''

    # Initialize differentiable Poisson solver
    dpsr = DPSR(dpsr_res, sig).to("cuda:0")
    # print('solving Poisson equation...')
    dpsr.eval()
    verts.requires_grad_()
    psr_grid = dpsr(verts, normals).unsqueeze(1) # Indicator function field should be (minibatch, dpsr_res, dpsr_res, dpsr_res)
    psr_grid = torch.tanh(psr_grid).to("cuda:0")

    # print("marching cube...")
    v, f, n = PSR2Mesh.apply(psr_grid)
    v = v * dpsr_res[0]/(dpsr_res[0] - 1) 

    return v, f, n

# Mitsuba 3 helper functions
def transform_object_to_position(v_mi, center, position, scale):
    # first translate object to the origin
    v_mi.x -= center[0]
    v_mi.y -= center[1]
    v_mi.z -= center[2]

    v_mi *= scale

    v_mi.x += position[0]
    v_mi.y += position[1]
    v_mi.z += position[2]

    return v_mi

def get_bsdf(material_type):
    if material_type == 'plastic':
        return mi.load_dict({
            'type': 'plastic',
            'diffuse_reflectance': {
                'type': 'rgb',
                'value': [0.5, 0.1, 0.1],
            },
            'int_ior': 1.8,
            'ext_ior': 1.0,
            'nonlinear': True,
        })
    elif material_type == 'metal':
        return mi.load_dict({
            'type': 'conductor',
            'material': 'Au'  # Example: Gold material
        })
    elif material_type == 'diffuse':
        return mi.load_dict({
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                'value': [0.5, 0.5, 0.5]
            }
        })
    # Add more materials as needed
    else:
        raise ValueError(f"Unknown material type: {material_type}")
    

class Mitsuba3RenderScene(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vertex_tensor, obj_dict, obj_location, frame_res=256, dpsr_res=512, dpsr_sig=3, alpha=0.03):
        """
        Here we pass a dummy vertex tensor with shape (num_objects, num_frames, num_vertices, 3) to satisfy the convention of self-defined Function
        This tensor could be directly obtained from obj_dict by calling .value() method and stack to PyTorch tensor
        """
        # record time
        time_start_render = time.time()
        imgs_tensor = []
        imgs = []
        res = dpsr_res
        sig = dpsr_sig
        num_frames = next(iter(obj_dict.values()))['verts'].shape[0]
        scene_params_list = []
        dl_dverts_list = []
        S_list = []

        center = None
        print("start rendering...")
        for i in tqdm(range(num_frames)):
            # print(f"rendering frame {i+1}...")
            scene_dict = {
                "type": "scene",
                "integrator": {
                    "type": "prb",
                    "max_depth": 8,
                },
                "light": {"type": "constant"},
                # "emitter_id":{
                    # "type": "envmap",
                    # "filename": "envmap.hdr"
                # },
                "floor_bsdf":{
                    "type": "twosided",
                    "material": {
                        "type": "diffuse",
                        "reflectance": {
                            "type": "checkerboard",
                            "color0": {
                                "type": "rgb",
                                "value": [0.325, 0.31, 0.25]
                            },
                            "color1": {
                                "type": "rgb",
                                "value": [0.725, 0.71, 0.68]
                            },
                        "to_uv": mi.ScalarTransform4f.scale([10.0, 10.0, 1.0])
                        }
                    }
                },
                "sensor": {
                    "type": "perspective",
                    'fov': 45,
                    "to_world": mi.ScalarTransform4f.look_at(
                        origin=[0, -4, 2], target=[0, 0, 0], up=[0, 0, 1]
                    ),
                    'film_id': {
                        'type': 'hdrfilm',
                        'width': frame_res,
                        'height': frame_res,
                    },
                    "sampler": {
                        "type": "stratified",
                        "sample_count": 81,
                    },
                },
                # "floor": {
                #     "type": "rectangle",
                #     "to_world": mi.ScalarTransform4f.translate([0,0,-2]).rotate(axis=[0,0,1], angle=-10).scale(30),
                #     "bsdf":{
                #         "type": "ref",
                #         "id": "floor_bsdf"
                #     }
                # }
            }

            mesh_params_list = []
            dl_dverts_dict = {}
            S_dict = {}
            for obj_name, obj_data in obj_dict.items():
                frame_tensor = obj_data['verts']
                material_type = obj_data['material']
                scale = obj_data["scale"]
                points = frame_tensor[i].unsqueeze(0)
                points.requires_grad_()
                verts, normals, S = compute_alpha_shape(points, alpha, 'o3d')

                S_dict[obj_name] = S
                verts.requires_grad_()
                # need to enable grad computation here
                torch.set_grad_enabled(True)
                v, f, n = alpha2mesh(verts, normals, (res, res, res), sig)

                # print("calculating dpsr grad... \n(this slows the forward rendering process for a little bit, but we do not need to compute vertex grad in backward again)")
                material_bsdf = get_bsdf(material_type)

                props = mi.Properties()
                props["mesh_bsdf"] = material_bsdf

                mesh = mi.Mesh(
                    obj_name,
                    vertex_count=v.size()[1],
                    face_count=f.size()[1],
                    has_vertex_normals=False,
                    has_vertex_texcoords=False,
                    props=props
                )
                v_cpu = v.cpu()
                f_cpu = f.cpu()
                n_cpu = n.cpu()                                
                v_mi = mi.Point3f(v_cpu.squeeze().detach().numpy())
                f_mi = mi.Vector3u(f_cpu.squeeze().detach().numpy())
                n_mi = mi.Normal3f(n_cpu.squeeze().detach().numpy())

                mesh_params = mi.traverse(mesh)
                mesh_params["vertex_positions"] = dr.ravel(v_mi)
                mesh_params["faces"] = dr.ravel(f_mi)
                mesh_params["vertex_normals"] = dr.ravel(n_mi)
                mesh_params.update()

                if i == 0:
                    center = mesh.bbox().center()
                position = obj_location.get(obj_name, [0, 0, 0])
                T = mi.Transform4f.translate([position[0],position[1],position[2]]).scale(scale).translate([-center[0],-center[1],-center[2]])
                v_after_trans = T@v_mi
                mesh_params["vertex_positions"] = dr.ravel(v_after_trans)
                mesh_params["faces"] = dr.ravel(f_mi)
                mesh_params["vertex_normals"] = dr.ravel(n_mi)
                mesh_params.update()

                scene_dict[obj_name] = mesh 
                mesh_params_list.append((obj_name, mesh_params)) 
            # then render
            scene = mi.load_dict(scene_dict)
            params = mi.traverse(scene)
            for obj_name, mesh_params in mesh_params_list:
                params.keep(f"{obj_name}.vertex_positions")

            # for _, mesh_params in mesh_params_list:
            #     dr.enable_grad(mesh_params["vertex_positions"])

            dr.enable_grad(mesh_params["vertex_positions"])
            scene_params_list.append(params)

            # render current frame
            img = mi.render(scene, params=params)
            dr.backward(img)
            grad = dr.grad(params)
            # rememver to multiply the scale because of the object transformation
            v_grad = {obj: scale*torch.tensor(value.numpy(),dtype=torch.float32).view(-1,3).to("cuda:0") for obj, value in grad.items()}
            dl_dverts_dict = {obj: torch.autograd.grad(v, verts, value.unsqueeze(0))[0].squeeze(0).to("cuda:0") for obj, value in v_grad.items()}

            dl_dverts_list.append(dl_dverts_dict)
            S_list.append(S_dict)

            imgs_tensor.append(torch.tensor(img.numpy()))
            imgs.append(img)

        video_tensor = torch.stack(imgs_tensor)[..., :3]
        ctx.imgs = imgs
        ctx.mesh_params_list = mesh_params_list
        ctx.scene_params_list = scene_params_list
        ctx.dl_dverts_list = dl_dverts_list
        ctx.S_list = S_list


        time_end_render = time.time()
        print(f"spend {time_end_render-time_start_render}s for rendering video with {num_frames} frames")
        return video_tensor.to("cuda:0")
    
    @staticmethod
    def backward(ctx, grad_output):
        imgs = ctx.imgs
        mesh_params_list = ctx.mesh_params_list
        dl_dverts_list = ctx.dl_dverts_list
        S_list = ctx.S_list

        num_frames = len(imgs)
        jacobian_dict = {}
        for obj_name, obj_params in mesh_params_list:
            jacobian = []
            for i in range(num_frames):
                dv_dverts = dl_dverts_list[i][f"{obj_name}.vertex_positions"]
                S = S_list[i][obj_name].to("cuda:0")
                grad = torch.matmul(S.t(), dv_dverts)
                jacobian.append(grad)
            jacobian_dict[obj_name] = torch.stack(jacobian)
        
        jacobian_list = list(jacobian_dict.values())
        vertex_tensor_gradient = torch.stack(jacobian_list).squeeze(0)
        return vertex_tensor_gradient.permute(1,2,0).cpu(), None, None, None, None, None, None