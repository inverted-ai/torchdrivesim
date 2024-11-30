"""
Base class for renderers producing rasterized birdview images from given background meshes and agent positions.
"""
import abc
import dataclasses
import pickle
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import logging

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from torchdrivesim.mesh import BirdviewMesh, RGBMesh, BaseMesh, rendering_mesh, generate_disc_mesh, tensor_color, set_colors_with_defaults
from torchdrivesim.traffic_controls import BaseTrafficControl
from torchdrivesim.utils import Resolution, rotate

logger = logging.getLogger(__name__)


@dataclass
class RendererConfig:
    """
    Determines behavior of the renderer.
    Subclasses determine renderer class used.
    """
    backend: str = 'default'
    render_agent_direction: bool = True
    left_handed_coordinates: bool = False
    highlight_ego_vehicle: bool = False
    shift_mesh_by_camera_before_rendering: bool = True


@dataclass
class DummyRendererConfig(RendererConfig):
    """
    For DummyRenderer.
    """
    backend: str = 'dummy'


class Cameras:
    """
    Lightweight version of pytorch3d.renderer.FoVOrthographicCameras.
    Used to define an interface that works without pytorch3d.
    """
    def __init__(self, xy: Tensor, sc: Tensor, scale: float):
        self.xy = xy
        self.sc = sc
        self.scale = scale

        world_to_view_transform = self.get_world_to_view_transform()
        view_to_proj_transform = self.get_view_to_proj_transform()
        self.world_to_clip_transform = world_to_view_transform @ view_to_proj_transform

    def get_camera_center(self) -> Tensor:
        return self.xy

    def get_world_to_view_transform(self) -> Tensor:
        camera_xy = self.xy
        camera_sin = self.sc[..., 0]
        camera_cos = self.sc[..., 1]
        batch_size = camera_xy.shape[0]

        rotation_matrix = torch.stack([
            torch.stack([camera_cos, - camera_sin], dim=-1),
            torch.stack([camera_sin, camera_cos], dim=-1),
        ], dim=-2)

        translation = torch.eye(4, dtype=camera_xy.dtype, device=camera_xy.device)
        translation = translation.unsqueeze(0).expand(batch_size, 4, 4).contiguous()
        translation[..., 3, :2] = - camera_xy

        rotation = torch.eye(4, dtype=camera_xy.dtype, device=camera_xy.device)
        rotation = rotation.unsqueeze(0).expand(batch_size, 4, 4).contiguous()
        rotation[..., :2, :2] = rotation_matrix

        world_to_view_transform = translation @ rotation
        return world_to_view_transform

    def get_view_to_proj_transform(self) -> Tensor:
        view_to_proj_transform = torch.zeros(1, 4, 4, device=self.xy.device)
        view_to_proj_transform[:, 0, 0] = -self.scale
        view_to_proj_transform[:, 1, 1] = -self.scale
        view_to_proj_transform[:, 3, 3] = 1.0

        # NOTE: This maps the z coordinate to the range [0, 1] and replaces the
        # the OpenGL z normalization to [-1, 1]
        z_sign = +1.0
        zfar, znear = 100.0, 1.0 # This sets the max and min z planes that will be visible
        view_to_proj_transform[:, 2, 2] = z_sign * (1.0 / (zfar - znear))
        view_to_proj_transform[:, 2, 3] = -znear / (zfar - znear)
        view_to_proj_transform = view_to_proj_transform.transpose(1, 2).contiguous()
        return view_to_proj_transform

    def project_world_to_clip_space(self, points: Tensor) -> Tensor:
        return F.pad(points, (0, 1), value=1.0) @ self.world_to_clip_transform

    def transform_points_screen(self, points: Tensor, res: Resolution) -> Tensor:
        rot_mat = torch.stack([
            self.sc.flip(dims=[-1]),
            self.sc * torch.tensor([-1, 1], device=points.device)
        ], dim=-2)

        # the operations below could be fused for efficiency
        points = points - self.xy.unsqueeze(1)
        points = torch.matmul(rot_mat.unsqueeze(1), points.unsqueeze(-1)).squeeze(-1)
        points = - points * self.scale
        points = points * min(res.height, res.width) / 2
        points = points + torch.tensor([res.width, res.height], device=points.device) / 2

        return points

    def reverse_transform_points_screen(self, points: Tensor, res: Resolution) -> Tensor:
        rot_mat = torch.stack([
            self.sc.flip(dims=[-1]),
            self.sc * torch.tensor([-1, 1], device=points.device)
        ], dim=-2)

        # the operations below could be fused for efficiency
        points = points - torch.tensor([res.width, res.height], device=points.device) / 2
        points = points / (min(res.height, res.width) / 2)
        points = - points / self.scale
        points = torch.matmul(rot_mat.unsqueeze(1).transpose(-1, -2), points.unsqueeze(-1)).squeeze(-1)
        points = points + self.xy.unsqueeze(1)

        return points


class BirdviewRenderer(abc.ABC):
    """
    A renderer producing simple 2D birdview images based on static background meshes and rectangular agents.
    Currently only square resolutions are supported. The renderer always operates in batch mode,
    with a single batch dimension on the left.

    Args:
        cfg: configuration object, usually subclassed
        color_map: a dictionary of RGB tuples in 0-255 range specifying colors of different rendered elements
        res: default resolution
    """
    def __init__(self, cfg: RendererConfig, color_map: Optional[Dict[str, Tuple[int, int, int]]] = None,
                 rendering_levels: Optional[Dict[str, float]] = None,
                 res: Resolution = Resolution(64, 64), fov: float = 35):
        self.cfg: RendererConfig = cfg
        self.res = res
        self.scale = 2.0 / fov

        self.color_map = color_map
        if self.color_map is None:
            self.color_map = get_default_color_map()
        self.rendering_levels = rendering_levels
        if self.rendering_levels is None:
            self.rendering_levels = get_default_rendering_levels()

    def get_color(self, element_type: str) -> Tuple[int, int, int]:
        return self.color_map[element_type]

    # def add_static_meshes(self, meshes: List[BirdviewMesh]) -> None:
    #     """
    #     Includes additional static elements to render.
    #     """
    #     self.static_mesh = self.static_mesh.concat(
    #         [self.static_mesh] + meshes
    #     )

    # def render_static_meshes(self, camera_xy: Optional[Tensor] = None, camera_sc: Optional[Tensor] = None,
    #                          res: Resolution = None, fov: float = None) -> Tensor:
    #     """
    #     Render a single birdview of the static mesh only. Nc is the number of cameras.
    #     C=3 is the number of RGB channels.

    #     Args:
    #         camera_xy: Ncx2 tensor of camera positions
    #         camera_sc: Ncx2 tensor of camera orientations (sine and cosine of yaw angle)
    #         res: Resolution, currently only square resolutions are supported
    #         fov: Field of view in meters

    #     Returns:
    #         birdview image tensor of shape NcxHxWxC
    #     """
    #     if camera_xy is None:
    #         camera_xy = self.world_center
    #         camera_psi = torch.ones_like(camera_xy[..., :1]) * np.pi / 2
    #         camera_sc = torch.cat([torch.sin(camera_psi), torch.cos(camera_psi)], dim=-1)
    #     scale = (2.0 / fov) if fov is not None else self.scale
    #     cameras = self.construct_cameras(camera_xy.reshape(-1, 2), camera_sc.reshape(-1, 2), scale=scale)
    #     image = self.render_mesh(self.static_mesh, res, cameras)
    #     return image

    def render_frame(self, rgb_mesh: RGBMesh, camera_xy: Tensor, camera_sc: Tensor,
                     res: Optional[Resolution] = None, fov: Optional[float] = None):
        """
        Renders a given rgb mesh according the camera position and rotation.

        Args:
            rgb_mesh: the given rgb mesh to render which should be already expanded to match the number of cameras.
            camera_xy: BxNcx2 tensor of camera positions, by default one camera placed on each agent
            camera_sc: BxNcx2 tensor of camera orientations (sine and cosine), by default matching agent orientations
            res: resolution HxW of the resulting image, currently only square resolutions are supported
            fov: Field of view in meters

        Returns:
            tensor image of float RGB values in [0,255] range with shape shape (B*Nc)xCxHxW
        """
        scale = (2.0 / fov) if fov is not None else self.scale
        n_cameras_per_batch = camera_xy.shape[-2]
        camera_xy, camera_sc = camera_xy.reshape(-1, 2), camera_sc.reshape(-1, 2)
        cameras = self.construct_cameras(camera_xy, camera_sc, scale=scale)
        if res is None:
            res = self.res

        image = self.render_rgb_mesh(rgb_mesh, res, cameras)
        try:
            image = self.render_rgb_mesh(rgb_mesh, res, cameras)
        except RuntimeError as e:
            logger.exception(e)
            image = torch.zeros((batch_size * n_cameras_per_batch, res.height, res.width, 3), device=mesh.verts.device)
            try:
                # save the problematic mesh for debugging purposes
                with open('bad-mesh.pkl', 'wb') as f:
                    pickle.dump((mesh.verts.detach().cpu(), mesh.faces.detach().cpu()), f)
            except RuntimeError:
                pass
        image = image.reshape(-1, res.height, res.width, 3)
        image = image.permute(0, 3, 1, 2)
        return image


    # def render_frame(
    #     self, agent_state: Tensor, agent_attributes: Tensor,
    #     camera_xy: Optional[Tensor] = None, camera_sc: Optional[Tensor] = None,
    #     rendering_mask: Optional[Tensor] = None, res: Optional[Resolution] = None,
    #     traffic_controls: Optional[Dict[str, BaseTrafficControl]] = None, fov: Optional[float] = None,
    #     waypoints: Optional[Tensor] = None, waypoints_rendering_mask: Optional[Tensor] = None,
    #     agent_types: Optional[Tensor] = None, agent_type_names: Optional[List[str]] = None,
    #     custom_agent_colors: Optional[Tensor] = None,
    # ) -> Tensor:
    #     """
    #     Renders the agents and traffic controls on top of the static mesh.
    #     Cameras batch size is (B*Nc), which corresponds to using Nc cameras per batch element.
    #     This extra dimension is added on the right of batch dimension and flattened,
    #     to match the semantics of `extend` from pytorch3d.
    #     If cameras is None, one egocentric camera per agent is used, that is Nc = A.

    #     Args:
    #         agent_state: maps agent types to state tensors of shape BxAxSt, where St >= 3 and the first three
    #             components are x coordinate, y coordinate, and orientation in radians
    #         agent_attributes: maps agent types to static attributes tensors of shape BxAxAttr, where Attr >= 2
    #             and the first two components are length and width of the agent
    #         camera_xy: BxNcx2 tensor of camera positions, by default one camera placed on each agent
    #         camera_sc: BxNcx2 tensor of camera orientations (sine and cosine), by default matching agent orientations
    #         rendering_mask:  BxNcxA tensor per agent type, indicating which cameras see which agents
    #         res: resolution HxW of the resulting image, currently only square resolutions are supported
    #         traffic_controls: traffic controls by type (traffic_light, yield, etc.)
    #         fov: Field of view in meters
    #         waypoints: BxNcxMx2 tensor of `M` waypoints per camera (x,y)
    #         waypoints_rendering_mask: BxNcxM tensor of `M` waypoint masks per camera,
    #             indicating which waypoints should be rendered
    #         agent_types: a tensor of BxA long tensors indicating the agent type index for each agent
    #         agent_type_names: a list of agent type names to index into
    #         custom_agent_colors: a BxNcxAx3 tensor of specifying what color each agent is to what camera

    #     Returns:
    #         tensor image of float RGB values in [0,255] range with shape shape (B*Nc)xAxCxHxW
    #     """
    #     batch_size = agent_state.shape[0]
    #     scale = (2.0 / fov) if fov is not None else self.scale
    #     agent_count = agent_state.shape[-2]
    #     if agent_types is None:
    #         agent_types = torch.zeros_like(agent_attributes[..., 0], dtype=torch.long)
    #     if agent_type_names is None:
    #         agent_type_names = ['vehicle']
    #     if camera_xy is None:
    #         xy = agent_state[..., :2]
    #         psi = agent_state[..., 2:3]
    #         sc = torch.cat([torch.sin(psi), torch.cos(psi)], dim=-1)
    #         n_cameras_per_batch = agent_count
    #         # Set orthographic camera on agents that are being predicted
    #         cameras = self.construct_cameras(
    #             # put agent dimension first to easier extend the mesh tensor
    #             xy.transpose(0, 1).reshape(-1, 2),
    #             sc.transpose(0, 1).reshape(-1, 2),
    #             scale=scale)
    #     else:
    #         n_cameras_per_batch = camera_xy.shape[-2]
    #         camera_xy, camera_sc = camera_xy.reshape(-1, 2), camera_sc.reshape(-1, 2)
    #         cameras = self.construct_cameras(camera_xy, camera_sc, scale=scale)

    #     static_mesh = self.static_mesh
    #     if self.cfg.highlight_ego_vehicle:
    #         if 'ego' not in agent_type_names:
    #             agent_type_names.append('ego')
    #         agent_types[..., 0] = agent_type_names.index('ego')
    #     actor_mesh = self.make_actor_mesh(agent_state, agent_attributes,
    #                                       agent_types, agent_type_names)

    #     actor_mesh = actor_mesh.expand(n_cameras_per_batch)
    #     static_mesh = static_mesh.expand(n_cameras_per_batch)

    #     if rendering_mask is not None:
    #         rendering_mask = rendering_mask.flatten(0, 1)
    #         mask_agents = lambda x: x * rendering_mask.repeat_interleave(
    #             x.shape[1] // agent_count, dim=-1
    #         ).unsqueeze(-1).expand_as(x)
    #         actor_mesh = dataclasses.replace(
    #             actor_mesh, faces=mask_agents(actor_mesh.faces)
    #         )

    #     if custom_agent_colors is not None:
    #         static_mesh = set_colors_with_defaults(static_mesh, self.color_map, self.rendering_levels)
    #         actor_mesh = set_colors_with_defaults(actor_mesh, self.color_map, self.rendering_levels)
    #         av = 4
    #         dv = 3 if self.cfg.render_agent_direction else 0
    #         step = av + dv
    #         for i in range(av):
    #             actor_mesh.attrs[:,i::step] = custom_agent_colors

    #     meshes = [
    #         static_mesh,
    #         actor_mesh,
    #     ]

    #     if traffic_controls is not None:
    #         traffic_controls = {k: v.extend(n_cameras_per_batch) for k, v in traffic_controls.items()}
    #         controls_mesh = self.make_traffic_controls_mesh(traffic_controls).to(self.device)
    #         if custom_agent_colors is not None:
    #             controls_mesh = set_colors_with_defaults(controls_mesh, self.color_map, self.rendering_levels)
    #         meshes.append(controls_mesh)

    #     if waypoints is not None:
    #         if waypoints.shape[1] != n_cameras_per_batch:
    #             raise ValueError((f"The given waypoints ({waypoints.shape[1]} do not match "
    #                 f"the number of cameras ({n_cameras_per_batch})."))
    #         n_waypoints = waypoints.shape[-2]
    #         waypoints_mesh = self.make_waypoint_mesh(waypoints, radius=2.0, num_triangles=10)
    #         if waypoints_rendering_mask is not None:
    #             waypoints_faces = waypoints_mesh.faces
    #             waypoints_mask = waypoints_rendering_mask.reshape(-1, n_waypoints, 1, 1).expand(-1, -1, 10, 3)
    #             waypoints_faces = waypoints_faces * waypoints_mask.reshape(-1, n_waypoints*10, 3)
    #             waypoints_mesh = dataclasses.replace(
    #                 waypoints_mesh, faces=waypoints_faces
    #             )
    #         if custom_agent_colors is not None:
    #             waypoints_mesh = set_colors_with_defaults(waypoints_mesh, self.color_map, self.rendering_levels)
    #         meshes.append(waypoints_mesh)

    #     mesh = static_mesh.concat(meshes)

    #     if res is None:
    #         res = self.res

    #     try:
    #         if custom_agent_colors is None:
    #             image = self.render_mesh(mesh, res, cameras)
    #         else:
    #             image = self.render_rgb_mesh(mesh, res, cameras)
    #     except RuntimeError as e:
    #         logger.exception(e)
    #         image = torch.zeros((batch_size * n_cameras_per_batch, res.height, res.width, 3), device=mesh.verts.device)
    #         try:
    #             # save the problematic mesh for debugging purposes
    #             with open('bad-mesh.pkl', 'wb') as f:
    #                 pickle.dump((mesh.verts.detach().cpu(), mesh.faces.detach().cpu()), f)
    #         except RuntimeError:
    #             pass

    #     # recover informative shape
    #     if camera_xy is None:
    #         image = image.reshape(agent_count, -1, res.height, res.width, 3)
    #         image = image.permute(1, 0, 4, 2, 3)
    #     else:
    #         image = image.reshape(-1, res.height, res.width, 3)
    #         image = image.permute(0, 3, 1, 2)
    #     if camera_xy is None:
    #         # recover original batch shape
    #         image = image.reshape((batch_size, *image.shape[1:]))
    #     return image


    @abc.abstractmethod
    def render_rgb_mesh(self, mesh: RGBMesh, res: Resolution, cameras: Cameras)\
            -> Tensor:
        """
        Renders a given mesh, producing BxHxWxC tensor image of float RGB values in [0,255] range.
        """
        pass

    def construct_cameras(self, xy: Tensor, sc: Tensor, scale: Optional[float] = None) -> Cameras:
        """
        Create PyTorch3D cameras object for given positions and orientations.
        Input tensor dimensions should be Bx2.
        """
        scale = self.scale if scale is None else scale
        return Cameras(xy=xy, sc=sc, scale=scale)


class DummyRenderer(BirdviewRenderer):
    """
    Produces a black image of the required size. Mostly used for debugging and benchmarking.
    """
    def render_rgb_mesh(self, mesh: RGBMesh, res: Resolution, cameras: Cameras) -> Tensor:
        camera_batch_size = cameras.get_camera_center().shape[0]
        shape = (camera_batch_size, res.height, res.width, 3)
        image = torch.zeros(shape, device=self.device, dtype=torch.float32)
        return image


def get_default_rendering_levels() -> Dict[str, float]:
    """
    Produces the default rendering levels, mapping object categories
    to their rendering level. Lower level renders on top,
    but levels lower than 0 don't render at all.
    """
    levels = dict(
        direction=2,
        ego=3,
        vehicle=4,
        bicycle=5,
        pedestrian=6,
        map_boundary=7,
        goal_waypoint=8,
        ground_truth=9,
        prediction=10,
        traffic_light=11,
        traffic_light_green=11,
        traffic_light_yellow=11,
        traffic_light_red=11,
        stop_sign=11,
        yield_sign=11,
        left_lane=12,
        joint_lane=13,
        right_lane=14,
        road=15,
    )
    return levels


def get_default_color_map() -> Dict[str, Tuple[int, int, int]]:
    """
    Produces the default color map, mapping object categories
    to RGB 3-tuples in [0,255] range.
    """
    color_map = dict(
        background=(0, 0, 0),
        road=(155, 155, 155),
        corridor=(0, 155, 0),
        ego=(255, 0, 0),
        vehicle=(32, 74, 135),
        bicycle=(24, 104, 225),
        pedestrian=(173, 127, 168),
        ground_truth=(196, 188, 165),
        prediction=(255, 155, 0),
        left_lane=(80, 127, 86),
        right_lane=(128, 0, 128),
        joint_lane=(255, 255, 255),
        direction=(100, 255, 255),
        rear_lights=(255, 255, 0),
        map_boundary=(255, 255, 0),
        traffic_light_green=(81, 179, 100),
        traffic_light_yellow=(240, 189, 39),
        traffic_light_red=(224, 53, 49),
        yield_sign=(210, 125, 45),
        stop_sign=(72, 60, 50),
        goal_waypoint=(139, 64, 0),
    )
    return color_map
