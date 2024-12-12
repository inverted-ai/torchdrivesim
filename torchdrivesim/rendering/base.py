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

    def copy(self):
        other = self.__class__(cfg=self.cfg, color_map=self.color_map.copy(), rendering_levels=self.rendering_levels.copy(),
                               res=self.res)
        other.scale = self.scale
        return other

    def get_color(self, element_type: str) -> Tuple[int, int, int]:
        return self.color_map[element_type]

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
