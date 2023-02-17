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
import pytorch3d
import pytorch3d.renderer
import torch
from torch import Tensor
from torch.nn import functional as F

from torchdrivesim.mesh import BirdviewMesh, BaseMesh, rendering_mesh
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


@dataclass
class DummyRendererConfig(RendererConfig):
    """
    For DummyRenderer.
    """
    backend: str = 'dummy'


class BirdviewRenderer(abc.ABC):
    """
    A renderer producing simple 2D birdview images based on static background meshes and rectangular agents.
    Currently only square resolutions are supported. The renderer always operates in batch mode,
    with a single batch dimension on the left.

    Args:
        cfg: configuration object, usually subclassed
        device: torch device used for rendering
        batch_size: if road_mesh is not specified, this is used to determine batch size
        static_mesh: BirdviewMesh object specifying drivable surface (empty mesh is used if not provided)
        world_center: Bx2 float tensor, defaults to geometric centre of the road mesh
        color_map: a dictionary of RGB tuples in 0-255 range specifying colors of different rendered elements
        res: default resolution
    """
    def __init__(self, cfg: RendererConfig, device: Optional[torch.device] = None, batch_size: Optional[int] = None,
                 static_mesh: Optional[BirdviewMesh] = None, world_center: Optional[Tensor] = None,
                 color_map: Optional[Dict[str, Tuple[int, int, int]]] = None,
                 rendering_levels: Optional[Dict[str, float]] = None,
                 res: Resolution = Resolution(64, 64), fov: float = 35):

        self.cfg: RendererConfig = cfg
        self.res = res
        self.scale = 2.0 / fov
        if device is None:
            if static_mesh is not None:
                self.device = static_mesh.device
            else:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.color_map = color_map
        if self.color_map is None:
            self.color_map = get_default_color_map()
        self.rendering_levels = rendering_levels
        if self.rendering_levels is None:
            self.rendering_levels = get_default_rendering_levels()

        if static_mesh is None:
            if batch_size is None:
                raise ValueError("Either road_mesh or batch_size needs to be specified for the renderer")
            static_mesh = BirdviewMesh.empty(batch_size=batch_size)
        self.static_mesh: BirdviewMesh = static_mesh

        if world_center is None:
            if 'road' in self.static_mesh.categories:
                world_center = self.static_mesh.separate_by_category()['road'].center
            else:
                world_center = self.static_mesh.center
        self.world_center = world_center.to(device)

    def get_color(self, element_type: str) -> Tuple[int, int, int]:
        return self.color_map[element_type]

    def to(self, device: torch.device):
        """
        Moves the renderer to another device in place.
        """
        self.device = device
        self.world_center = self.world_center.to(device)
        self.static_mesh = self.static_mesh.to(device)
        return self

    def add_static_meshes(self, meshes: List[BirdviewMesh]) -> None:
        """
        Includes additional static elements to render.
        """
        self.static_mesh = self.static_mesh.concat(
            [self.static_mesh] + meshes
        )

    def copy(self):
        return self.expand(1)

    def expand(self, n: int):
        """
        Adds another dimension with size n on the right of the batch dimension and flattens them.
        Returns a new renderer, without modifying the current one.
        """
        expand = lambda x: x.unsqueeze(1).expand((x.shape[0], n) + x.shape[1:]).reshape((n * x.shape[0],) + x.shape[1:])\
            if x is not None else None
        other = self.__class__(cfg=self.cfg, device=self.device, batch_size=self.static_mesh.batch_size,
                               world_center=expand(self.world_center), color_map=self.color_map.copy(),
                               rendering_levels=self.rendering_levels.copy())
        other.static_mesh = self.static_mesh.expand(n)

        return other

    def select_batch_elements(self, idx: Tensor):
        """
        Selects given elements from the batch, potentially with repetitions.
        Returns a new renderer, without modifying the current one.

        Args:
            idx: one-dimensional integer tensor
        """
        other = self.__class__(cfg=self.cfg, device=self.device, batch_size=self.static_mesh.batch_size,
                               world_center=self.world_center[idx], color_map=self.color_map.copy(),
                               rendering_levels=self.rendering_levels.copy())
        other.static_mesh = self.static_mesh[idx]
        return other

    def render_static_meshes(self, camera_xy: Optional[Tensor] = None, camera_sc: Optional[Tensor] = None,
                             res: Resolution = None, fov: float = None) -> Tensor:
        """
        Render a single birdview of the static mesh only. Nc is the number of cameras.
        C=3 is the number of RGB channels.

        Args:
            camera_xy: Ncx2 tensor of camera positions
            camera_sc: Ncx2 tensor of camera orientations (sine and cosine of yaw angle)
            res: Resolution, currently only square resolutions are supported
            fov: Field of view in meters

        Returns:
            birdview image tensor of shape NcxHxWxC
        """
        if camera_xy is None:
            camera_xy = self.world_center
            camera_psi = torch.ones_like(camera_xy[..., :1]) * np.pi / 2
            camera_sc = torch.cat([torch.sin(camera_psi), torch.cos(camera_psi)], dim=-1)
        scale = (2.0 / fov) if fov is not None else self.scale
        cameras = self.construct_cameras(camera_xy.reshape(-1, 2), camera_sc.reshape(-1, 2), scale=scale)
        image = self.render_mesh(self.static_mesh, res, cameras)
        return image

    def transform(self, points: Tensor, pose: Tensor) -> Tensor:
        """
        Given points relative to a pose, produce absolute positions of the points.
        There can be zero or more batch dimensions.

        Args:
            points: BxNx2 tensor
            pose: Bx3 tensor of position (x,y) and orientation (yaw angle in radians)

        Returns:
            Bx2 tensor of absolute positions
        """
        xy = pose[..., :2].unsqueeze(-2).expand_as(points)
        psi = pose[..., 2:3].unsqueeze(-2).expand_as(points[..., :1])
        return rotate(points, psi) + xy

    def make_actor_mesh(self, agent_state: Dict[str, Tensor], agent_attributes: Dict[str, Tensor]) -> BirdviewMesh:
        """
        Creates a mesh representing given actors. Each vertex and each face corresponds to a unique agent and
        both vertices and faces for each agent are continuous in the resulting tensor to allow for subsequent masking.
        For each agent there are seven vertices and three faces, specifying its bounding box and direction.
        Direction vertices use the 'direction' category, while agent categories are copied from input dictionaries.
        """
        meshes = []
        for k in agent_state.keys():
            lenwid = agent_attributes[k]
            n_actors = lenwid.shape[-2]
            length, width = lenwid[..., 0], lenwid[..., 1]
            state = agent_state[k]
            corners = torch.stack([
                torch.stack([x, y], dim=-1) for (x, y) in
                [(length, width), (length, - width), (- length, - width), (- length, width)]
            ], dim=-2) * 0.5
            batch_size = state.size()[0]
            actor_verts = self.transform(corners, state).reshape(batch_size, n_actors * 4, 2)

            actor_faces = torch.tensor([[0, 1, 3], [1, 3, 2]], dtype=torch.long, device=self.device)
            actor_faces = actor_faces.expand(batch_size, n_actors, 2, 3)
            offsets = 4 * torch.arange(start=0, end=n_actors, dtype=torch.long,
                                    device=self.device).reshape(n_actors, 1, 1).expand_as(actor_faces)
            actor_faces = actor_faces + offsets
            actor_faces = actor_faces.reshape(batch_size, n_actors * 2, 3)

            if self.cfg.render_agent_direction:
                direction_mesh = self.make_direction_mesh(lenwid=lenwid, pose=state[..., :3])
                # custom concatenation of tensors, so that both vertices and faces belonging
                # to each agent (both bbox and direction) are contiguous
                # this allows for subsequent masking of agents
                av = 4  # verts per actor
                dv = 3  # verts per direction
                verts = torch.cat([
                    actor_verts.reshape(batch_size, n_actors, av, 2),
                    direction_mesh.verts.reshape(batch_size, n_actors, dv, 2)
                ], dim=-2).reshape(batch_size, n_actors * (av + dv), 2)
                actor_faces = actor_faces + actor_faces.div(av, rounding_mode='trunc') * dv
                direction_faces = direction_mesh.faces + av * (direction_mesh.faces.div(dv, rounding_mode='trunc') + 1)
                faces = torch.cat([
                    actor_faces.reshape(batch_size, n_actors, 2, 3),
                    direction_faces.reshape(batch_size, n_actors, 1, 3)
                ], dim=-2).reshape(batch_size, n_actors * 3, 3)
                mesh = BirdviewMesh(
                    verts=verts, faces=faces, categories=[k, 'direction'],
                    vert_category=torch.cat([
                        torch.zeros((batch_size, n_actors, av), dtype=torch.int64, device=verts.device),
                        torch.ones((batch_size, n_actors, dv), dtype=torch.int64, device=verts.device)
                    ], dim=-1).reshape(batch_size, n_actors * (av + dv)),
                    colors=dict(), zs=dict(),
                )
            else:
                mesh = BirdviewMesh.set_properties(BaseMesh(verts=actor_verts, faces=actor_faces), category=k)
            meshes.append(mesh)
        return BirdviewMesh.concat(meshes)

    def render_frame(
        self, agent_state: Dict[str, Tensor], agent_attributes: Dict[str, Tensor],
        camera_xy: Optional[Tensor] = None, camera_sc: Optional[Tensor] = None,
        rendering_mask: Dict[str, Tensor] = None, res: Optional[Resolution] = None,
        traffic_controls: Optional[Dict[str, BaseTrafficControl]] = None, fov: Optional[float] = None
    ) -> Tensor:
        """
        Renders the agents and traffic controls on top of the static mesh.
        Cameras batch size is (B*Nc), which corresponds to using Nc cameras per batch element.
        This extra dimension is added on the right of batch dimension and flattened,
        to match the semantics of `extend` from pytorch3d.
        If cameras is None, one egocentric camera per agent is used, that is Nc = A.

        Args:
            agent_state: maps agent types to state tensors of shape BxAxSt, where St >= 3 and the first three
                components are x coordinate, y coordinate, and orientation in radians
            agent_attributes: maps agent types to static attributes tensors of shape BxAxAttr, where Attr >= 2
                and the first two components are length and width of the agent
            camera_xy: BxNcx2 tensor of camera positions, by default one camera placed on each agent
            camera_sc: BxNcx2 tensor of camera orientations (sine and cosine), by default matching agent orientations
            rendering_mask:  BxNcxA tensor per agent type, indicating which cameras see which agents
            res: resolution HxW of the resulting image, currently only square resolutions are supported
            traffic_controls: traffic controls by type (traffic-light, yield, etc.)
            fov: Field of view in meters

        Returns:
            tensor image of float RGB values in [0,255] range with shape shape (B*Nc)xAxCxHxW
        """
        batch_size = max([v.shape[0] for v in agent_state.values()])
        scale = (2.0 / fov) if fov is not None else self.scale
        agent_count = sum([v.shape[-2] for v in agent_state.values()])
        if camera_xy is None:
            xy = torch.cat([x[..., :2] for x in agent_state.values()], dim=-2)
            psi = torch.cat([x[..., 2:3] for x in agent_state.values()], dim=-2)
            sc = torch.cat([torch.sin(psi), torch.cos(psi)], dim=-1)
            n_cameras_per_batch = agent_count
            # Set orthographic camera on agents that are being predicted
            cameras = self.construct_cameras(
                # put agent dimension first to easier extend the mesh tensor
                xy.transpose(0, 1).reshape(-1, 2),
                sc.transpose(0, 1).reshape(-1, 2),
                scale=scale)
        else:
            n_cameras_per_batch = camera_xy.shape[-2]
            camera_xy, camera_sc = camera_xy.reshape(-1, 2), camera_sc.reshape(-1, 2)
            cameras = self.construct_cameras(camera_xy, camera_sc, scale=scale)

        static_mesh = self.static_mesh
        actor_mesh = self.make_actor_mesh(agent_state, agent_attributes)

        actor_mesh = actor_mesh.expand(n_cameras_per_batch)
        static_mesh = static_mesh.expand(n_cameras_per_batch)

        if rendering_mask is not None:
            rendering_mask = torch.cat(list(rendering_mask.values()), dim=-1)
            rendering_mask = rendering_mask.flatten(0, 1)
            mask_agents = lambda x: x * rendering_mask.repeat_interleave(
                x.shape[1] // agent_count, dim=-1
            ).unsqueeze(-1).expand_as(x)
            actor_mesh = dataclasses.replace(
                actor_mesh, faces=mask_agents(actor_mesh.faces)
            )

        meshes = [
            static_mesh,
            actor_mesh,
        ]

        if traffic_controls is not None:
            traffic_controls = {k: v.extend(n_cameras_per_batch) for k, v in traffic_controls.items()}
            controls_mesh = self.make_traffic_controls_mesh(traffic_controls)
            meshes.append(controls_mesh)

        mesh = BirdviewMesh.concat(meshes)

        if res is None:
            res = self.res

        try:
            image = self.render_mesh(mesh, res, cameras)
        except RuntimeError as e:
            logger.exception(e)
            image = torch.zeros((batch_size * n_cameras_per_batch, res.height, res.width, 3), device=mesh.verts.device)
            try:
                # save the problematic mesh for debugging purposes
                with open('bad-mesh.pkl', 'wb') as f:
                    pickle.dump((mesh.verts.detach().cpu(), mesh.faces.detach().cpu()), f)
            except RuntimeError:
                pass

        # recover informative shape
        if camera_xy is None:
            image = image.reshape(agent_count, -1, res.height, res.width, 3)
            image = image.permute(1, 0, 4, 2, 3)
        else:
            image = image.reshape(-1, res.height, res.width, 3)
            image = image.permute(0, 3, 1, 2)
        if camera_xy is None:
            # recover original batch shape
            image = image.reshape((batch_size, *image.shape[1:]))
        return image

    @abc.abstractmethod
    def render_mesh(self, mesh: BirdviewMesh, res: Resolution, cameras: pytorch3d.renderer.FoVOrthographicCameras)\
            -> Tensor:
        """
        Renders a given mesh, producing BxHxWxC tensor image of float RGB values in [0,255] range.
        """
        pass

    def construct_cameras(self, xy: Tensor, sc: Tensor, scale: Optional[float] = None)\
            -> pytorch3d.renderer.FoVOrthographicCameras:
        """
        Create PyTorch3D cameras object for given positions and orientations.
        Input tensor dimensions should be Bx2.
        """
        scale = self.scale if scale is None else scale
        return construct_pytorch3d_cameras(xy, sc, scale=scale)

    def build_verts_faces_from_bounding_box(self, bbs: Tensor, z: float = 2) -> Tuple[Tensor, Tensor]:
        """
        Triangulates actors for rendering. Input is a tensor of bounding boxes of shape ...xAx4x2,
        where A is the number of actors. Outputs are shaped ...x4*Ax2 and ...x2*Ax3 respectively.
        """
        batch_dims = bbs.size()[:-3]
        n_actors = bbs.size()[-3]
        verts = bbs.reshape(*batch_dims, -1, 2)

        faces = torch.tensor([[0, 1, 3], [1, 3, 2]], dtype=torch.long, device=self.device)
        faces = faces.unsqueeze(0).expand(*batch_dims, n_actors, 2, 3)
        offsets = 4 * torch.arange(start=0, end=n_actors, dtype=torch.long,
                                   device=self.device).reshape(n_actors, 1, 1).expand_as(faces)
        faces = faces + offsets
        faces = faces.reshape(*batch_dims, 2 * n_actors, 3)

        return verts, faces

    def make_traffic_controls_mesh(self, traffic_controls: Dict[str, BaseTrafficControl]) -> BirdviewMesh:
        """
        Create a mesh showing traffic controls.
        """
        if traffic_controls:
            batch_size = max(element.corners.shape[0] for element in traffic_controls.values())
        else:
            batch_size = 1
        meshes = []
        for control_type, element in traffic_controls.items():
            if element.corners.shape[-2] == 0:
                continue
            verts, faces = self.build_verts_faces_from_bounding_box(element.corners)
            if control_type == 'traffic-light':
                categories = [f'{control_type}_{state}' for state in element.allowed_states]
                vert_category = element.state.unsqueeze(-1).expand(element.state.shape + (4,)).flatten(-2, -1)
                meshes.append(BirdviewMesh(
                    verts=verts, faces=faces, categories=categories, vert_category=vert_category,
                    zs=dict(), colors=dict()
                ))
            else:
                meshes.append(rendering_mesh(
                    BaseMesh(verts=verts, faces=faces), category=control_type # TODO: add light state
                ))
        if meshes:
            return BirdviewMesh.concat(meshes)
        else:
            return BirdviewMesh.empty(dim=2, batch_size=batch_size)

    def make_direction_mesh(self, lenwid: Tensor, pose: Tensor, size: float = 0.3) -> BaseMesh:
        """
        Create a mesh indicating the direction of each agent.

        Args:
            lenwid: BxAx2 tensor specifying length and width of the agents
            pose: Bx3 tensor of position (x,y) and orientation (yaw angle in radians)
            size: determines the size of the triangle indicating direction
        """
        batch_size = lenwid.shape[0]
        n_actors = lenwid.shape[-2]
        corners = torch.stack([
            F.pad( lenwid[..., 0:1] * size, (1, 0), value=0.0),
            F.pad( lenwid[..., 1:2] * 0.5,  (0, 1), value=0.0),
            F.pad(-lenwid[..., 1:2] * 0.5,  (0, 1), value=0.0),
        ], dim=-2).flip([-1])
        offset = torch.cat([
            lenwid[..., 0:1]*(0.5 - size),
            torch.zeros_like(lenwid[..., 1:2])
        ], dim=-1).unsqueeze(-2)
        corners = corners + offset
        verts = self.transform(corners, pose)
        verts = verts.reshape(batch_size, n_actors * 3, 2)
        # verts = verts.unsqueeze(0)
        faces = torch.tensor(
            [[[0,  1,  2]]], dtype=torch.long, device=self.device
        ).expand(batch_size, n_actors, 3)
        faces_offset = 3 * torch.arange(
            start=0, end=n_actors, dtype=torch.long, device=self.device
        ).reshape(1, n_actors, 1).expand_as(faces)
        faces = faces + faces_offset
        # faces = faces.expand(batch_size, n_actors, 3)
        return BaseMesh(verts=verts, faces=faces)


class DummyRenderer(BirdviewRenderer):
    """
    Produces a black image of the required size. Mostly used for debugging and benchmarking.
    """
    def render_mesh(self, mesh: BirdviewMesh, res: Resolution, cameras: pytorch3d.renderer.FoVOrthographicCameras)\
            -> Tensor:
        camera_batch_size = cameras.get_camera_center().shape[0]
        shape = (camera_batch_size, res.height, res.width, 3)
        image = torch.zeros(shape, device=self.device, dtype=torch.float32)
        return image


def construct_pytorch3d_cameras(xy: Tensor, sc: Tensor, scale: float) -> pytorch3d.renderer.FoVOrthographicCameras:
    """
    Create PyTorch3D cameras object for given positions and orientations.
    Input tensor dimensions should be Bx2.
    """
    assert xy.shape == sc.shape
    device = xy.device
    cs_neg = torch.flip(sc, dims=(-1,)) * torch.tensor([[1, -1]], dtype=sc.dtype, device=sc.device)
    rotation_matrix = torch.stack([cs_neg, sc], dim=-2)
    # pytorch3d seems to rotate the provided translation vector
    reverse_rotation = rotation_matrix.transpose(-1, -2)
    rotated_translation = reverse_rotation.matmul(-xy.unsqueeze(-1)).squeeze(-1)
    t = F.pad(rotated_translation, (0, 1), mode='constant', value=0)
    r = F.pad(rotation_matrix, (0, 1, 0, 1), mode='constant', value=0)
    r[..., -1, -1] = 1
    scale = torch.tensor([[scale, scale, 1]], dtype=xy.dtype, device=device).expand_as(t)
    cameras = pytorch3d.renderer.FoVOrthographicCameras(device=device, scale_xyz=scale, T=t, R=r)
    return cameras


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
        ground_truth=8,
        prediction=9,
        traffic_light=10,
        traffic_light_green=10,
        traffic_light_yellow=10,
        traffic_light_red=10,
        stop_sign=10,
        yield_sign=10,
        left_lane=11,
        joint_lane=12,
        right_lane=13,
        road=14,
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
        ego=(32, 74, 135),
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
    )
    return color_map
