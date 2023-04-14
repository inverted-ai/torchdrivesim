"""
Nvdiffrast-based renderers, equivalent to those based on PyTorch3D but sometimes faster.
This module imports correctly if nvdiffrast is missing, but the renderer will raise the NvdiffrastNotFound exception.
"""
from dataclasses import dataclass
from typing import Optional

import torch

import logging

from torch.nn import functional as F
import pytorch3d

try:
    import nvdiffrast.torch as dr
    is_available = True
except ImportError:
    dr = None
    is_available = False

from torchdrivesim.mesh import BirdviewMesh, tensor_color
from torchdrivesim.rendering.base import RendererConfig, BirdviewRenderer
from torchdrivesim.utils import Resolution

logger = logging.getLogger(__name__)

glctx_sessions = {}


class NvdiffrastNotFound(ImportError):
    """
    Nvdiffrast is not installed.
    """
    pass


def get_glctx_session(device, opengl=True):
    if not is_available:
        raise NvdiffrastNotFound()
    device = torch.device(device)
    if device.type == 'cpu':
        logger.debug('\'nvdiffrast\' supports only rendering on GPU.')
        return None
    global glctx_sessions
    if device not in glctx_sessions:
        if opengl or not hasattr(dr, 'RasterizeCudaContext'):
            glctx_sessions[device] = dr.RasterizeGLContext(device=device, output_db=False)
        else:
            glctx_sessions[device] = dr.RasterizeCudaContext(device=device)
    else:
        if hasattr(dr, 'RasterizeCudaContext'):
            if opengl and isinstance(glctx_sessions[device], dr.RasterizeCudaContext):
                glctx_sessions[device] = dr.RasterizeGLContext(device=device, output_db=False)
            elif not opengl and isinstance(glctx_sessions[device], dr.RasterizeGLContext):
                glctx_sessions[device] = dr.RasterizeCudaContext(device=device)
    return glctx_sessions[device]


@dataclass
class NvdiffrastRendererConfig(RendererConfig):
    """
    Configuration of nvdiffrast-based renderer.
    """
    backend: str = 'nvdiffrast'
    antialias: bool = False
    opengl: bool = True  #: if False, use CUDA for rendering
    max_minibatch_size: Optional[int] = None  #: used to pre-allocate memory, which may speed up rendering


class NvdiffrastRenderer(BirdviewRenderer):
    """
    Similar to PyTorch3DRenderer, and producing indistinguishable images, but sometimes faster.
    Note that nvdiffrast requires separate installation and is subject to its own license terms.
    """
    def __init__(self, cfg: NvdiffrastRendererConfig, *args, **kwargs):
        if not is_available:
            raise NvdiffrastNotFound()
        super().__init__(cfg, *args, **kwargs)
        self.cfg: NvdiffrastRendererConfig = cfg
        self.glctx = get_glctx_session(self.device, opengl=self.cfg.opengl)
        if self.glctx is None:
            raise RuntimeError('Failed to obtain glctx session for nvdiffrast')

    def render_mesh(self, mesh: BirdviewMesh, res: Resolution, cameras: pytorch3d.renderer.FoVOrthographicCameras)\
            -> torch.Tensor:
        for k in mesh.categories:
            if k not in mesh.colors:
                mesh.colors[k] = tensor_color(self.color_map[k])
            if k not in mesh.zs:
                mesh.zs[k] = self.rendering_levels[k]
        if self.cfg.highlight_ego_vehicle:
            mesh.colors["ego"] = tensor_color((self.color_map["ego"]))
        meshes = mesh.pytorch3d()
        mesh = mesh.fill_attr()
        if not hasattr(self.glctx, 'initial_dummy_frame_rendered') and \
                self.cfg.max_minibatch_size is not None:
            maximum_min_batch_size = self.cfg.max_minibatch_size
            dummy_verts = torch.Tensor([[0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1]]).to(self.device)
            dummy_faces = torch.IntTensor([[0, 1, 2]]).to(self.device)
            dummy_ranges = torch.IntTensor([[0, 1]]).expand(maximum_min_batch_size, -1).contiguous()
            _, _ = dr.rasterize(self.glctx, dummy_verts, dummy_faces, resolution=[self.res.height, self.res.width],
                                ranges=dummy_ranges)
            self.glctx.initial_dummy_frame_rendered = True
        meshes_proj = meshes.update_padded(new_verts_padded=cameras.transform_points_ndc(meshes.verts_padded()))
        verts_ndc = meshes_proj.verts_packed()
        # We need to flip x and y because the coordinate system of OpenGL is different from Pytorch3D
        verts_ndc = F.pad(torch.cat([-verts_ndc[..., :2], verts_ndc[..., 2:3]], dim=-1), (0, 1), value=1.0)
        faces_packed = meshes_proj.faces_packed().to(torch.int32)
        tris_first_idx = meshes_proj.mesh_to_faces_packed_first_idx()
        tris_count = meshes_proj.num_faces_per_mesh()
        ranges = torch.stack([tris_first_idx, tris_count], dim=-1).cpu().to(torch.int32)
        rast, _ = dr.rasterize(self.glctx, verts_ndc, faces_packed, resolution=[res.height, res.width],
                               ranges=ranges)
        vertices_attributes = mesh.attrs.reshape(-1, 3)
        image, _ = dr.interpolate(vertices_attributes, rast, faces_packed)
        # Change background color in case it's not black
        if sum(self.get_color('background')) != 0:
            image = torch.where(rast[..., 3:4] > 0, image,
                                torch.tensor([x / 255.0 for x in self.get_color('background')],
                                             device=image.device))
        if self.cfg.antialias:
            image = dr.antialias(image, rast, verts_ndc, faces_packed)

        image = image[..., :3] * 255

        image = image.transpose(-2, -3)  # point x upwards, flip to right-handed coordinate frame
        if self.cfg.left_handed_coordinates:
            image = image.flip(dims=(-2,))  # flip horizontally

        return image
