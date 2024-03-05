"""
PyTorch3D-based renderers, used by default in TorchDriveSim.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Tuple

try:
    import pytorch3d
    import pytorch3d.renderer
    is_available = True
except ImportError:
    pytorch3d = None
    is_available = False
import torch
from torch.nn import functional as F

from torchdrivesim.mesh import BirdviewMesh, tensor_color
from torchdrivesim.rendering.base import RendererConfig, BirdviewRenderer, Cameras
from torchdrivesim.utils import Resolution


class Pytorch3DNotFound(ImportError):
    pass


class RenderingBlend(Enum):
    """
    Blending choices from pytorch3d. May be ignored by other renderer types.
    https://pytorch3d.readthedocs.io/en/latest/modules/renderer/blending.html
    """
    hard = 'hard'
    soft = 'soft'
    sigmoid = 'sigmoid'


@dataclass
class Pytorch3DRendererConfig(RendererConfig):
    """
    Configuration of pytorch3d-based renderer.
    """
    backend: str = 'pytorch3d'
    differentiable_rendering: RenderingBlend = RenderingBlend.soft


class Shader2D(torch.nn.Module):
    """
    Shader that ignores lighting, based on https://github.com/facebookresearch/pytorch3d/issues/84
    """

    def __init__(self, device="cpu", background_color: Tuple[float, float, float] = (0, 0, 0),
                 blend=RenderingBlend.soft):
        super().__init__()
        self.background_color = background_color
        self.blend = blend

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        if not is_available:
            raise Pytorch3DNotFound()
        from pytorch3d.renderer import BlendParams
        pixel_colors = meshes.sample_textures(fragments)
        if self.blend == RenderingBlend.soft:
            images = pytorch3d.renderer.softmax_rgb_blend(pixel_colors, fragments,
                                                          BlendParams(background_color=self.background_color))
        elif self.blend == RenderingBlend.hard:
            images = pytorch3d.renderer.hard_rgb_blend(pixel_colors, fragments,
                                                       BlendParams(background_color=self.background_color))
        elif self.blend == RenderingBlend.sigmoid:
            images = pytorch3d.renderer.sigmoid_alpha_blend(pixel_colors, fragments,
                                                            BlendParams(background_color=self.background_color))
        else:
            raise ValueError("Unrecognized blend type: " + str(self.blend))
        return images


class Pytorch3DRenderer(BirdviewRenderer):
    """
    Renderer based on pytorch3d, using an orthographic projection and a trivial shader.
    Works on both GPU and CPU, but CPU is very slow.
    """
    def __init__(self, cfg: Pytorch3DRendererConfig, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.cfg: Pytorch3DRendererConfig = cfg
        self.renderer = self.make_renderer(
            self.res, blend=self.cfg.differentiable_rendering,
            background_color=tuple([x / 255.0 for x in self.get_color('background')])
        )

    def render_mesh(self, mesh: BirdviewMesh, res: Resolution, cameras: Cameras) -> torch.Tensor:
        for k in mesh.categories:
            if k not in mesh.colors:
                mesh.colors[k] = tensor_color(self.color_map[k])
            if k not in mesh.zs:
                mesh.zs[k] = self.rendering_levels[k]
        if self.cfg.highlight_ego_vehicle:
            mesh.colors["ego"] = tensor_color((self.color_map["ego"]))
        meshes = mesh.pytorch3d()
        if res != self.res:
            renderer = self.make_renderer(res, self.cfg.differentiable_rendering,
                                          tuple([x / 255.0 for x in self.get_color('background')]))
        else:
            renderer = self.renderer
        renderer.rasterizer.cameras = construct_pytorch3d_cameras(cameras)
        image = renderer(meshes)

        image = image[..., :3] * 255

        image = image.transpose(-2, -3)  # point x upwards, flip to right-handed coordinate frame
        if self.cfg.left_handed_coordinates:
            image = image.flip(dims=(-2,))  # flip horizontally

        return image

    @classmethod
    def make_renderer(cls, res, blend, background_color):
        if not is_available:
            raise Pytorch3DNotFound()
        settings = pytorch3d.renderer.mesh.rasterizer.RasterizationSettings(
            image_size=res.height,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        renderer = pytorch3d.renderer.MeshRenderer(
            rasterizer=pytorch3d.renderer.MeshRasterizer(
                raster_settings=settings
            ),
            shader=Shader2D(background_color=background_color,
                            blend=blend)  # type: ignore
        )
        return renderer


def construct_pytorch3d_cameras(cameras: Cameras) -> "pytorch3d.renderer.FoVOrthographicCameras":
    if not is_available:
        raise Pytorch3DNotFound()
    xy, sc, scale = cameras.xy, cameras.sc, cameras.scale
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
