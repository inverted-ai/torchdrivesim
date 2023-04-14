"""
PyTorch3D-based renderers, used by default in TorchDriveSim.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import pytorch3d
import torch
from pytorch3d.renderer import BlendParams

from torchdrivesim.mesh import BirdviewMesh, tensor_color
from torchdrivesim.rendering.base import RendererConfig, BirdviewRenderer
from torchdrivesim.utils import Resolution


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
        if res != self.res:
            renderer = self.make_renderer(res, self.cfg.differentiable_rendering,
                                          tuple([x / 255.0 for x in self.get_color('background')]))
        else:
            renderer = self.renderer
        renderer.rasterizer.cameras = cameras
        image = renderer(meshes)

        image = image[..., :3] * 255

        image = image.transpose(-2, -3)  # point x upwards, flip to right-handed coordinate frame
        if self.cfg.left_handed_coordinates:
            image = image.flip(dims=(-2,))  # flip horizontally

        return image

    @classmethod
    def make_renderer(cls, res, blend, background_color):

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
