"""
Renderers used to visualize the state of the environments.
Currently three backends are supported (opencv, pytorch3d and nvdiffrast), along with a dummy renderer generating black images.
"""
import logging

from omegaconf import DictConfig, OmegaConf, SCMode

import torchdrivesim.rendering.pytorch3d
from torchdrivesim.rendering.base import RendererConfig, DummyRendererConfig, BirdviewRenderer, DummyRenderer
from torchdrivesim.rendering.cv2 import CV2RendererConfig, CV2Renderer
from torchdrivesim.rendering.pytorch3d import Pytorch3DRendererConfig, Pytorch3DRenderer
from torchdrivesim.rendering.nvdiffrast import NvdiffrastRendererConfig, NvdiffrastRenderer

logger = logging.getLogger(__name__)


def renderer_from_config(cfg: RendererConfig, *args, **kwargs) -> BirdviewRenderer:
    """
    Construct the selected renderer from config, by default using Pytorch3DRenderer.
    Additional arguments are passed to the constructor.
    """
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True, structured_config_mode=SCMode.INSTANTIATE)
    assert isinstance(cfg, RendererConfig)

    if cfg.backend == 'default':
        if torchdrivesim.rendering.pytorch3d.is_available:
            cfg = Pytorch3DRendererConfig(
                left_handed_coordinates=cfg.left_handed_coordinates,
                render_agent_direction=cfg.render_agent_direction,
                highlight_ego_vehicle=cfg.highlight_ego_vehicle
            )
        else:
            cfg = CV2RendererConfig(
                left_handed_coordinates=cfg.left_handed_coordinates,
                render_agent_direction=cfg.render_agent_direction,
                highlight_ego_vehicle=cfg.highlight_ego_vehicle
            )

    if isinstance(cfg, DummyRendererConfig):
        return DummyRenderer(cfg, *args, **kwargs)
    elif isinstance(cfg, CV2RendererConfig):
        return CV2Renderer(cfg, *args, **kwargs)
    elif isinstance(cfg, Pytorch3DRendererConfig):
        return Pytorch3DRenderer(cfg, *args, **kwargs)
    elif isinstance(cfg, NvdiffrastRendererConfig):
        return NvdiffrastRenderer(cfg, *args, **kwargs)
    else:
        raise ValueError(f'Unrecognized renderer type: {type(cfg)}')
