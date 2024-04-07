"""
Simple demonstration for how to grab a map and visualize it.
Downloading maps from the IAI API requires IAI_API_KEY to be set.
"""
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import imageio
from omegaconf import OmegaConf

from torchdrivesim.map import find_map_config, download_iai_map, traffic_controls_from_map_config
from torchdrivesim.rendering import  renderer_from_config, RendererConfig
from torchdrivesim.utils import Resolution


@dataclass
class MapVisualizationConfig:
    map_name: str = "carla_Town03"
    iai_location_to_download: Optional[str] = None
    res: int = 1024
    fov: float = 400
    center: Optional[Tuple[float, float]] = None
    map_origin: Tuple[float, float] = (0, 0)
    orientation: float = 0
    save_path: str = './map_visualization.png'


def visualize_map(cfg: MapVisualizationConfig):
    device = 'cuda'
    res = Resolution(cfg.res, cfg.res)
    if cfg.iai_location_to_download is not None:
        download_iai_map(cfg.iai_location_to_download, save_path=f'{cfg.map_name}')
    map_cfg = find_map_config(cfg.map_name)
    driving_surface_mesh = map_cfg.road_mesh.to(device)
    renderer_cfg = RendererConfig(left_handed_coordinates=map_cfg.left_handed_coordinates)
    renderer = renderer_from_config(
        renderer_cfg, device=device, static_mesh=driving_surface_mesh
    )

    traffic_controls = traffic_controls_from_map_config(map_cfg)
    controls_mesh = renderer.make_traffic_controls_mesh(traffic_controls).to(renderer.device)
    renderer.add_static_meshes([controls_mesh])

    map_image = renderer.render_static_meshes(res=res, fov=cfg.fov)
    os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)
    imageio.imsave(
        cfg.save_path, map_image[0].cpu().numpy().astype(np.uint8)
    )


if __name__ == '__main__':
    cli_cfg: MapVisualizationConfig = OmegaConf.structured(
        MapVisualizationConfig(**OmegaConf.from_dotlist(sys.argv[1:]))
    )
    visualize_map(cli_cfg)  # type: ignore
