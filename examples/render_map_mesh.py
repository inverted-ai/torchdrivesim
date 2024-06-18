"""
Simple demonstration for how to grab a map and visualize it.
Downloading maps from the IAI API requires IAI_API_KEY to be set.
"""
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple
import time

import numpy as np
import imageio
from omegaconf import OmegaConf

from torchdrivesim.map import find_map_config, download_iai_map, traffic_controls_from_map_config, MapConfig
from torchdrivesim.rendering import  renderer_from_config, RendererConfig
from torchdrivesim.utils import Resolution


@dataclass
class MapVisualizationConfig:
    mesh_path: str = "./mesh_file_path.json"
    lanelet_path: str = "./lanelet_file_path.osm"
    stoplines_path: str = "./stopline_file_path.json"
    res: int = 1024
    fov: float = 400
    center: Optional[Tuple[float, float]] = None
    map_origin: Tuple[float, float] = (0, 0)
    orientation: float = 0
    save_path: str = f'./{str(int(time.time()))}_map_visualization.png'


def visualize_map(cfg: MapVisualizationConfig):
    device = 'cuda'
    res = Resolution(cfg.res, cfg.res)
    # if cfg.iai_location_to_download is not None:
    #     download_iai_map(cfg.iai_location_to_download, save_path=f'{cfg.map_name}')
    # map_cfg = find_map_config(cfg.map_name)
    map_cfg = MapConfig(
        name="default",
        center=cfg.center,
        lanelet_path=cfg.lanelet_path,
        mesh_path=cfg.mesh_path,
        stoplines_path=cfg.stoplines_path,
    )
    driving_surface_mesh = map_cfg.road_mesh.to(device)
    renderer_cfg = RendererConfig(left_handed_coordinates=map_cfg.left_handed_coordinates)
    renderer = renderer_from_config(
        renderer_cfg, device=device, static_mesh=driving_surface_mesh
    )

    # traffic_controls = traffic_controls_from_map_config(map_cfg)
    # controls_mesh = renderer.make_traffic_controls_mesh(traffic_controls).to(renderer.device)
    # renderer.add_static_meshes([controls_mesh])

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
