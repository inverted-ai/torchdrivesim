"""
Simple demonstration for how to grab a map and visualize it.
"""
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import imageio
import lanelet2
from omegaconf import OmegaConf

from torchdrive.lanelet2 import load_lanelet_map, road_mesh_from_lanelet_map, lanelet_map_to_lane_mesh
from torchdrive.mesh import BirdviewMesh
from torchdrive.rendering import BirdviewRenderer, renderer_from_config, RendererConfig
from torchdrive.simulator import TorchDriveConfig
from torchdrive.utils import Resolution


@dataclass
class MapVisualizationConfig:
    maps_path: str
    map_name: str
    res: int = 1024
    fov: float = 200
    center: Optional[Tuple[float, float]] = None
    map_origin: Tuple[float, float] = (0, 0)
    orientation: float = 0
    save_path: str = './map_visualization.png'


def visualize_map(cfg: MapVisualizationConfig):
    device = 'cuda'
    res = Resolution(cfg.res, cfg.res)
    map_path = os.path.join(cfg.maps_path, f'{cfg.map_name}.osm')
    lanelet_map = load_lanelet_map(map_path, origin=cfg.map_origin)
    road_mesh = BirdviewMesh.set_properties(road_mesh_from_lanelet_map(lanelet_map), category='road')
    lane_mesh = lanelet_map_to_lane_mesh(lanelet_map)
    map_mesh = BirdviewMesh.concat(
        [road_mesh, lane_mesh]
    ).to(device)
    renderer_cfg = RendererConfig()
    renderer = renderer_from_config(
        renderer_cfg, device=device, static_mesh=map_mesh
    )
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
