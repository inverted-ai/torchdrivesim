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

from torchdrivesim.lanelet2 import load_lanelet_map, road_mesh_from_lanelet_map, lanelet_map_to_lane_mesh
from torchdrivesim.mesh import BirdviewMesh
from torchdrivesim.rendering import BirdviewRenderer, renderer_from_config, RendererConfig
from torchdrivesim.simulator import TorchDriveConfig
from torchdrivesim.utils import Resolution


@dataclass
class MapVisualizationConfig:
    driving_surface_mesh_path: str = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "../resources/maps/carla/meshes/Town03_driving_surface_mesh.pkl"
    )
    map_name: str = "Town03"
    res: int = 1024
    fov: float = 1000
    center: Optional[Tuple[float, float]] = None
    map_origin: Tuple[float, float] = (0, 0)
    orientation: float = 0
    save_path: str = './map_visualization.png'


def visualize_map(cfg: MapVisualizationConfig):
    device = 'cuda'
    res = Resolution(cfg.res, cfg.res)
    driving_surface_mesh = BirdviewMesh.unpickle(cfg.driving_surface_mesh_path).to(device)
    renderer_cfg = RendererConfig(left_handed_coordinates=True)
    renderer = renderer_from_config(
        renderer_cfg, device=device, static_mesh=driving_surface_mesh
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
