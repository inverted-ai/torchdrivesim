"""
A test script to check the alignment between a local map and one used by IAI server.
"""
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
import invertedai

from torchdrivesim.behavior.common import InitializationFailedError
from torchdrivesim.kinematic import KinematicBicycle
from torchdrivesim.lanelet2 import load_lanelet_map, road_mesh_from_lanelet_map, lanelet_map_to_lane_mesh
from torchdrivesim.mesh import BirdviewMesh
from torchdrivesim.rendering import renderer_from_config, RendererConfig
from torchdrivesim.simulator import TorchDriveConfig, Simulator
from torchdrivesim.utils import Resolution


@dataclass
class AlignmentCheckConfig:
    osm_path: Optional[str] = None  #: local file with Lanelet2 map
    mesh_path: Optional[str] = None  #: if file exists, osm_path is ignored, otherwise it is created
    iai_location_name: str = ""  #: location name accepted by IAI API, corresponding to the above map
    center: Optional[Tuple[float, float]] = None  #: point around which to initialize, in map's coordinate frame
    map_origin: Tuple[float, float] = (0, 0)  #: lat/lon of the origin for UTM projection of the OSM file
    local_save_path: str = './local.png'  #: where to save the visualization of the local map
    remote_save_path: str = './remote.png'  #: where to save the visualization of the remote map
    left_handed: bool = False  #: whether the map's coordinate frame is left-handed
    agent_count: int = 5  #: how many agents to initialize
    # image parameters set to match the remote default
    res: int = 512
    fov: float = 120
    orientation: float = np.pi / 2


def visualize_maps(cfg: AlignmentCheckConfig):
    device = 'cuda'
    res = Resolution(cfg.res, cfg.res)
    simulator_cfg = TorchDriveConfig(left_handed_coordinates=cfg.left_handed,
                                     renderer=RendererConfig(left_handed_coordinates=cfg.left_handed))
    if not cfg.iai_location_name:
        raise ValueError('iai_location_name not specified')
    location = cfg.iai_location_name
    if cfg.mesh_path and os.path.exists(cfg.mesh_path):
        driving_surface_mesh = BirdviewMesh.unpickle(cfg.mesh_path).to(device)
    elif cfg.osm_path:
        lanelet_map = load_lanelet_map(cfg.osm_path, origin=cfg.map_origin)
        road_mesh = BirdviewMesh.set_properties(mesh=road_mesh_from_lanelet_map(lanelet_map), category='road')
        lane_mesh = lanelet_map_to_lane_mesh(lanelet_map)
        driving_surface_mesh = BirdviewMesh.concat([road_mesh, lane_mesh])
        if cfg.mesh_path:
            driving_surface_mesh.pickle(cfg.mesh_path)
    else:
        raise ValueError('Either mesh_path or osm_path needs to be specified and exist')
    driving_surface_mesh = driving_surface_mesh.to(device)
    try:
        response = invertedai.api.initialize(
            location=location, agent_count=cfg.agent_count,
            location_of_interest=tuple(cfg.center) if cfg.center is not None else None,
            get_birdview=True,
        )
    except invertedai.error.InvalidRequestError:
        raise InitializationFailedError()
    agent_attributes = torch.stack(
        [torch.tensor(at.tolist()) for at in response.agent_attributes], dim=-2
    )
    agent_states = torch.stack(
        [torch.tensor(st.tolist()) for st in response.agent_states], dim=-2
    )
    agent_attributes, agent_states = agent_attributes.unsqueeze(0), agent_states.unsqueeze(0)
    agent_attributes, agent_states = agent_attributes.to(device).to(torch.float32), agent_states.to(device).to(torch.float32)
    kinematic_model = KinematicBicycle()
    kinematic_model.set_params(lr=agent_attributes[..., 2])
    kinematic_model.set_state(agent_states)
    renderer = renderer_from_config(simulator_cfg.renderer, static_mesh=driving_surface_mesh)
    simulator = Simulator(
        cfg=simulator_cfg, road_mesh=driving_surface_mesh,
        kinematic_model=dict(vehicle=kinematic_model), agent_size=dict(vehicle=agent_attributes[..., :2]),
        initial_present_mask=dict(vehicle=torch.ones_like(agent_states[..., 0], dtype=torch.bool)),
        renderer=renderer,
    )
    print(f'Local map center: {simulator.get_world_center().cpu().squeeze(0).numpy().tolist()}')
    print(f'Agent states:\n{simulator.get_state()["vehicle"].cpu().squeeze(0).numpy()}')

    response.birdview.decode_and_save(cfg.remote_save_path)

    if cfg.center is None:
        camera_xy = simulator.get_world_center().to(device)
    else:
        camera_xy = torch.tensor(cfg.center).unsqueeze(0).to(torch.float32).to(device)
    camera_psi = torch.ones_like(camera_xy[..., :1]) * cfg.orientation
    local_image = simulator.render(camera_xy=camera_xy, camera_psi=camera_psi, res=res, fov=cfg.fov)
    os.makedirs(os.path.dirname(cfg.local_save_path), exist_ok=True)
    cv2.imwrite(cfg.local_save_path,
                cv2.cvtColor(local_image.cpu().squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8), cv2.COLOR_BGR2RGB)
                )


if __name__ == '__main__':
    cli_cfg: AlignmentCheckConfig = OmegaConf.structured(
        AlignmentCheckConfig(**OmegaConf.from_dotlist(sys.argv[1:]))
    )
    visualize_maps(cli_cfg)  # type: ignore
