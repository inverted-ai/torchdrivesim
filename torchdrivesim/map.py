import dataclasses
import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import torchdrivesim
from torchdrivesim.lanelet2 import LaneletMap, load_lanelet_map, road_mesh_from_lanelet_map, lanelet_map_to_lane_mesh
from torchdrivesim.mesh import BirdviewMesh


@dataclass
class Stopline:
    actor_id: int
    agent_type: str
    x: float
    y: float
    length: float
    width: float
    orientation: float


@dataclass
class MapConfig:
    """
    Encapsulates various map metadata, including where to find files defining the map.
    Map definition includes a coordinate frame and traffic signals.
    """
    name: str
    left_handed_coordinates: bool = False
    center: Optional[Tuple[float, float]] = None

    lanelet_path: Optional[str] = None
    lanelet_map_origin: Tuple[float, float] = (0, 0)
    mesh_path: Optional[str] = None
    stoplines_path: Optional[str] = None
    traffic_light_controller_path: Optional[str] = None

    iai_location_name: Optional[str] = None
    note: Optional[str] = None

    @property
    def lanelet_map(self) -> Optional[LaneletMap]:
        if self.lanelet_path is None:
            return None
        return load_lanelet_map(self.lanelet_path, origin=self.lanelet_map_origin)

    @property
    def road_mesh(self) -> Optional[BirdviewMesh]:
        if self.mesh_path is None:
            if self.lanelet_path is None:
                return None
            else:
                lanelet_map = self.lanelet_map
                road_mesh = road_mesh_from_lanelet_map(lanelet_map)
                road_mesh = BirdviewMesh.set_properties(road_mesh, category='road').to(road_mesh.device)
                lane_mesh = lanelet_map_to_lane_mesh(lanelet_map, left_handed=self.left_handed_coordinates)
                combined_mesh = lane_mesh.merge(road_mesh)
                return combined_mesh
        else:
            return BirdviewMesh.load(self.mesh_path)

    @property
    def stoplines(self) -> List[Stopline]:
        if self.stoplines_path is None:
            return []
        with open(self.stoplines_path, 'r') as f:
            stoplines = [Stopline(**d) for d in json.load(f)]
        return stoplines

    # TODO: traffic light controllers


def _filename_defaults(name: str) -> Dict[str, str]:
    return dict(
        lanelet_path=f'{name}.osm',
        mesh_path=f'{name}_mesh.json',
        stoplines_path=f'{name}_stoplines.json',
        traffic_light_controller_path=f'{name}_traffic_light_controller.json',
    )


def resolve_paths_to_absolute(cfg: MapConfig, root: str) -> MapConfig:
    resolved_paths = dict()
    for pathname, default in _filename_defaults(cfg.name).items():
        existing_path = getattr(cfg, pathname)
        if existing_path is None:
            existing_path = default
        if os.path.isabs(existing_path):
            continue
        candidate_path = os.path.join(root, existing_path)
        if os.path.exists(candidate_path):
            resolved_paths[pathname] = candidate_path
    resolved = dataclasses.replace(cfg, **resolved_paths)
    return resolved


def load_map_config(json_path: str, resolve_paths: bool = True) -> MapConfig:
    with open(json_path, 'r') as f:
        cfg = MapConfig(**json.load(f))
    if resolve_paths:
        cfg = resolve_paths_to_absolute(cfg, os.path.dirname(json_path))
    return cfg


def store_map_config(cfg: MapConfig, json_path: str, store_absolute_paths: bool = False) -> None:
    if not store_absolute_paths:
        cfg = dataclasses.replace(
            cfg, **{pathname: os.path.basename(getattr(cfg, pathname)) if getattr(cfg, pathname) is not None else None
                    for pathname in _filename_defaults('').keys()}
        )
    with open(json_path, 'w') as f:
        json.dump(dataclasses.asdict(cfg), f, indent=4)


def find_map_config(map_name: str, resolve_paths: bool = True) -> Optional[MapConfig]:
    """
    To retrieve configs for maps not bundled with the package,
    folders with corresponding names must be placed inside one of the directories
    listed in TDS_RESOURCE_PATH environment variable.
    Note that map names should be globally unique.
    """
    resource_path = torchdrivesim._resource_path
    for root in resource_path:
        map_path = os.path.join(root, map_name)
        if os.path.exists(map_path):
            break
    else:
        return None
    metadata_path = os.path.join(map_path, 'metadata.json')
    if os.path.exists(metadata_path):
        cfg = load_map_config(metadata_path)
    else:
        cfg = MapConfig(name=map_name)

    if resolve_paths:
        cfg = resolve_paths_to_absolute(cfg, root=map_path)

    return cfg


def download_iai_map(location_name: str, save_path: str) -> None:
    """
    Downloads the map information using LOCATION_INFO from the Inverted AI API.
    IAI_API_KEY needs to be set for this to succeed.
    Basename of save_path will be used as the map name for TorchDriveSim.
    If the dirpath of save_path is in TDS_RESOURCE_PATH, the map will be immediately
    available in `find_map_config`.
    """
    from invertedai import location_info
    info = location_info(location_name, include_map_source=True)
    os.makedirs(save_path, exist_ok=True)

    map_name = os.path.basename(save_path)
    pathname_defaults = _filename_defaults(map_name)
    center = info.map_center.x, info.map_center.y
    lanelet_path = os.path.join(save_path, pathname_defaults['lanelet_path'])
    info.osm_map.save_osm_file(lanelet_path)
    origin = info.osm_map.origin.x, info.osm_map.origin.y

    stoplines_path = os.path.join(save_path, pathname_defaults['stoplines_path'])
    stoplines = [
        dataclasses.asdict(Stopline(
            actor_id=sa.actor_id, agent_type=sa.agent_type, x=sa.center.x, y=sa.center.y,
            length=sa.length, width=sa.width, orientation=sa.orientation,
        )) for sa in info.static_actors
    ]
    with open(stoplines_path, 'w') as f:
        json.dump(stoplines, f, indent=4)

    cfg = MapConfig(
        name=map_name, center=center, lanelet_map_origin=origin,
        iai_location_name=location_name,
        left_handed_coordinates=location_name.split(':')[0] == 'carla',  # TODO: update once IAI API returns this info
        lanelet_path=os.path.abspath(lanelet_path),
    )
    mesh_path = os.path.join(save_path, pathname_defaults['mesh_path'])
    cfg.road_mesh.save(mesh_path)
    cfg.mesh_path = os.path.abspath(mesh_path)

    store_map_config(cfg, os.path.join(save_path, 'metadata.json'))
