"""
A simple example demonstrating how to convert a Lanelet2 map into a mesh used by TorchDriveSim.
A trivial Lanelet2 map is constructed here, but the same procedure works with externally provided maps.
"""
from omegaconf import OmegaConf
from dataclasses import dataclass
from torchdrivesim.lanelet2 import lanelet_map_to_lane_mesh, road_mesh_from_lanelet_map
from torchdrivesim.mesh import BirdviewMesh
import lanelet2
import sys


@dataclass
class LaneletToMeshConfig:
    maps_file_path: str
    mesh_save_path: str


def revert_map(lanelet_map):
    reverted_map = lanelet2.core.LaneletMap()
    for lanelet in lanelet_map.laneletLayer:
        left_boundary = [p for p in lanelet.leftBound]
        right_boundary = [p for p in lanelet.rightBound]
        if len(left_boundary) > 0 and len(right_boundary) > 0:
            left_boundary = lanelet2.core.LineString3d(lanelet2.core.getId(), left_boundary)
            right_boundary = lanelet2.core.LineString3d(lanelet2.core.getId(), right_boundary)
            # The lanelets in the originally generated maps were inverted linestrings.
            new_lanelet = lanelet2.core.Lanelet(lanelet2.core.getId(), left_boundary, right_boundary)
            new_lanelet = new_lanelet.invert()
            new_lanelet.attributes['type'] = lanelet.attributes['type']
            new_lanelet.attributes['subtype'] = lanelet.attributes['subtype']
            new_lanelet.attributes['is_intersection'] = lanelet.attributes['is_intersection']
            reverted_map.add(new_lanelet)

    return reverted_map


def build_map_mesh(cfg: LaneletToMeshConfig):
    map_file_path, mesh_save_path = cfg.maps_file_path, cfg.mesh_save_path
    origin = (0, 0)
    projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(*origin))
    lanelet_map = lanelet2.io.load(map_file_path, projector)
    lanelet_map = revert_map(lanelet_map)  # Fixing for Carla left-handed coordinates
    road_mesh = road_mesh_from_lanelet_map(lanelet_map)
    road_mesh = BirdviewMesh.set_properties(road_mesh, category='road').to(road_mesh.device)
    lane_mesh = lanelet_map_to_lane_mesh(
        lanelet_map,
        left_handed=True)
    driving_surface_mesh = lane_mesh.merge(road_mesh)
    driving_surface_mesh.pickle(mesh_save_path)


if __name__ == '__main__':
    cli_cfg: LaneletToMeshConfig = OmegaConf.structured(
        LaneletToMeshConfig(**OmegaConf.from_dotlist(sys.argv[1:]))
    )
    build_map_mesh(cli_cfg)  # type: ignore
