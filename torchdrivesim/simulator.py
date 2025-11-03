import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union, Dict, List, Iterable, Callable, Any

from typing_extensions import Self

import numpy as np
import torch
from torch import Tensor

import torchdrivesim.rendering.pytorch3d
from torchdrivesim.goals import WaypointGoal
from torchdrivesim.kinematic import KinematicModel
from torchdrivesim.lanelet2 import LaneletMap, LaneFeatures
from torchdrivesim.mesh import generate_trajectory_mesh, BirdviewMesh, BirdviewRGBMeshGenerator
from torchdrivesim.rendering import BirdviewRenderer, RendererConfig, renderer_from_config
from torchdrivesim.infractions import offroad_infraction_loss, lanelet_orientation_loss, iou_differentiable, \
    compute_agent_collisions_metric_pytorch3d, compute_agent_collisions_metric, collision_detection_with_discs
from torchdrivesim.traffic_controls import BaseTrafficControl
from torchdrivesim.utils import Resolution, is_inside_polygon, isin, relative, assert_equal
from torchdrivesim.observation_noise import ObservationNoise, ObservationNoiseConfig

logger = logging.getLogger(__name__)


class CollisionMetric(Enum):
    """
    Method used to calculate collisions between agents.
    """
    iou = 'iou'  #: approximate differentiable IoU of oriented rectangles
    discs = 'discs'  #: differentiable overlap of oriented rectangles approximated as a union of circles
    nograd = 'nograd'  #: non-differentiable, exact IoU of oriented rectangles
    nograd_pytorch3d = 'nograd-pytorch3d'  #: non-differentiable IoU of oriented rectangles with pytorch3d


@dataclass
class TorchDriveConfig:
    """
    Top-level configuration for a TorchDriveSim simulator.
    """
    renderer: RendererConfig = field(default_factory=lambda:RendererConfig())  #: how to visualize the world, for the user and for the agents
    single_agent_rendering: bool = False  #: if set, agents don't see each other
    collision_metric: CollisionMetric = field(default_factory=lambda:CollisionMetric.discs)  #: method to use for computing collisions
    offroad_threshold: float = 0.5  #: how much the agents can go off-road without counting that as infraction
    left_handed_coordinates: bool = False  #: whether the coordinate system is left-handed (z always points upwards)
    wrong_way_angle_threshold: float = np.pi / 2  #: how far the agents can point away from the lane direction
        # without counting as infraction
    lanelet_inclusion_tolerance: float = 1.0  #: cars less than this many meters away from a lanelet boundary will still
        # be considered inside for the purposes of calculating the wrong way infractions
    waypoint_removal_threshold: float = 2.0  #: how close the agent needs to get to the waypoint to consider it achieved


class SpawnController:
    """
    Handles spawning and despawning of NPCs.
    If exit_boundary is provided, NPCs will be despawned if they are outside the boundary.
    If spawn_states and spawn_masks are provided, NPCs will be spawned at the specified states and masks if they're not already present.

    Args:
        exit_boundary: Bx2xN tensor, where N is the number of vertices of the polygon.
        spawn_states: BxAxTx4 tensor, where A is the number of NPCs, T is the number of timesteps.
        spawn_masks: BxAxT boolean tensor, where A is the number of NPCs, T is the number of timesteps.
    """
    def __init__(self, exit_boundary: Optional[Tensor] = None, spawn_states: Optional[Tensor] = None, spawn_masks: Optional[Tensor] = None):
        self.exit_boundary = exit_boundary
        self.spawn_states = spawn_states
        self.spawn_masks = spawn_masks
        self.time = 0

    def spawn_despawn_npcs(self, simulator: "Simulator") -> None:
        npc_present_mask = simulator.npc_controller.npc_present_mask
        npc_states = simulator.npc_controller.npc_state
        if self.exit_boundary is not None:
            npc_position = simulator.npc_controller.npc_state[..., :2]
            inside_boundary = is_inside_polygon(npc_position, self.exit_boundary)
            npc_present_mask = npc_present_mask & inside_boundary
        if self.spawn_states is not None and self.spawn_masks is not None:
            to_spawn = self.spawn_masks[..., self.time] & ~npc_present_mask
            npc_present_mask = npc_present_mask | to_spawn
            npc_states = self.spawn_states[..., self.time, :].where(to_spawn.unsqueeze(-1), npc_states)
        simulator.npc_controller.npc_present_mask = npc_present_mask
        simulator.npc_controller.npc_state = npc_states
        self.time += 1
        return None

    def to(self, device):
        if self.exit_boundary is not None:
            self.exit_boundary = self.exit_boundary.to(device)
        if self.spawn_states is not None:
            self.spawn_states = self.spawn_states.to(device)
        if self.spawn_masks is not None:
            self.spawn_masks = self.spawn_masks.to(device)
        return self

    def copy(self):
        return self.__class__(self.exit_boundary, self.spawn_states, self.spawn_masks)

    def extend(self, n, in_place=True):
        if not in_place:
            other = self.copy()
            other.extend(n, in_place=True)
            return other

        enlarge = lambda x: x.unsqueeze(1).expand((x.shape[0], n) + x.shape[1:]).reshape((n * x.shape[0],) + x.shape[1:])
        if self.exit_boundary is not None:
            self.exit_boundary = enlarge(self.exit_boundary)
        if self.spawn_states is not None:
            self.spawn_states = enlarge(self.spawn_states)
        if self.spawn_masks is not None:
            self.spawn_masks = enlarge(self.spawn_masks)
        return self

    def select_batch_elements(self, idx, in_place=True):
        if not in_place:
            return self.copy().select_batch_elements(idx, in_place=True)

        if self.exit_boundary is not None:
            self.exit_boundary = self.exit_boundary[idx]
        if self.spawn_states is not None:
            self.spawn_states = self.spawn_states[idx]
        if self.spawn_masks is not None:
            self.spawn_masks = self.spawn_masks[idx]
        return self



class NPCController:
    """
    Base class for non-playable character controllers. It leaves the state unchanged on each step.
    """
    def __init__(self, npc_size: Tensor, npc_state: Tensor, npc_present_mask: Optional[Tensor] = None,
                 npc_types: Optional[Tensor] = None, agent_type_names: Optional[List[str]] = None,
                 spawn_controller: Optional[SpawnController] = None):
        self.npc_size = npc_size
        self.npc_state = npc_state
        self.npc_present_mask = npc_present_mask
        if self.npc_present_mask is None:
            self.npc_present_mask = torch.ones_like(npc_state[..., 0], dtype=torch.bool)
        self.npc_types = npc_types
        if self.npc_types is None:
            self.npc_types = torch.zeros_like(npc_present_mask).long()
        self.agent_type_names = agent_type_names
        if self.agent_type_names is None:
            self.agent_type_names = ['vehicle']
        self.spawn_controller = spawn_controller
        if self.spawn_controller is None:
            self.spawn_controller = SpawnController()

    def get_npc_state(self):
        return self.npc_state

    def get_npc_size(self):
        return self.npc_size

    def get_npc_types(self):
        return self.npc_types

    def get_npc_present_mask(self):
        return self.npc_present_mask

    def spawn_despawn_npcs(self, simulator: "Simulator") -> None:
        self.spawn_controller.spawn_despawn_npcs(simulator)
        return None

    def advance_npcs(self, simulator: "Simulator") -> None:
        self.spawn_despawn_npcs(simulator)

    def to(self, device):
        self.npc_size = self.npc_size.to(device)
        self.npc_state = self.npc_state.to(device)
        self.npc_present_mask = self.npc_present_mask.to(device)
        self.npc_types = self.npc_types.to(device)
        self.spawn_controller.to(device)
        return self

    def copy(self):
        return self.__class__(self.npc_size, self.npc_state, self.npc_present_mask, self.npc_types, self.agent_type_names, self.spawn_controller.copy())

    def extend(self, n, in_place=True):
        if not in_place:
            other = self.copy()
            other.extend(n, in_place=True)
            return other

        enlarge = lambda x: x.unsqueeze(1).expand((x.shape[0], n) + x.shape[1:]).reshape((n * x.shape[0],) + x.shape[1:])
        self.npc_size = enlarge(self.npc_size)
        self.npc_state = enlarge(self.npc_state)
        self.npc_present_mask = enlarge(self.npc_present_mask)
        self.npc_types = enlarge(self.npc_types)
        self.spawn_controller.extend(n, in_place=True)
        return self

    def select_batch_elements(self, idx, in_place=True):
        if not in_place:
            return self.copy().select_batch_elements(idx, in_place=True)

        self.npc_size = self.npc_size[idx]
        self.npc_state = self.npc_state[idx]
        self.npc_present_mask = self.npc_present_mask[idx]
        self.npc_types = self.npc_types[idx]
        self.spawn_controller.select_batch_elements(idx, in_place=True)
        return self


class CompoundNPCController(NPCController):
    """
    Combines multiple NPC controllers by assigning each agent to one of the controllers.

    Args:
        controllers: List of NPCController objects
        controller_indices: BxA tensor of indices into the controllers list
    """
    def __init__(self, controllers: List[NPCController], controller_indices: Tensor):
        batch_size, num_agents = controller_indices.shape

        npc_size = torch.zeros((batch_size, num_agents, 2), device=controller_indices.device)
        npc_state = torch.zeros((batch_size, num_agents, 4), device=controller_indices.device)
        npc_present_mask = torch.zeros((batch_size, num_agents), device=controller_indices.device, dtype=torch.bool)
        npc_types = None if controllers[0].npc_types is None else torch.zeros((batch_size, num_agents), device=controller_indices.device, dtype=torch.long)

        super().__init__(npc_size, npc_state, npc_present_mask, npc_types, controllers[0].agent_type_names)
        self.controllers = controllers
        self.controller_indices = controller_indices

        self.gather_npc_states()


    def gather_npc_states(self):
        # Fill tensors based on controller_indices
        for i, controller in enumerate(self.controllers):
            mask = (self.controller_indices == i)
            self.npc_size = controller.npc_size.where(mask.unsqueeze(-1), self.npc_size)
            self.npc_state = controller.npc_state.where(mask.unsqueeze(-1), self.npc_state)
            self.npc_present_mask = controller.npc_present_mask.where(mask, self.npc_present_mask)
            if self.npc_types is not None:
                self.npc_types = controller.npc_types.where(mask, self.npc_types)

        # Propagate all npc states to the controllers
        for controller in self.controllers:
            controller.npc_size = self.npc_size
            controller.npc_state = self.npc_state
            controller.npc_present_mask = self.npc_present_mask
            if self.npc_types is not None:
                controller.npc_types = self.npc_types

    def advance_npcs(self, simulator: "Simulator") -> None:
        for controller in self.controllers:
            controller.advance_npcs(simulator)
        self.gather_npc_states()

    def to(self, device):
        super().to(device)
        self.controller_indices = self.controller_indices.to(device)
        for controller in self.controllers:
            controller.to(device)
        return self

    def copy(self):
        return self.__class__(
            [c.copy() for c in self.controllers],
            self.controller_indices.clone()
        )

    def extend(self, n, in_place=True):
        super().extend(n, in_place)
        self.controller_indices = self.controller_indices.expand(n, -1)
        for controller in self.controllers:
            controller.extend(n, in_place)
        return self

    def select_batch_elements(self, idx, in_place=True):
        super().select_batch_elements(idx, in_place)
        self.controller_indices = self.controller_indices[idx]
        for controller in self.controllers:
            controller.select_batch_elements(idx, in_place)
        return self


class Simulator:
    """

    Args:
        road_mesh: a mesh indicating the driveable area
        kinematic_model: determines the action space, constraints, and the initial state of all agents
        agent_size: a functor of Bx2 tensors indicating agent length and width
        initial_present_mask: a functor of BxA tensors indicating which agents are initially present and not padding
        cfg: holds various configuration options
        renderer: specify if using a non-standard renderer or static meshes beyond the road mesh (default from config)
        lanelet_map: provide the map to compute orientation losses, one map per batch element where available
        recenter_offset: if the coordinate system from lanelet_map was shifted, this value will be used to shift it back
        internal_time: initial value for step counter
        traffic_controls: applicable traffic controls by type
        waypoint_goals: waypoints for each agent
        agent_types: a tensor of BxA long tensors indicating the agent type index for each agent
        agent_type_names: a list of agent type names to index into
    """

    def __init__(self, road_mesh: BirdviewMesh, kinematic_model: KinematicModel,
                 agent_size: Tensor, initial_present_mask: Tensor,
                 cfg: TorchDriveConfig, renderer: Optional[BirdviewRenderer] = None,
                 lanelet_map: Optional[List[Optional[LaneletMap]]] = None, recenter_offset: Optional[Tensor] = None,
                 birdview_mesh_generator: Optional[BirdviewRGBMeshGenerator] = None,
                 internal_time: int = 0, traffic_controls: Optional[Dict[str, BaseTrafficControl]] = None,
                 waypoint_goals: Optional[WaypointGoal] = None,
                 agent_types: Optional[Tensor] = None, agent_type_names: Optional[List[str] ] = None,
                 npc_controller: Optional[NPCController] = None, agent_lr: Optional[Tensor] = None,
                 lane_features: Optional[LaneFeatures] = None, observation_noise_model: Optional[ObservationNoise] = None):
        self.road_mesh = road_mesh
        self.lanelet_map = lanelet_map
        self.recenter_offset = recenter_offset
        self.kinematic_model = kinematic_model
        self.agent_size = agent_size
        self.present_mask = initial_present_mask

        if not agent_type_names:
            agent_type_names = ['vehicle']
        if agent_types is None:
            agent_types = torch.zeros_like(initial_present_mask).long()
        if len(agent_types) == 1:
            agent_types = agent_types.expand_as(initial_present_mask)

        if agent_lr is None:
            agent_lr = torch.zeros_like(initial_present_mask).to(agent_size.dtype)
        if len(agent_lr) == 1:
            agent_lr = agent_lr.expand_as(initial_present_mask)

        self._agent_types = agent_type_names
        self._batch_size = self.road_mesh.batch_size
        self.agent_type = agent_types
        self.agent_lr = agent_lr

        self.lane_features = lane_features

        self.npc_controller = npc_controller
        if self.npc_controller is None:
            self.npc_controller = NPCController(
                npc_size=torch.zeros((self._batch_size, 0, 2), dtype=self.agent_size.dtype, device=initial_present_mask.device),
                npc_state=torch.zeros((self._batch_size, 0, 4), dtype=self.get_state().dtype, device=initial_present_mask.device),
                npc_present_mask=torch.zeros((self._batch_size, 0), dtype=torch.bool, device=initial_present_mask.device),
                npc_types=torch.zeros((self._batch_size, 0), dtype=torch.long, device=initial_present_mask.device),
                agent_type_names=agent_type_names,
            )


        self.validate_agent_types()
        self.validate_tensor_shapes()

        self.cfg: TorchDriveConfig = cfg
        if renderer is None:
            cfg.renderer.left_handed_coordinates = cfg.left_handed_coordinates
            self.renderer: BirdviewRenderer = renderer_from_config(cfg=cfg.renderer)
        else:
            self.renderer = renderer

        self.traffic_controls = traffic_controls
        self.waypoint_goals = waypoint_goals

        if cfg.left_handed_coordinates:
            self.kinematic_model.left_handed = cfg.left_handed_coordinates

        self.warned_no_lanelet = False
        self.internal_time = internal_time

        if birdview_mesh_generator is None:
            self.birdview_mesh_generator = BirdviewRGBMeshGenerator(background_mesh=self.road_mesh,
                                                                    color_map=self.renderer.color_map,
                                                                    rendering_levels=self.renderer.rendering_levels)
            self.birdview_mesh_generator.initialize_actors_mesh(self.get_all_agent_size(), self.get_all_agent_type(),
                                                                self.agent_types)
            if self.traffic_controls is not None:
                self.birdview_mesh_generator.initialize_traffic_controls_mesh(self.traffic_controls)
        else:
            self.birdview_mesh_generator = birdview_mesh_generator

        if observation_noise_model is None:
            self.observation_noise_model = ObservationNoise(ObservationNoiseConfig())
        else:
            self.observation_noise_model = observation_noise_model

    @property
    def agent_types(self) -> Optional[List[str]]:
        """
        List of agent types used by this simulator, or `None` if only one type used.
        """
        return self._agent_types

    @property
    def action_size(self) -> int:
        """
        Defines the size of the action space for each agent type.
        """
        return self.kinematic_model.action_size

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def to(self, device) -> Self:
        """
        Modifies the simulator in-place, putting all tensors on the device provided.
        """
        self.road_mesh = self.road_mesh.to(device)
        self.recenter_offset = self.recenter_offset.to(device) if self.recenter_offset is not None else None
        self.agent_size = self.agent_size.to(device)
        self.agent_type = self.agent_type.to(device)
        self.agent_lr = self.agent_lr.to(device)
        self.present_mask = self.present_mask.to(device)

        self.kinematic_model = self.kinematic_model.to(device)  # type: ignore
        self.traffic_controls = {k: v.to(device) for (k, v) in self.traffic_controls.items()} if self.traffic_controls is not None else None
        self.waypoint_goals = self.waypoint_goals.to(device) if self.waypoint_goals is not None else None
        self.birdview_mesh_generator = self.birdview_mesh_generator.to(device)
        self.npc_controller = self.npc_controller.to(device)

        self.lane_features = self.lane_features.to(device) if self.lane_features is not None else None
        return self

    def copy(self) -> Self:
        """
        Duplicates this simulator, allowing for independent subsequent execution.
        The copy is relatively shallow, in that the tensors are the same objects
        but dictionaries referring to them are shallowly copied.
        """
        other = self.__class__(
            road_mesh=self.road_mesh, kinematic_model=self.kinematic_model.copy(),
            agent_size=self.agent_size, initial_present_mask=self.present_mask,
            cfg=self.cfg, renderer=self.renderer.copy(), lanelet_map=self.lanelet_map,
            birdview_mesh_generator=self.birdview_mesh_generator.copy(),
            recenter_offset=self.recenter_offset, internal_time=self.internal_time,
            traffic_controls={k: v.copy() for k, v in self.traffic_controls.items()} if self.traffic_controls is not None else None,
            waypoint_goals=self.waypoint_goals.copy() if self.waypoint_goals is not None else None,
            agent_types=self.agent_type if self.agent_type is not None else None,
            agent_type_names=self.agent_types if self.agent_types is not None else None,
            agent_lr=self.agent_lr if self.agent_lr is not None else None,
            npc_controller=self.npc_controller.copy(),
            lane_features=self.lane_features.copy() if self.lane_features is not None else None,
            observation_noise_model=self.observation_noise_model,
        )
        return other

    def extend(self, n: int, in_place: bool = True) -> Self:
        """
        Multiplies the first batch dimension by the given number.
        Like in pytorch3d, this is equivalent to introducing extra batch
        dimension on the right and then flattening.
        """
        if not in_place:
            other = self.copy()
            other.extend(n, in_place=True)
            return other

        self.road_mesh = self.road_mesh.expand(n)
        enlarge = lambda x: x.unsqueeze(1).expand((x.shape[0], n) + x.shape[1:]).reshape((n * x.shape[0],) + x.shape[1:])
        self.agent_size = enlarge(self.agent_size)
        self.agent_type = enlarge(self.agent_type)
        self.agent_lr = enlarge(self.agent_lr)
        self.present_mask = enlarge(self.present_mask)
        self.recenter_offset = enlarge(self.recenter_offset) if self.recenter_offset is not None else None
        self.lanelet_map = [lanelet_map for lanelet_map in self.lanelet_map for _ in range(n)] if self.lanelet_map is not None else None

        self.lane_features = self.lane_features.extend(n) if self.lane_features is not None else None

        # kinematic models are modified in place
        self.kinematic_model.extend(n)
        self._batch_size *= n
        self.birdview_mesh_generator = self.birdview_mesh_generator.expand(n)
        if self.traffic_controls is not None:
            self.traffic_controls={k: v.extend(n) for k, v in self.traffic_controls.items()}

        if self.waypoint_goals is not None:
            self.waypoint_goals = self.waypoint_goals.extend(n)

        self.npc_controller = self.npc_controller.extend(n)

        return self

    def select_batch_elements(self, idx, in_place=True) -> Self:
        """
        Picks selected elements of the batch.
        The input is a tensor of indices into the batch dimension.
        """
        if not in_place:
            other = self.copy()
            other.select_batch_elements(idx, in_place=True)
            return other

        self.road_mesh = self.road_mesh[idx]
        self.recenter_offset = self.recenter_offset[idx] if self.recenter_offset is not None else None
        self.lanelet_map = [self.lanelet_map[i] for i in idx] if self.lanelet_map is not None else None
        self.agent_size = self.agent_size[idx]
        self.agent_type = self.agent_type[idx]
        self.agent_lr = self.agent_lr[idx]
        self.present_mask = self.present_mask[idx]

        self.lane_features = self.lane_features.select_batch_elements(idx) if self.lane_features is not None else None

        # kinematic models are modified in place
        self.kinematic_model.select_batch_elements(idx)

        self._batch_size = len(idx)
        self.birdview_mesh_generator = self.birdview_mesh_generator.select_batch_elements(idx)
        if self.traffic_controls is not None:
            self.traffic_controls={k: v.select_batch_elements(idx) for k, v in self.traffic_controls.items()}
        if self.waypoint_goals is not None:
            self.waypoint_goals = self.waypoint_goals.select_batch_elements(idx)

        self.npc_controller = self.npc_controller.select_batch_elements(idx)
        return self

    def __getitem__(self, item: Tensor) -> Self:
        """
        Allows indexing syntax. `item` should be an iterable collection of indices.
        """
        return self.select_batch_elements(item, in_place=False)

    @property
    def agent_count(self) -> int:
        """
        How many agents of each type are there in the simulation.
        This counts the available slots, not taking present masks into consideration.
        """
        return self.get_agent_size().shape[-2]

    @property
    def npc_count(self) -> int:
        """
        How many non-playable characters are there in the simulation.
        """
        return self.get_npc_size().shape[-2]

    def validate_agent_types(self):
        return  # nothing to check here anymore

    def validate_tensor_shapes(self):
        # check that tensors have the expected number of dimensions
        assert_equal(len(self.kinematic_model.get_state().shape), 3)
        assert_equal(len(self.agent_size.shape), 3)
        assert_equal(len(self.agent_type.shape), 2)
        assert_equal(len(self.agent_lr.shape), 2)
        assert_equal(len(self.present_mask.shape), 2)

        # check that batch size is the same everywhere
        b = self.batch_size
        assert_equal(self.road_mesh.batch_size, b)
        assert_equal(self.kinematic_model.get_state().shape[0], b)
        assert_equal(self.agent_size.shape[0], b)
        assert_equal(self.agent_type.shape[0], b)
        assert_equal(self.agent_lr.shape[0], b)
        assert_equal(self.present_mask.shape[0], b)

        # check that the number of agents is the same everywhere
        assert_equal(self.kinematic_model.get_state().shape[-2], self.agent_count)
        assert_equal(self.agent_size.shape[-2], self.agent_count)
        assert_equal(self.agent_type.shape[-1], self.agent_count)
        assert_equal(self.agent_lr.shape[-1], self.agent_count)
        assert_equal(self.present_mask.shape[-1], self.agent_count)

    def get_world_center(self) -> Tensor:
        """
        Returns a Bx2 tensor with the coordinates of the map center.
        """
        return self.birdview_mesh_generator.world_center

    def get_state(self) -> Tensor:
        """
        Returns a functor of BxAxSt tensors representing current agent states.
        """
        return self.kinematic_model.get_state()

    def get_waypoints(self, count: int = 1) -> Tensor:
        """
        Returns a functor of BxAxcount*Mx2 tensors representing current agent waypoints.
        """
        return self.waypoint_goals.get_waypoints(count=count) if self.waypoint_goals is not None else None

    def get_waypoints_state(self) -> Tensor:
        """
        Returns a functor of BxAx1 tensors representing current agent waypoints state.
        """
        return self.waypoint_goals.state if self.waypoint_goals is not None else None

    def get_waypoints_mask(self, count: int = 1) -> Tensor:
        """
        Returns a functor of BxAxcount*M boolean tensors representing current agent waypoints present mask.
        """
        return self.waypoint_goals.get_masks(count=count) if self.waypoint_goals is not None else None

    def compute_wrong_way(self) -> Tensor:
        """
        Wrong-way metric for each agent, based on the inner product between the agent and lane direction.
        See `torchdrivesim.infractions.lanelet_orientation_loss` for details.

        Returns:
            a functor of BxA tensors
        """
        if self.lanelet_map is not None:
            if isinstance(self.lanelet_map, Iterable) and None in self.lanelet_map and not self.warned_no_lanelet:
                idx_no_map = [i for i, item in enumerate(self.lanelet_map) if item is None]
                logger.debug(f"Batches {idx_no_map} have no lanelet map. Returning zeros for wrong_way losses.")
                self.warned_no_lanelet = True
            return lanelet_orientation_loss(
                self.lanelet_map, self.get_state(), self.recenter_offset,
                direction_angle_threshold=self.cfg.wrong_way_angle_threshold,
                lanelet_dist_tolerance=self.cfg.lanelet_inclusion_tolerance,
            ) * self.get_present_mask()
        else:
            if not self.warned_no_lanelet:
                logger.debug("No lanelet map is provided. Returning zeros for wrong_way losses.")
                self.warned_no_lanelet = True
            state = self.get_state()
            return torch.zeros(state.shape[0], state.shape[1]).to(state.device)

    def get_agent_size(self) -> Tensor:
        """
        Returns a functor of BxAx2 tensors representing agent length and width.
        """
        return self.agent_size

    def get_agent_type(self) -> Tensor:
        """
        Returns a functor of BxA long tensors containing agent type indexes relative to the list containing all agent types
            as returned by `Simulator.agent_types`.
        """
        return self.agent_type

    def get_agent_type_names(self) -> List[str]:
        """
        Returns a list of all agent types used in the simulation.
        """
        return self._agent_types

    def get_agent_lr(self) -> Tensor:
        """
        Returns a functor of BxA long tensors containing the rear offset
        """
        return self.agent_lr

    def get_present_mask(self) -> Tensor:
        """
        Returns a functor of BxA boolean tensors indicating which agents are currently present in the simulation.
        """
        return self.present_mask

    def get_noisy_state(self) -> Tensor:
        """
        Returns a functor of BxAx(A+Npc)xSt tensors representing current agent states.
        """
        return self.observation_noise_model.get_noisy_state(self)

    def get_noisy_agent_size(self) -> Tensor:
        """
        Returns a functor of BxAx(A+Npc)x2 tensors representing agent length and width.
        """
        return self.observation_noise_model.get_noisy_agent_size(self)

    def get_noisy_present_mask(self) -> Tensor:
        """
        Returns a functor of BxAx(A+Npc) boolean tensors indicating which agents are currently present in the simulation.
        """
        return self.observation_noise_model.get_noisy_present_mask(self)

    def get_npc_state(self) -> Tensor:
        """
        Returns a functor of BxNpcxSt tensors representing current non-playable character states.
        """
        return self.npc_controller.get_npc_state()

    def get_npc_size(self) -> Tensor:
        """
        Returns a functor of BxNpcx2 tensors representing non-playable character length and width.
        """
        return self.npc_controller.get_npc_size()

    def get_npc_present_mask(self) -> Tensor:
        """
        Returns a functor of BxNpc boolean tensors indicating which non-playable characters are currently present in the simulation.
        """
        return self.npc_controller.get_npc_present_mask()

    def get_npc_types(self) -> Tensor:
        """
        Returns a functor of BxNpc long tensors containing non-playable character type indexes relative to the list containing all agent types
            as returned by `Simulator.agent_types`.
        """
        return self.npc_controller.get_npc_types()

    def get_all_agent_state(self) -> Tensor:
        """
        Returns a functor of Bx(A+Npc)x4 tensors, where the last dimension contains the following information: x, y, psi, v.
        """
        return torch.cat([self.get_state(), self.get_npc_state()], dim=-2)

    def get_all_agent_size(self) -> Tensor:
        """
        Returns a functor of Bx(A+Npc)x2 tensors, where the last dimension contains the following information: length, width.
        """
        return torch.cat([self.get_agent_size(), self.get_npc_size()], dim=-2)

    def get_all_agent_present_mask(self) -> Tensor:
        """
        Returns a functor of Bx(A+Npc) boolean tensors, indicating which agents are currently present in the simulation.
        """
        return torch.cat([self.get_present_mask(), self.get_npc_present_mask()], dim=-1)

    def get_all_agent_type(self) -> Tensor:
        """
        Returns a functor of Bx(A+Npc) long tensors, indicating the agent type index for each agent.
        """
        return torch.cat([self.get_agent_type(), self.get_npc_types()], dim=-1)

    def get_all_agents_absolute(self) -> Tensor:
        """
        Returns a functor of Bx(A+Npc)x6 tensors,
        where the last dimension contains the following information: x, y, psi, length, width, present.
        Typically used to implement non-visual observation modalities.
        """
        agent_info = torch.cat([self.get_state()[..., :3], self.get_agent_size(), self.get_present_mask().unsqueeze(-1)], dim=-1)
        npc_info = torch.cat([self.get_npc_state()[..., :3], self.get_npc_size(), self.get_npc_present_mask().unsqueeze(-1)], dim=-1)
        return torch.cat([agent_info, npc_info], dim=-2)

    def get_noisy_all_agents_absolute(self) -> Tensor:
        """
        Returns a functor of BxAx(A+Npc)x6 tensors,
        where the last dimension contains the following information: x, y, psi, length, width, present.
        Typically used to implement non-visual observation modalities.
        """
        return torch.cat([self.get_noisy_state()[..., :3], self.get_noisy_agent_size(), self.get_noisy_present_mask()[..., None]], dim=-1)

    def get_all_agents_relative(self, exclude_self: bool = True) -> Tensor:
        """
        Returns a functor of BxAx(A+Npc)x6 tensors, specifying for each of A agents the relative position about
        the other agents. 'All' is the number of all agents in the simulation, including hidden ones, across all
        agent types. If `exclude_self` is set, for each agent in A, that agent itself is removed from All.
        The final dimension has the same meaning as in `get_all_agents_absolute`, except now the positions
        and orientations are relative to the specified agent.
        """
        abs_agent_pos = self.get_all_agents_absolute()
        all_agent_count = self.agent_count + self.npc_count

        xy, psi = abs_agent_pos[..., :self.agent_count, :2], abs_agent_pos[..., :self.agent_count, 2:3]  # current agent type
        all_xy, all_psi = abs_agent_pos[..., :2], abs_agent_pos[..., 2:3]  # all agent types
        # compute relative position of all agents w.r.t. each agent from of current type
        rel_xy, rel_psi = relative(origin_xy=xy.unsqueeze(-2), origin_psi=psi.unsqueeze(-2),
                                    target_xy=all_xy.unsqueeze(-3), target_psi=all_psi.unsqueeze(-3))
        rel_state = torch.cat([rel_xy, rel_psi], dim=-1)
        # insert the info that doesn't vary with the coordinate frame
        rel_pos = torch.cat([rel_state, abs_agent_pos[..., 3:].unsqueeze(-3).expand_as(rel_state)], dim=-1)
        if exclude_self:
            if self.agent_count == 1:
                rel_pos = rel_pos[..., 1:, :]
            else:
                # remove the diagonal of the current agent type
                # TODO: find a non-blocking version that's correct for multiple agents
                # indexing with a boolean mask triggers CUDA synchronization
                to_keep = torch.eye(self.agent_count, dtype=torch.bool, device=rel_pos.device).logical_not()
                to_keep = torch.cat([to_keep, torch.ones(self.agent_count, self.npc_count, dtype=torch.bool, device=rel_pos.device)], dim=-1)
                # need to flatten to index two dimensions simultaneously
                to_keep = torch.flatten(to_keep)
                rel_pos = rel_pos.flatten(start_dim=-3, end_dim=-2)
                rel_pos = rel_pos[..., to_keep, :]
                # the result has one less agent in the penultimate dimension
                rel_pos = rel_pos.reshape((*rel_pos.shape[:-2], self.agent_count, all_agent_count - 1, 6))
        return rel_pos

    def get_noisy_all_agents_relative(self, exclude_self: bool = True) -> Tensor:
        """
        Returns a functor of BxAx(A+Npc)x6 tensors, specifying for each of A agents the relative position about
        the other agents. 'All' is the number of all agents in the simulation, including hidden ones, across all
        agent types. If `exclude_self` is set, for each agent in A, that agent itself is removed from All.
        The final dimension has the same meaning as in `get_noisy_all_agents_absolute`, except now the positions
        and orientations are relative to the specified agent.
        """
        abs_agent_pos = self.get_noisy_all_agents_absolute()  # BxAx(A+Npc)x6
        all_agent_count = self.agent_count + self.npc_count
        agent_indices = torch.arange(self.agent_count, device=abs_agent_pos.device)
        agent_own_pos = abs_agent_pos[:, agent_indices, agent_indices, :]  # BxAx6
        # Agent's own position and orientation (origin for relative transformation)
        xy, psi = agent_own_pos[..., :2], agent_own_pos[..., 2:3]  # BxAx...
        # All entities as observed by each agent
        all_xy, all_psi = abs_agent_pos[..., :2], abs_agent_pos[..., 2:3]  # BxAx(A+Npc)x...
        # Compute relative position of all entities w.r.t. each agent
        rel_xy, rel_psi = relative(origin_xy=xy.unsqueeze(-2), origin_psi=psi.unsqueeze(-2),
                                   target_xy=all_xy, target_psi=all_psi)
        rel_state = torch.cat([rel_xy, rel_psi], dim=-1)  # BxAx(A+Npc)x3
        # Insert the info that doesn't vary with the coordinate frame
        rel_pos = torch.cat([rel_state, abs_agent_pos[..., 3:]], dim=-1)  # BxAx(A+Npc)x6
        if exclude_self:
            if self.agent_count == 1:
                rel_pos = rel_pos[..., 1:, :]
            else:
                # remove the diagonal of the current agent type
                # TODO: find a non-blocking version that's correct for multiple agents
                # indexing with a boolean mask triggers CUDA synchronization
                to_keep = torch.eye(self.agent_count, dtype=torch.bool, device=rel_pos.device).logical_not()
                to_keep = torch.cat([to_keep, torch.ones(self.agent_count, self.npc_count, dtype=torch.bool, device=rel_pos.device)], dim=-1)
                # need to flatten to index two dimensions simultaneously
                to_keep = torch.flatten(to_keep)
                rel_pos = rel_pos.flatten(start_dim=-3, end_dim=-2)
                rel_pos = rel_pos[..., to_keep, :]
                # the result has one less agent in the penultimate dimension
                rel_pos = rel_pos.reshape((*rel_pos.shape[:-2], self.agent_count, all_agent_count - 1, 6))
        return rel_pos

    def get_traffic_controls(self) -> Dict[str, BaseTrafficControl]:
        """
        Produces all traffic controls existing in the simulation, grouped by type.
        """
        return self.traffic_controls

    def step(self, agent_action: Tensor) -> None:
        """
        Runs the simulation for one step with given agent actions.
        Input is a functor of BxAxAc tensors, where Ac is determined by the kinematic model.
        """
        self.internal_time += 1
        # validate tensor shape lengths
        assert_equal(len(agent_action.shape), 3)
        # validate batch size
        assert_equal(agent_action.shape[0], self.batch_size)
        # validate agent numbers
        assert_equal(agent_action.shape[-2], self.agent_count)

        self.npc_controller.advance_npcs(self)
        self.kinematic_model.step(agent_action)

        if self.traffic_controls is not None:
            for traffic_control_type, traffic_control in self.traffic_controls.items():
                traffic_control.step(self.internal_time)
        if self.waypoint_goals is not None:
            self.waypoint_goals.step(self.get_state(), self.internal_time, threshold=self.cfg.waypoint_removal_threshold)

    def set_state(self, agent_state: Tensor, mask: Optional[Tensor] = None) -> None:
        """
        Arbitrarily set the state of the agents, without advancing the simulation.
        The change is effective immediately, without waiting for the next step.

        Args:
            agent_state: a functor of BxAx4 tensors with agent states
            mask: a functor of BxA boolean tensors, deciding which agent states to update; all by default
        """
        if mask is None:
            mask = torch.ones_like(agent_state[..., 0], dtype=torch.bool)
        # validate tensor shape lengths
        assert_equal(len(agent_state.shape), 3)
        assert_equal(len(mask.shape), 2)
        # validate batch size
        b = self.batch_size
        assert_equal(agent_state.shape[0], b)
        assert_equal(mask.shape[0], b)
        # validate agent numbers
        assert_equal(agent_state.shape[-2], self.agent_count)
        assert_equal(mask.shape[-1], self.agent_count)

        state_from_kinematic = self.kinematic_model.get_state()
        state_size, state_from_kinematic_size = agent_state.shape[-1], state_from_kinematic.shape[-1]
        assert state_size <= state_from_kinematic_size
        state = agent_state
        if state_size < state_from_kinematic_size:
            state = torch.cat([state, state_from_kinematic[..., (state_size-state_from_kinematic_size):]], dim=-1)
        new_state = state.where(mask.unsqueeze(-1).expand_as(agent_state), state_from_kinematic)
        self.kinematic_model.set_state(new_state)

    def update_present_mask(self, present_mask: Tensor) -> None:
        """
        Sets the present mask of agents to the provided value.

        Args:
            present_mask: a functor of BxA boolean tensors
        """
        assert_equal(len(present_mask.shape), 2)
        assert_equal(present_mask.shape[0], self.batch_size)
        assert_equal(present_mask.shape[-1], self.agent_count)

        self.present_mask = present_mask

    def fit_action(self, future_state: Tensor, current_state: Optional[Tensor] = None)\
            -> Tensor:
        """
        Computes an action that would (aproximately) produce the desired state.

        Args:
            future_state: a functor of BxAx4 tensors defining the desired state
            current_state: if different from the current simulation state, in the same format as future state
        Returns:
            a functor of BxAxAc tensors
        """
        return self.kinematic_model.fit_action(future_state=future_state, current_state=current_state)

    def render(self, camera_xy: Tensor, camera_psi: Tensor, res: Optional[Resolution] = None,
               rendering_mask: Optional[Tensor] = None, fov: Optional[float] = None,
               waypoints: Optional[Tensor] = None, waypoints_rendering_mask: Optional[Tensor] = None,
               custom_agent_colors: Optional[Tensor] = None) -> Tensor:
        """
        Renders the world from bird's eye view using cameras in given positions.

        Args:
            camera_xy: BxNx2 tensor of x-y positions for N cameras
            camera_psi: BxNx1 tensor of orientations for N cameras
            res: desired image resolution (only square resolutions are supported; by default use value from config)
            rendering_mask: functor of BxNxAll tensors, indicating which agents should be rendered each camera
            fov: the field of view of the resulting image in meters (by default use value from config)
            waypoints: BxNxMx2 tensor of `M` waypoints per camera (x,y)
            waypoints_rendering_mask: BxNxM tensor of `M` waypoint masks per camera,
                indicating which waypoints should be rendered
            custom_agent_colors: BxNxAllx3 RGB tensor defining the color of each agent to each camera
        Returns:
             BxNxCxHxW tensor of resulting RGB images for each camera
        """
        camera_sc = torch.cat([torch.sin(camera_psi), torch.cos(camera_psi)], dim=-1)
        if len(camera_xy.shape) == 2:
            # Reshape from Bx2 to Bx1x2
            camera_xy = camera_xy.unsqueeze(1)
            camera_sc = camera_sc.unsqueeze(1)
        n_cameras = camera_xy.shape[-2]
        target_shape = self.get_all_agent_present_mask().shape
        present_mask = self.get_all_agent_present_mask().unsqueeze(-2).expand(target_shape[:-1] + (n_cameras,) + target_shape[-1:])
        rendering_mask = present_mask if rendering_mask is None else present_mask.logical_and(rendering_mask)

        # TODO: we assume the same agent states for all cameras but we can give the option
        #       to pass different states for each camera.
        rbg_mesh = self.birdview_mesh_generator.generate(n_cameras,
            agent_state=self.get_all_agent_state()[:, None].expand(-1, n_cameras, -1, -1), present_mask=rendering_mask,
            traffic_lights=self.traffic_controls['traffic_light'].extend(n_cameras, in_place=False)
                if self.traffic_controls is not None and 'traffic_light' in self.traffic_controls else None,
            waypoints=waypoints, waypoints_rendering_mask=waypoints_rendering_mask,
            custom_agent_colors=custom_agent_colors,
        )
        return self.renderer.render_frame(rbg_mesh, camera_xy, camera_sc, res=res, fov=fov)

    def render_egocentric(self, ego_rotate: bool = True, res: Optional[Resolution] = None, fov: Optional[float] = None,
                          visibility_matrix: Optional[Tensor] = None, custom_agent_colors: Optional[Tensor] = None,
                          n_subsequent_waypoints: int = 1)\
            -> Tensor:
        """
        Renders the world using cameras placed on each agent.

        Args:
            ego_rotate: whether to orient the cameras such that the ego agent faces up in the image
            res: desired image resolution (only square resolutions are supported; by default use value from config)
            fov: the field of view of the resulting image in meters (by default use value from config)
            visibility_matrix: a BxAxAll boolean tensor indicating which agents can see each other
            custom_agent_colors: a BxAxAllx3 RGB tensor specifying what colors agent see each other as
            n_subsequent_waypoints: the number of subsequent waypoints to render
        Returns:
             a functor of BxAxCxHxW tensors of resulting RGB images for each agent.
        """
        camera_xy = self.get_state()[..., :2]
        camera_psi = self.get_state()[..., 2:3]
        waypoints = self.get_waypoints(count=n_subsequent_waypoints)
        if waypoints is not None:
            waypoints_mask = self.get_waypoints_mask(count=n_subsequent_waypoints)
        else:
            waypoints, waypoints_mask = None, None
        if not ego_rotate:
            camera_psi = torch.ones_like(camera_psi) * (np.pi / 2)
        rendering_mask = None
        if visibility_matrix is not None:
            rendering_mask = visibility_matrix
        if custom_agent_colors is not None:
            custom_agent_colors = custom_agent_colors
        if self.cfg.single_agent_rendering:
            rendering_mask = torch.eye(camera_xy[0].shape[1]).to(camera_xy.device).unsqueeze(0).expand(camera_xy[0].shape[0], -1, -1)

        bv = self.render(camera_xy, camera_psi, rendering_mask=rendering_mask, res=res, fov=fov,
                         waypoints=waypoints, waypoints_rendering_mask=waypoints_mask, custom_agent_colors=custom_agent_colors)
        total_agents = self.agent_count
        bv = bv.reshape((bv.shape[0] // total_agents, total_agents) + bv.shape[1:])
        return bv

    def compute_offroad(self) -> Tensor:
        """
        Offroad metric for each agent, defined as the distance to the road mesh.
        See `torchdrivesim.infractions.offroad_infraction_loss` for details.

        Returns:
            a functor of BxA tensors
        """
        return offroad_infraction_loss(self.get_state(), self.get_agent_size(),
                                       self.road_mesh, threshold=self.cfg.offroad_threshold) * self.get_present_mask()

    def compute_traffic_lights_violations(self) -> Tensor:
        """
        Boolean value indicating whether each agent is committing a traffic light violation.
        See `torchdrivesim.infractions.traffic_controls.TrafficLightControl.compute_violations` for details.

        Returns:
            a functor of BxA tensors
        """
        state = self.get_state()
        if self.get_traffic_controls() is not None and 'traffic_light' in self.get_traffic_controls():
            lenwid = self.get_agent_size()[..., :2]
            violation = self.get_traffic_controls()['traffic_light'].compute_violation(
                torch.cat([state[..., :2], lenwid, state[..., 2:3]], dim=-1)
            ) * self.get_present_mask().to(state.dtype)
        else:
            violation = torch.zeros(state.shape[0], state.shape[1], dtype=torch.bool, device=state.device)
        return violation

    def _compute_collision_of_single_agent(self, box: Tensor, remove_self_overlap: Optional[Tensor] = None, agent_types: Optional[List[str]] = None) -> Tensor:
        """
        Computes the collision metric for an agent specified as a bounding box.
        Includes collisions with all agents in the simulation,
        including the ones not exposed through the interface of this class.
        Used with `discs` and `iou` metrics.

        Args:
            box: Bx5 tensor, with the last dimension being (x,y,length,width,psi).
            remove_self_overlap: B boolean tensor, where if the input agent is present in the simulation,
                set this to subtract self-overlap. By default it is assumed that self overlapping exists and will be removed.
            agent_types: An optional list of specific agent types for computing collisions with.
                By default all available agent types will be used.
        Returns:
            a tensor with a single dimension of B elements
        """
        assert len(box.shape) == 2
        assert box.shape[0] == self.batch_size
        assert box.shape[-1] == 5

        states = self.get_all_agent_state()
        mask = self.get_all_agent_present_mask()
        if agent_types is not None:
            agent_types = [t for t in agent_types if t in self.agent_types]
            allowed_agent_type_indices = torch.tensor([self.agent_types.index(agent_type) for agent_type in agent_types], device=box.device)
            mask = mask.logical_and(torch.isin(self.get_all_agent_type(), allowed_agent_type_indices))
        if states.shape[-2] == 0:
            return torch.zeros_like(box[..., 0])
        sizes = self.get_all_agent_size()
        all_boxes = torch.cat([states[..., :2], sizes, states[..., 2:3]], dim=-1)  # TODO: cache this result
        expanded_box = box.unsqueeze(-2).expand_as(all_boxes)
        all_boxes = torch.nan_to_num(all_boxes, nan=0.0)
        expanded_box = torch.nan_to_num(expanded_box, nan=0.0)
        if self.cfg.collision_metric == CollisionMetric.iou:
            overlap = iou_differentiable(expanded_box, all_boxes)
        elif self.cfg.collision_metric == CollisionMetric.discs:
            overlap = collision_detection_with_discs(expanded_box, all_boxes)
        else:
            raise ValueError("Unrecognized collision metric: " + str(self.cfg.collision_metric))
        overlap = torch.nan_to_num(overlap, nan=0.0)
        overlap = overlap * mask.to(overlap.dtype)
        collision = overlap.sum(dim=-1)
        if remove_self_overlap is None:
            remove_self_overlap = torch.ones_like(collision)
        collision = collision - overlap.max(dim=-1)[0] * remove_self_overlap.to(collision.dtype)  # self-overlap is always highest
        return collision

    def _compute_collision_of_multi_agents(self, mask: Optional[Tensor] = None) -> Tensor:
        """
        Computes the collision metric for selected (default all) agents in the simulation.
        Includes collisions with all agents in the simulation,
        including the ones not exposed through the interface of this class.
        Used with `nograd` and `nograd-pytorch3d` metrics.

        Args:
            mask: a functor of BxA boolean tensors, indicating for which agents to compute the loss
                (by default use present mask)
        Returns:
            a functor of BxA tensors
        """
        # TODO: also compute collisions with NPCs
        collision_mask = self.get_present_mask() if mask is None else mask  # BxA
        states = self.get_state()
        sizes = self.get_agent_size()
        present_mask = self.get_present_mask()
        device = states.device

        if self.cfg.collision_metric == CollisionMetric.nograd:
            def build_presented_boxes(state, size):
                return np.concatenate([state[:, :2], size, state[:, 2:3]], axis=-1)

            def extract_presented():
                boxes, collision_masks = \
                    zip(*[(build_presented_boxes(states[batch][present_mask_i], sizes[batch][present_mask_i]),
                          (present_mask_i * collision_mask[batch])[present_mask_i])
                        for batch, present_mask_i in enumerate(present_mask)])
                return boxes, collision_masks

            present_mask = present_mask.cpu().detach().numpy()
            collision_mask = collision_mask.cpu().detach().numpy()
            states = states.cpu().detach().numpy()
            sizes = sizes.cpu().detach().numpy()
            all_presented_boxes, all_presented_collision_masks = extract_presented()
            collision = torch.tensor(
                compute_agent_collisions_metric(all_presented_boxes, all_presented_collision_masks, present_mask),
                device=device)
        elif self.cfg.collision_metric == CollisionMetric.nograd_pytorch3d:
            if not torchdrivesim.rendering.pytorch3d.is_available:
                raise torchdrivesim.rendering.pytorch3d.Pytorch3DNotFound(
                    "You can use a different collision metric, e.g. CollisionMetric.nograd"
                )
            all_boxes = torch.cat([states[..., :2], sizes, states[..., 2:3]], dim=-1)
            collision = compute_agent_collisions_metric_pytorch3d(all_boxes, collision_mask)
        else:
            raise ValueError("Unrecognized collision metric: " + str(self.cfg.collision_metric))
        return collision

    def compute_collision(self, agent_types: Optional[List[str]] = None) -> Tensor:
        """
        Compute the collision metric for agents exposed through the interface of this class.
        Includes collisions with agents not exposed through the interface.
        Collisions are defined as overlap of agents' bounding boxes, with details determined
        by the specific method chosen in the config.

        Args:
            agent_types: An optional list of specific agent types for computing collisions with. Not supported by
                the collision metrics `nograd` and `nograd-pytorch3d`.
        Returns:
            a BxA tensor
        """
        if self.cfg.collision_metric in [CollisionMetric.nograd, CollisionMetric.nograd_pytorch3d]:
            assert agent_types is None, 'The argument `agent_types` is not supported by the selected collision metric.'
            agent_collisions = self._compute_collision_of_multi_agents()
        else:
            state = self.get_state()
            size = self.get_agent_size()[..., :2]
            box = torch.cat([state[..., :2], size, state[..., 2:3]], dim=-1)
            agent_count = box.shape[-2]
            if agent_count == 0:
                return torch.zeros_like(box[..., 0])
            else:
                # TODO: batch across agent dimension
                collisions = []
                for i in range(box.shape[-2]):
                    remove_self_overlap = None
                    collision = self._compute_collision_of_single_agent(box[..., i, :],
                        remove_self_overlap=remove_self_overlap, agent_types=agent_types)
                    collisions.append(collision)
                agent_collisions = torch.stack(collisions, dim=-1)

        return agent_collisions
