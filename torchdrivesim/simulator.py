import abc
import logging
import os
import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union, Dict, List, Iterable, Callable, Any

import imageio
from typing_extensions import Self

import numpy as np
import torch
from torch import Tensor

import torchdrivesim.rendering.pytorch3d
from torchdrivesim.goals import WaypointGoal
from torchdrivesim.kinematic import KinematicModel
from torchdrivesim.lanelet2 import LaneletMap
from torchdrivesim.mesh import generate_trajectory_mesh, BirdviewMesh, BirdviewRGBMeshGenerator
from torchdrivesim.rendering import BirdviewRenderer, RendererConfig, renderer_from_config
from torchdrivesim.infractions import offroad_infraction_loss, lanelet_orientation_loss, iou_differentiable, \
    compute_agent_collisions_metric_pytorch3d, compute_agent_collisions_metric, collision_detection_with_discs
from torchdrivesim.traffic_controls import BaseTrafficControl
from torchdrivesim.utils import Resolution, is_inside_polygon, isin, relative, assert_equal

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


class SimulatorInterface(metaclass=abc.ABCMeta):
    """
    Abstract interface for a 2D differentiable driving simulator.
    """

    @property
    @abc.abstractmethod
    def agent_types(self) -> Optional[List[str]]:
        """
        List of agent types used by this simulator, or `None` if only one type used.
        """
        pass

    @property
    @abc.abstractmethod
    def action_size(self) -> int:
        """
        Defines the size of the action space for each agent type.
        """
        pass

    @property
    @abc.abstractmethod
    def batch_size(self) -> int:
        pass

    @abc.abstractmethod
    def to(self, device) -> Self:
        """
        Modifies the simulator in-place, putting all tensors on the device provided.
        """
        pass

    @abc.abstractmethod
    def copy(self) -> Self:
        """
        Duplicates this simulator, allowing for independent subsequent execution.
        The copy is relatively shallow, in that the tensors are the same objects
        but dictionaries referring to them are shallowly copied.
        """
        pass

    @abc.abstractmethod
    def extend(self, n: int, in_place: bool = True) -> Self:
        """
        Multiplies the first batch dimension by the given number.
        Like in pytorch3d, this is equivalent to introducing extra batch
        dimension on the right and then flattening.
        """
        pass

    @abc.abstractmethod
    def select_batch_elements(self, idx, in_place=True) -> Self:
        """
        Picks selected elements of the batch.
        The input is a tensor of indices into the batch dimension.
        """
        pass

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

    @abc.abstractmethod
    def get_world_center(self) -> Tensor:
        """
        Returns a Bx2 tensor with the coordinates of the map center.
        """
        pass

    @abc.abstractmethod
    def get_state(self) -> Tensor:
        """
        Returns a functor of BxAxSt tensors representing current agent states.
        """
        pass

    @abc.abstractmethod
    def get_agent_size(self) -> Tensor:
        """
        Returns a functor of BxAx2 tensors representing agent length and width.
        """
        pass

    @abc.abstractmethod
    def get_agent_type(self) -> Tensor:
        """
        Returns a functor of BxA long tensors containing agent type indexes relative to the list containing all agent types
            as returned by `SimulatorInterface.agent_types`.
        """
        pass

    @abc.abstractmethod
    def get_agent_type_names(self) -> List[str]:
        """
        Returns a list of all agent types used in the simulation.
        """
        pass

    @abc.abstractmethod
    def get_present_mask(self) -> Tensor:
        """
        Returns a functor of BxA boolean tensors indicating which agents are currently present in the simulation.
        """
        pass

    @abc.abstractmethod
    def get_all_agents_absolute(self) -> Tensor:
        """
        Returns a functor of BxAx6 tensors,
        where the last dimension contains the following information: x, y, psi, length, width, present.
        Typically used to implement non-visual observation modalities.
        """
        pass

    @abc.abstractmethod
    def get_all_agents_relative(self, exclude_self: bool = True) -> Tensor:
        """
        Returns a functor of BxAxAllx6 tensors, specifying for each of A agents the relative position about
        the other agents. 'All' is the number of all agents in the simulation, including hidden ones, across all
        agent types. If `exclude_self` is set, for each agent in A, that agent itself is removed from All.
        The final dimension has the same meaning as in `get_all_agents_absolute`, except now the positions
        and orientations are relative to the specified agent.
        """
        pass

    def get_traffic_controls(self) -> Dict[str, BaseTrafficControl]:
        """
        Produces all traffic controls existing in the simulation, grouped by type.
        """
        return self.get_innermost_simulator().traffic_controls

    @abc.abstractmethod
    def get_innermost_simulator(self) -> Self:
        """
        Returns the innermost concrete Simulator object.
        The type signature is misleading due to Python limitations.
        """
        pass

    @abc.abstractmethod
    def get_waypoints(self) -> Tensor:
        """
        Returns a functor of BxAxMx2 tensors representing current agent waypoints.
        """
        pass

    @abc.abstractmethod
    def get_waypoints_state(self) -> Tensor:
        """
        Returns a functor of BxAx1 tensors representing current agent waypoints state.
        """
        pass

    @abc.abstractmethod
    def get_waypoints_mask(self) -> Tensor:
        """
        Returns a functor of BxAxM boolean tensors representing current agent waypoints present mask.
        """
        pass

    @abc.abstractmethod
    def step(self, agent_action: Tensor) -> None:
        """
        Runs the simulation for one step with given agent actions.
        Input is a functor of BxAxAc tensors, where Ac is determined by the kinematic model.
        """
        pass

    @abc.abstractmethod
    def set_state(self, agent_state: Tensor, mask: Optional[Tensor] = None) -> None:
        """
        Arbitrarily set the state of the agents, without advancing the simulation.
        The change is effective immediately, without waiting for the next step.

        Args:
            agent_state: a functor of BxAx4 tensors with agent states
            mask: a functor of BxA boolean tensors, deciding which agent states to update; all by default
        """
        pass

    @abc.abstractmethod
    def update_present_mask(self, present_mask: Tensor) -> None:
        """
        Sets the present mask of agents to the provided value.

        Args:
            present_mask: a functor of BxA boolean tensors
        """
        pass

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
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
            rendering_mask: functor of BxNxA tensors, indicating which agents should be rendered each camera
            fov: the field of view of the resulting image in meters (by default use value from config)
            waypoints: BxNxMx2 tensor of `M` waypoints per camera (x,y)
            waypoints_rendering_mask: BxNxM tensor of `M` waypoint masks per camera,
                indicating which waypoints should be rendered
            custom_agent_colors: BxNxAx3 RGB tensor defining the color of each agent to each camera
        Returns:
             BxNxCxHxW tensor of resulting RGB images for each camera
        """
        pass

    def render_egocentric(self, ego_rotate: bool = True, res: Optional[Resolution] = None, fov: Optional[float] = None,
                          visibility_matrix: Optional[Tensor] = None, custom_agent_colors: Optional[Tensor] = None)\
            -> Tensor:
        """
        Renders the world using cameras placed on each agent.

        Args:
            ego_rotate: whether to orient the cameras such that the ego agent faces up in the image
            res: desired image resolution (only square resolutions are supported; by default use value from config)
            fov: the field of view of the resulting image in meters (by default use value from config)
            visibility_matrix: a BxAxA boolean tensor indicating which agents can see each other
            custom_agent_colors: a BxAxAx3 RGB tensor specifying what colors agent see each other as
        Returns:
             a functor of BxAxCxHxW tensors of resulting RGB images for each agent.
        """
        camera_xy = self.get_state()[..., :2]
        camera_psi = self.get_state()[..., 2:3]
        waypoints = self.get_waypoints()
        if waypoints is not None:
            waypoints_mask = self.get_waypoints_mask()
        else:
            waypoints, waypoints_mask = None, None
        if not ego_rotate:
            camera_psi = torch.ones_like(camera_psi) * (np.pi / 2)
        rendering_mask = None
        if visibility_matrix is not None:
            rendering_mask = visibility_matrix.flatten(0, 1)
        if custom_agent_colors is not None:
            custom_agent_colors = custom_agent_colors.flatten(0, 1)
        if self.get_innermost_simulator().cfg.single_agent_rendering:
            rendering_mask = torch.eye(camera_xy[0].shape[1]).to(camera_xy.device).unsqueeze(0).expand(camera_xy[0].shape[0], -1, -1)

        bv = self.render(camera_xy, camera_psi, rendering_mask=rendering_mask, res=res, fov=fov,
                         waypoints=waypoints, waypoints_rendering_mask=waypoints_mask, custom_agent_colors=custom_agent_colors)
        total_agents = self.agent_count
        bv = bv.reshape((bv.shape[0] // total_agents, total_agents) + bv.shape[1:])
        return bv

    @abc.abstractmethod
    def compute_offroad(self) -> Tensor:
        """
        Offroad metric for each agent, defined as the distance to the road mesh.
        See `torchdrivesim.infractions.offroad_infraction_loss` for details.

        Returns:
            a functor of BxA tensors
        """
        pass

    @abc.abstractmethod
    def compute_wrong_way(self) -> Tensor:
        """
        Wrong-way metric for each agent, based on the inner product between the agent and lane direction.
        See `torchdrivesim.infractions.lanelet_orientation_loss` for details.

        Returns:
            a functor of BxA tensors
        """
        pass

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

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
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
        return

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
        innermost_simulator = self.get_innermost_simulator()
        if innermost_simulator.cfg.collision_metric in [CollisionMetric.nograd, CollisionMetric.nograd_pytorch3d]:
            assert agent_types is None, 'The argument `agent_types` is not supported by the selected collision metric.'
            agent_collisions = self._compute_collision_of_multi_agents()
        else:
            state = self.get_state()
            size = self.get_agent_size()[..., :2]
            box = torch.cat([state[..., :2], size, state[..., 2:3]], dim=-1)
            box_type = self.get_agent_type()
            agent_count = box.shape[-2]
            if agent_count == 0:
                return torch.zeros_like(box[..., 0])
            else:
                # TODO: batch across agent dimension
                collisions = []
                for i in range(box.shape[-2]):
                    remove_self_overlap = None
                    collision = innermost_simulator._compute_collision_of_single_agent(box[..., i, :],
                        remove_self_overlap=remove_self_overlap, agent_types=agent_types)
                    collisions.append(collision)
                agent_collisions = torch.stack(collisions, dim=-1)

        return agent_collisions


class Simulator(SimulatorInterface):
    """
    Base simulator, where the agent functor is a dictionary indexed with agent type.

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
                 internal_time: int = 0, traffic_controls: Optional[Dict[str, BaseTrafficControl]] = None,
                 waypoint_goals: Optional[WaypointGoal] = None,
                 agent_types: Optional[Tensor] = None, agent_type_names: Optional[List[str] ] = None):
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

        self._agent_types = agent_type_names
        self._batch_size = self.road_mesh.batch_size
        self.agent_type = agent_types

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

        self._birdview_mesh_generator = BirdviewRGBMeshGenerator(background_mesh=self.road_mesh,
                                                                 color_map=self.renderer.color_map,
                                                                 rendering_levels=self.renderer.rendering_levels)
        self._birdview_mesh_generator.initialize_actors_mesh(self.agent_size, self.agent_type, self._agent_types)
        if self.traffic_controls is not None:
            self._birdview_mesh_generator.initialize_traffic_controls_mesh(self.traffic_controls)

    @property
    def agent_types(self):
        return self._agent_types

    @property
    def action_size(self) -> int:
        return self.kinematic_model.action_size

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def to(self, device):
        self.road_mesh = self.road_mesh.to(device)
        self.recenter_offset = self.recenter_offset.to(device) if self.recenter_offset is not None else None
        self.agent_size = self.agent_size.to(device)
        self.agent_type = self.agent_type.to(device)
        self.present_mask = self.present_mask.to(device)

        self.kinematic_model = self.kinematic_model.to(device)  # type: ignore
        self.traffic_controls = {k: v.to(device) for (k, v) in self.traffic_controls.items()} if self.traffic_controls is not None else None
        self.waypoint_goals = self.waypoint_goals.to(device) if self.waypoint_goals is not None else None
        self._birdview_mesh_generator = self._birdview_mesh_generator.to(device)

        return self

    def copy(self):
        other = self.__class__(
            road_mesh=self.road_mesh, kinematic_model=self.kinematic_model.copy(),
            agent_size=self.agent_size, initial_present_mask=self.present_mask,
            cfg=self.cfg, renderer=copy.deepcopy(self.renderer), lanelet_map=self.lanelet_map,
            recenter_offset=self.recenter_offset, internal_time=self.internal_time,
            traffic_controls={k: v.copy() for k, v in self.traffic_controls.items()} if self.traffic_controls is not None else None,
            waypoint_goals=self.waypoint_goals.copy() if self.waypoint_goals is not None else None,
            agent_types=self.agent_type if self.agent_type is not None else None,
            agent_type_names=self.agent_types if self.agent_types is not None else None
        )
        return other

    def extend(self, n, in_place=True):
        if not in_place:
            other = self.copy()
            other.extend(n, in_place=True)
            return other

        self.road_mesh = self.road_mesh.expand(n)
        enlarge = lambda x: x.unsqueeze(1).expand((x.shape[0], n) + x.shape[1:]).reshape((n * x.shape[0],) + x.shape[1:])
        self.agent_size = enlarge(self.agent_size)
        self.agent_type = enlarge(self.agent_type)
        self.present_mask = enlarge(self.present_mask)
        self.recenter_offset = enlarge(self.recenter_offset) if self.recenter_offset is not None else None
        self.lanelet_map = [lanelet_map for lanelet_map in self.lanelet_map for _ in range(n)] if self.lanelet_map is not None else None

        # kinematic models are modified in place
        self.kinematic_model.extend(n)
        self._batch_size *= n
        self._birdview_mesh_generator = self._birdview_mesh_generator.expand(n)
        if self.traffic_controls is not None:
            self.traffic_controls={k: v.extend(n) for k, v in self.traffic_controls.items()}

        if self.waypoint_goals is not None:
            self.waypoint_goals = self.waypoint_goals.extend(n)

        return self

    def select_batch_elements(self, idx, in_place=True):
        if not in_place:
            other = self.copy()
            other.select_batch_elements(idx, in_place=True)
            return other

        self.road_mesh = self.road_mesh[idx]
        self.recenter_offset = self.recenter_offset[idx] if self.recenter_offset is not None else None
        self.lanelet_map = [self.lanelet_map[i] for i in idx] if self.lanelet_map is not None else None
        self.agent_size = self.agent_size[idx]
        self.agent_type = self.agent_type[idx]
        self.present_mask = self.present_mask[idx]

        # kinematic models are modified in place
        self.kinematic_model.select_batch_elements(idx)

        self._batch_size = len(idx)
        self._birdview_mesh_generator = self._birdview_mesh_generator.select_batch_elements(idx)
        if self.traffic_controls is not None:
            self.traffic_controls={k: v.select_batch_elements(idx) for k, v in self.traffic_controls.items()}
        if self.waypoint_goals is not None:
            self.waypoint_goals = self.waypoint_goals.select_batch_elements(idx)
        return self

    def validate_agent_types(self):
        return  # nothing to check here anymore
        # check that all dicts have the same keys and iterate in the same order
        assert list(self.kinematic_model.keys()) == self.agent_types
        assert list(self.agent_size.keys()) == self.agent_types
        assert list(self.agent_type.keys()) == self.agent_types
        assert list(self.present_mask.keys()) == self.agent_types

    def validate_tensor_shapes(self):
        # check that tensors have the expected number of dimensions
        assert_equal(len(self.kinematic_model.get_state().shape), 3)
        assert_equal(len(self.agent_size.shape), 3)
        assert_equal(len(self.agent_type.shape), 2)
        assert_equal(len(self.present_mask.shape), 2)

        # check that batch size is the same everywhere
        b = self.batch_size
        assert_equal(self.road_mesh.batch_size, b)
        assert_equal(self.kinematic_model.get_state().shape[0], b)
        assert_equal(self.agent_size.shape[0], b)
        assert_equal(self.agent_type.shape[0], b)
        assert_equal(self.present_mask.shape[0], b)

        # check that the number of agents is the same everywhere
        assert_equal(self.kinematic_model.get_state().shape[-2], self.agent_count)
        assert_equal(self.agent_size.shape[-2], self.agent_count)
        assert_equal(self.agent_type.shape[-1], self.agent_count)
        assert_equal(self.present_mask.shape[-1], self.agent_count)

    def get_world_center(self):
        return self._birdview_mesh_generator.world_center

    def get_state(self):
        return self.kinematic_model.get_state()

    def get_waypoints(self):
        return self.waypoint_goals.get_waypoints() if self.waypoint_goals is not None else None

    def get_waypoints_state(self):
        return self.waypoint_goals.state if self.waypoint_goals is not None else None

    def get_waypoints_mask(self):
        return self.waypoint_goals.get_masks() if self.waypoint_goals is not None else None

    def compute_wrong_way(self):
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

    def get_agent_size(self):
        return self.agent_size

    def get_agent_type(self):
        return self.agent_type

    def get_agent_type_names(self) -> List[str]:
        return self._agent_types

    def get_present_mask(self):
        return self.present_mask

    def get_all_agents_absolute(self):
        return torch.cat([self.get_state()[..., :3], self.get_agent_size(), self.get_present_mask().unsqueeze(-1)], dim=-1)

    def get_all_agents_relative(self, exclude_self=True):
        abs_agent_pos = self.get_all_agents_absolute()
        all_agent_count = self.agent_count

        xy, psi = abs_agent_pos[..., :2], abs_agent_pos[..., 2:3]  # current agent type
        all_xy, all_psi = abs_agent_pos[..., :2], abs_agent_pos[..., 2:3]  # all agent types
        # compute relative position of all agents w.r.t. each agent from of current type
        rel_xy, rel_psi = relative(origin_xy=xy.unsqueeze(-2), origin_psi=psi.unsqueeze(-2),
                                    target_xy=all_xy.unsqueeze(-3), target_psi=all_psi.unsqueeze(-3))
        rel_state = torch.cat([rel_xy, rel_psi], dim=-1)
        # insert the info that doesn't vary with the coordinate frame
        rel_pos = torch.cat([rel_state, abs_agent_pos[..., 3:].unsqueeze(-3).expand_as(rel_state)], dim=-1)
        if exclude_self:
            # remove the diagonal of the current agent type
            to_keep = torch.eye(all_agent_count, dtype=torch.bool, device=rel_pos.device).logical_not()
            # need to flatten to index two dimensions simultaneously
            to_keep = torch.flatten(to_keep)
            rel_pos = rel_pos.flatten(start_dim=-3, end_dim=-2)
            rel_pos = rel_pos[..., to_keep, :]
            # the result has one less agent in the penultimate dimension
            rel_pos = rel_pos.reshape((*rel_pos.shape[:-2], all_agent_count, all_agent_count - 1, 6))
        return rel_pos

    def get_innermost_simulator(self) -> Self:
        return self

    def step(self, agent_action):
        self.internal_time += 1
        # validate tensor shape lengths
        assert_equal(len(agent_action.shape), 3)
        # validate batch size
        assert_equal(agent_action.shape[0], self.batch_size)
        # validate agent numbers
        assert_equal(agent_action.shape[-2], self.agent_count)

        self.kinematic_model.step(agent_action)

        if self.traffic_controls is not None:
            for traffic_control_type, traffic_control in self.traffic_controls.items():
                traffic_control.step(self.internal_time)
        if self.waypoint_goals is not None:
            self.waypoint_goals.step(self.get_state(), self.internal_time, threshold=self.cfg.waypoint_removal_threshold)

    def set_state(self, agent_state, mask=None):
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

    def update_present_mask(self, present_mask):
        assert_equal(len(present_mask.shape), 2)
        assert_equal(present_mask.shape[0], self.batch_size)
        assert_equal(present_mask.shape[-1], self.agent_count)

        self.present_mask = present_mask

    def fit_action(self, future_state, current_state=None):
        return self.kinematic_model.fit_action(future_state=future_state, current_state=current_state)

    def render(self, camera_xy, camera_psi, res=None, rendering_mask=None, fov=None,
               waypoints=None, waypoints_rendering_mask=None, custom_agent_colors=None):
        camera_sc = torch.cat([torch.sin(camera_psi), torch.cos(camera_psi)], dim=-1)
        if len(camera_xy.shape) == 2:
            # Reshape from Bx2 to Bx1x2
            camera_xy = camera_xy.unsqueeze(1)
            camera_sc = camera_sc.unsqueeze(1)
        n_cameras = camera_xy.shape[-2]
        target_shape = self.get_present_mask().shape
        present_mask = self.get_present_mask().unsqueeze(-2).expand(target_shape[:-1] + (n_cameras,) + target_shape[-1:])
        rendering_mask = present_mask if rendering_mask is None else present_mask.logical_and(rendering_mask)

        # TODO: we assume the same agent states for all cameras but we can give the option
        #       to pass different states for each camera.
        rbg_mesh = self._birdview_mesh_generator.generate(n_cameras,
            agent_state=self.get_state()[:, None].expand(-1, n_cameras, -1, -1), present_mask=rendering_mask,
            traffic_lights=self.traffic_controls['traffic_light'].extend(n_cameras, in_place=False)
                if self.traffic_controls is not None and 'traffic_light' in self.traffic_controls else None,
            waypoints=waypoints, waypoints_rendering_mask=waypoints_rendering_mask,
            custom_agent_colors=custom_agent_colors,
        )
        return self.renderer.render_frame(rbg_mesh, camera_xy, camera_sc, res=res, fov=fov)

    def compute_offroad(self):
        return offroad_infraction_loss(self.get_state(), self.get_agent_size(),
                                       self.road_mesh, threshold=self.cfg.offroad_threshold) * self.get_present_mask()

    def _compute_collision_of_single_agent(self, box, remove_self_overlap=None, agent_types=None):
        assert len(box.shape) == 2
        assert box.shape[0] == self.batch_size
        assert box.shape[-1] == 5

        states = self.get_state()
        mask = self.get_present_mask()
        if agent_types is not None:
            agent_types = [t for t in agent_types if t in self.agent_types]
            allowed_agent_type_indices = torch.tensor([self.agent_types.index(agent_type) for agent_type in agent_types], device=box.device)
            mask = mask.logical_and(torch.isin(self.get_agent_type(), allowed_agent_type_indices))
        if states.shape[-2] == 0:
            return torch.zeros_like(box[..., 0])
        sizes = self.get_agent_size()
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

    def _compute_collision_of_multi_agents(self, mask=None):
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


class SimulatorWrapper(SimulatorInterface):
    """
    Modifies the behavior of an existing simulator, itself acting like a simulator.
    This base class simply delegates all method calls to the inner simulator.
    """
    def __init__(self, simulator: SimulatorInterface):
        self.inner_simulator: SimulatorInterface = simulator
        self.cfg: TorchDriveConfig = self.get_innermost_simulator().cfg

    @property
    def agent_types(self):
        return self.inner_simulator.agent_types

    @property
    def action_size(self) -> int:
        return self.inner_simulator.action_size

    @property
    def batch_size(self) -> int:
        return self.inner_simulator.batch_size

    def to(self, device) -> Self:
        self.inner_simulator.to(device)
        return self

    def copy(self):
        inner_copy = self.inner_simulator.copy()
        new_object = self.__class__(inner_copy)
        return new_object

    def extend(self, n, in_place=True):
        if not in_place:
            other = self.copy()
            other.extend(n, in_place=True)
            return other
        self.inner_simulator.extend(n)
        return self

    def select_batch_elements(self, idx, in_place=True):
        if not in_place:
            other = self.copy()
            other = other.select_batch_elements(idx, in_place=True)
            return other

        self.inner_simulator = self.inner_simulator.select_batch_elements(idx, in_place=in_place)
        return self

    def get_world_center(self):
        return self.inner_simulator.get_world_center()

    def get_state(self):
        return self.inner_simulator.get_state()

    def get_agent_size(self):
        return self.inner_simulator.get_agent_size()

    def get_agent_type(self):
        return self.inner_simulator.get_agent_type()

    def get_agent_type_names(self) -> List[str]:
        return self.inner_simulator.get_agent_type_names()

    def get_present_mask(self):
        return self.inner_simulator.get_present_mask()

    def get_waypoints(self):
        return self.inner_simulator.get_waypoints()

    def get_waypoints_state(self):
        return self.inner_simulator.get_waypoints_state()

    def get_waypoints_mask(self):
        return self.inner_simulator.get_waypoints_mask()

    def get_all_agents_absolute(self):
        return self.inner_simulator.get_all_agents_absolute()

    def get_all_agents_relative(self, exclude_self=True):
        return self.inner_simulator.get_all_agents_relative(exclude_self=exclude_self)

    def get_innermost_simulator(self) -> Simulator:
        return self.inner_simulator.get_innermost_simulator()

    def step(self, *args, **kwargs):
        self.inner_simulator.step(*args, **kwargs)

    def set_state(self, *args, **kwargs):
        self.inner_simulator.set_state(*args, **kwargs)

    def update_present_mask(self, *args, **kwargs):
        self.inner_simulator.update_present_mask(*args, **kwargs)

    def fit_action(self, *args, **kwargs):
        return self.inner_simulator.fit_action(*args, **kwargs)

    def compute_offroad(self):
        return self.inner_simulator.compute_offroad()

    def _compute_collision_of_single_agent(self, box, remove_self_overlap=None, agent_types=None):
        return self.inner_simulator._compute_collision_of_single_agent(box, remove_self_overlap=remove_self_overlap, agent_types=agent_types)

    def _compute_collision_of_multi_agents(self, mask=None):
        return self.inner_simulator._compute_collision_of_multi_agents(mask)

    def compute_wrong_way(self):
        return self.inner_simulator.compute_wrong_way()

    def render(self, camera_xy, camera_psi, res=None, rendering_mask=None, fov=None, waypoints=None,
               waypoints_rendering_mask=None, custom_agent_colors=None):
        return self.inner_simulator.render(camera_xy, camera_psi, res=res, rendering_mask=rendering_mask, fov=fov,
                                           waypoints=waypoints, waypoints_rendering_mask=waypoints_rendering_mask, custom_agent_colors=custom_agent_colors)


class NPCWrapper(SimulatorWrapper):
    """
    Designates a certain subset of agents as non-playable characters (NPCs) and removes them from
    the simulator interface, although they remain in simulation and can be interacted with.
    Note that the designation of which agents are NPCs needs to be the same across batch elements.
    Subclasses implement specific policies controlling NPC behaviors.
    At a minimum, they should implement either `_get_npc_action` or `_npc_teleport_to`.

    Args:
        npc_mask: A functor of tensors with a single dimension of size A, indicating which agents to replay.
    """
    def __init__(self, simulator: SimulatorInterface, npc_mask: Tensor):
        super().__init__(simulator)
        self.npc_mask = npc_mask

    def _update_npc_present_mask(self) -> Tensor:
        """
        Computes updated present masks for NPCs, with arbitrary padding for the remaining agents.
        By default, leaves present masks unchanged.

        Returns:
            a functor of BxA boolean tensors, where A is the number of agents in the inner simulator
        """
        return self.inner_simulator.get_present_mask()

    def _get_npc_action(self) -> Tensor:
        """
        Computes the actions for NPCs, with arbitrary padding for actions of the remaining agents.
        By default, the actions are all zeros, but subclasses can implement more intelligent behavior.

        Returns:
            a functor of BxAxAc tensors, where A is the number of agents in the inner simulator
        """
        state = self.inner_simulator.get_state()
        return torch.zeros(size=state.shape[:-1] + (self.action_size,), dtype=state.dtype, device=state.device)

    def _npc_teleport_to(self) -> Optional[Tensor]:
        """
        Provides the states to which the NPCs should be set after `step`,
        with arbitrary padding for the remaining agents.
        By default, no teleportation is performed, but subclasses may use it instead of,
        or on top of defining the NPC action.

        Returns:
            a functor of BxAxSt tensors, where A is the number of agents in the inner simulator,
            or `None` if no teleportation is required
        """
        return None

    def to(self, device) -> Self:
        self.npc_mask = self.npc_mask.to(device)
        return super().to(device)

    def copy(self):
        inner_copy = self.inner_simulator.copy()
        other = self.__class__(inner_copy, npc_mask=self.npc_mask)
        return other

    def get_state(self):
        return self.inner_simulator.get_state()[..., self.npc_mask.logical_not(), :]

    def get_waypoints(self):
        waypoints = self.inner_simulator.get_waypoints()
        if waypoints is not None:
            waypoints = waypoints[..., self.npc_mask.logical_not(), :, :]
        return waypoints

    def get_waypoints_state(self):
        waypoints_state = self.inner_simulator.get_waypoints_state()
        if waypoints_state is not None:
            waypoints_state = waypoints_state[..., self.npc_mask.logical_not(), :]
        return waypoints_state

    def get_waypoints_mask(self):
        masks = self.inner_simulator.get_waypoints_mask()
        if masks is not None:
            masks = masks[..., self.npc_mask.logical_not(), :]
        return masks

    def get_agent_size(self):
        sizes = self.inner_simulator.get_agent_size()[..., self.npc_mask.logical_not(), :]
        return sizes

    def get_agent_type(self):
        sizes = self.inner_simulator.get_agent_type()[..., self.npc_mask.logical_not()]
        return sizes

    def get_present_mask(self):
        present_mask = self.inner_simulator.get_present_mask()[..., self.npc_mask.logical_not()]
        return present_mask

    def get_all_agents_relative(self, exclude_self=True):
        agent_info = self.inner_simulator.get_all_agents_relative(exclude_self=exclude_self)
        agent_info = agent_info[..., self.npc_mask.logical_not(), :, :]
        return agent_info

    def set_state(self, agent_state, mask=None):
        if mask is None:
            mask = torch.ones_like(agent_state[..., 0], dtype=torch.bool)

        # validate tensor shape lengths
        assert_equal(len(agent_state.shape), 3)
        assert_equal(len(mask.shape), 2)
        # validate batch shape
        b = self.batch_size
        assert_equal(agent_state.shape[0], b)
        assert_equal(mask.shape[0], b)
        # validate agent numbers
        assert_equal(agent_state.shape[-2], self.agent_count)
        assert_equal(mask.shape[-1], self.agent_count)

        old_state = self.inner_simulator.get_state()
        new_state = agent_state
        with_padding = torch.zeros_like(old_state)
        with_padding[..., torch.logical_not(self.npc_mask), :] = new_state  # I *think* autodiff handles this correctly
        selection_mask = self.extend_tensor(self.npc_mask, old_state.shape[:-2]).unsqueeze(-1).expand_as(old_state)
        updated_state = old_state.where(selection_mask, with_padding)
        states = updated_state

        current_mask = mask
        non_replay_mask = self.npc_mask.logical_not()
        full_mask = torch.zeros_like(non_replay_mask, dtype=torch.bool)
        full_mask = NPCWrapper.extend_tensor(full_mask, current_mask.shape[:-1]).clone()
        full_mask[..., non_replay_mask] = current_mask

        self.inner_simulator.set_state(states, mask=full_mask)

    def step(self, action):
        # validate tensor shape lengths
        assert_equal(len(action.shape), 3)
        # validate batch shape
        assert_equal(action.shape[0], self.batch_size)
        # validate agent numbers
        assert_equal(action.shape[-2], self.agent_count)

        # step all agents, with dummy action for replay agents
        npc_action = self._get_npc_action()
        mask = self.npc_mask
        given_action = action
        with_padding = torch.zeros_like(npc_action)
        with_padding[..., torch.logical_not(mask), :] = given_action
        selection_mask = self.extend_tensor(mask, npc_action.shape[:-2]).unsqueeze(-1).expand_as(npc_action)
        full_action = npc_action.where(selection_mask, with_padding)
        self.inner_simulator.step(full_action)

        # set target state for replay vehicles
        npc_state = self._npc_teleport_to()
        if npc_state is not None:
            full_npc_mask = self.npc_mask[None, :].expand_as(self.inner_simulator.get_present_mask())
            self.inner_simulator.set_state(npc_state, mask=full_npc_mask)

        # update presence mask of NPCs in case it changed
        non_replay_present_mask = self.inner_simulator.get_present_mask()[..., torch.logical_not(self.npc_mask)]
        self.update_present_mask(non_replay_present_mask)

    def update_present_mask(self, present_mask):
        assert_equal(len(present_mask.shape), 2)
        assert_equal(present_mask.shape[0], self.batch_size)
        assert_equal(present_mask.shape[-1], self.agent_count)

        recorded_present_mask = self._update_npc_present_mask()
        new_present_mask = recorded_present_mask.clone()
        new_present_mask[..., torch.logical_not(self.npc_mask)] = present_mask
        self.inner_simulator.update_present_mask(new_present_mask)

    # The commented out implementation is probably not useful, as it makes the NPC disappear from the rendering.
    # It seems more useful to specify the full visibility mask for all agents.
    # def render(self, camera_xy, camera_psi, res=None, rendering_mask=None, fov=None, waypoints=None,
    #            waypoints_rendering_mask=None):
    #     if rendering_mask is not None:
    #         new_mask = torch.zeros(rendering_mask.shape[0], rendering_mask.shape[1], self.npc_mask.shape[0], device=rendering_mask.device)
    #         new_mask[..., torch.logical_not(self.npc_mask)] = rendering_mask
    #         rendering_mask = new_mask
    #     return self.inner_simulator.render(camera_xy, camera_psi, res, rendering_mask, fov=fov, waypoints=waypoints,
    #                                        waypoints_rendering_mask=waypoints_rendering_mask)

    def fit_action(self, future_state, current_state=None):
        full_state = self.inner_simulator.get_state()
        full_future_state = full_state.clone()
        full_future_state[..., self.npc_mask.logical_not(), :] = future_state
        if current_state is None:
            full_current_state = None
        else:
            full_current_state = full_state.clone()
            full_current_state[..., self.npc_mask.logical_not(), :] = current_state

        full_action = self.inner_simulator.fit_action(full_future_state, full_current_state)
        action = full_action[..., self.npc_mask.logical_not(), :]
        return action

    def compute_offroad(self):
        offroad = offroad_infraction_loss(
            self.get_state(), self.get_agent_size(),
            self.get_innermost_simulator().road_mesh, threshold=self.cfg.offroad_threshold
        ) * self.get_present_mask()
        return offroad

    def compute_wrong_way(self):
        innermost_simulator = self.get_innermost_simulator()
        if innermost_simulator.lanelet_map is not None:
            if isinstance(innermost_simulator.lanelet_map, Iterable) and None in innermost_simulator.lanelet_map \
                    and not innermost_simulator.warned_no_lanelet:
                idx_no_map = [i for i, item in enumerate(innermost_simulator.lanelet_map) if item is None]
                logger.debug(f"Batches {idx_no_map} have no lanelet map. Returning zeros for wrong_way losses.")
                innermost_simulator.warned_no_lanelet = True
            return lanelet_orientation_loss(
                innermost_simulator.lanelet_map, self.get_state(), innermost_simulator.recenter_offset,
                direction_angle_threshold=self.cfg.wrong_way_angle_threshold,
                lanelet_dist_tolerance=self.cfg.lanelet_inclusion_tolerance,
            ) * self.get_present_mask()
        else:
            if not innermost_simulator.warned_no_lanelet:
                logger.debug("No lanelet map is provided. Returning zeros for wrong_way losses.")
                innermost_simulator.warned_no_lanelet = True
            return torch.zeros_like(self.get_state()[..., 0])

    def _compute_collision_of_multi_agents(self, mask=None):
        batched_non_replay_mask = ~self.npc_mask.expand(self.get_innermost_simulator().batch_size, -1)
        if mask is not None:
            batched_non_replay_mask = batched_non_replay_mask * mask
        collision = self.inner_simulator._compute_collision_of_multi_agents(batched_non_replay_mask)[..., self.npc_mask.logical_not()]
        return collision

    @staticmethod
    def extend_tensor(x, batch_dims):
        # add specified dimensions to the front of the tensor
        x_dims = x.shape
        for d in batch_dims:
            x.unsqueeze(0)
        return x.expand(batch_dims + x_dims)


class RecordingWrapper(SimulatorWrapper):
    """
    Doesn't modify the behavior of the simulator, but records some information at each time step.

    Args:
        record_functions: those functions are called on `self` after each `step` and their outputs are recorded
        initial_recording: whether to record values when constructing the object, before the first step
    """

    def __init__(self, simulator: SimulatorInterface, record_functions: Dict[str, Callable[[SimulatorInterface], Any]],
                 initial_recording: bool = True):
        super().__init__(simulator)

        self.record_functions = record_functions
        self.records = {record_name: [] for record_name in self.record_functions.keys()}

        if initial_recording:
            self.record()

    def copy(self):
        inner_copy = self.inner_simulator.copy()
        other = self.__class__(inner_copy, record_functions=dict())
        other.record_functions = self.record_functions
        other.records = {k: v.copy() for (k, v) in self.records.items()}
        return other

    def record(self):
        """
        Appends the results based on a current state to records.
        """
        for (k, f) in self.record_functions.items():
            self.records[k].append(f(self))

    def get_records(self):
        """
        Returns the dictionary of recorded values.
        Each entry is a list with results from subsequent invocations of `record`.
        """
        return self.records

    def step(self, agent_action, record=True):
        """
        By default performs recording at the end of step,
        but this can be disabled to manually control when to record.
        """
        self.inner_simulator.step(agent_action)
        if record:
            self.record()


class BirdviewRecordingWrapper(RecordingWrapper):
    """
    Records a visualization of the simulation state at each step.
    The visualization is obtained by calling `self.render`.

    Args:
        simulator: a simulator object to wrap
        res: resolution for the recorded images
        fov: field of view for the recorded images
        camera_xy: a Bx2 tensor of x-y camera positions (world center by default)
        camera_psi: a Bx1 tensor of camera orientations
        initial_recording: whether to record an image before the first step
        to_cpu: whether to move recorded images to CPU memory, releasing GPU memory
    """

    def __init__(self, simulator: SimulatorInterface, res: Resolution = Resolution(1024, 1024), fov: float = 100,
                 camera_xy: Optional[Tensor] = None, camera_psi: Optional[Tensor] = None,
                 initial_recording: bool = True, to_cpu: bool = False):
        self.res = res
        self.fov = fov
        self.camera_xy = camera_xy
        self.camera_psi = camera_psi
        self.to_cpu = to_cpu

        if self.camera_xy is None:
            self.camera_xy = simulator.get_world_center()
        if self.camera_psi is None:
            self.camera_psi = torch.ones_like(self.camera_xy[..., :1]) * np.pi / 2

        def record_birdview(simulator):
            s = simulator
            # TODO: figure out how to handle waypoints here
            waypoints = None # s.get_waypoints()
            if waypoints is not None:
                waypoints_mask = s.get_waypoints_mask()
            else:
                waypoints, waypoints_mask = None, None
            bv = s.render(s.camera_xy, s.camera_psi, res=self.res, fov=self.fov, waypoints=waypoints, waypoints_rendering_mask=waypoints_mask)
            if self.to_cpu:
                bv = bv.cpu()
            return bv

        record_functions = dict(birdview=record_birdview)
        super().__init__(simulator, record_functions, initial_recording)

    def to(self, device) -> Self:
        self.camera_xy = self.camera_xy.to(device)
        self.camera_psi = self.camera_psi.to(device)
        return super().to(device)

    def copy(self):
        inner_copy = self.inner_simulator.copy()
        other = self.__class__(inner_copy, res=self.res, camera_xy=self.camera_xy, camera_psi=self.camera_psi)
        other.record_functions = self.record_functions
        other.records = {k: v.copy() for (k, v) in self.records.items()}
        return other

    def extend(self, n, in_place=True):
        other = super().extend(n, in_place=in_place)
        f = lambda x: x.unsqueeze(1).expand((x.shape[0], n) + x.shape[1:]).reshape((n * x.shape[0],) + x.shape[1:])
        other.camera_xy = f(other.camera_xy)
        other.camera_psi = f(other.camera_psi)
        return other

    def select_batch_elements(self, idx, in_place=True):
        other = super().select_batch_elements(idx, in_place=in_place)
        other.camera_xy = other.camera_xy[idx]
        other.camera_psi = other.camera_psi[idx]
        return other

    def get_birdviews(self, stack: bool = False) -> Union[Tensor, List[Tensor]]:
        """
        Extracts recorded images.

        Args:
            stack: whether to concatenate results across time, which is placed after batch dimension
        Returns:
            a BxTxCxHxW tensor of RGB images if `stack` is set, a list of BxCxHxW images with the same content otherwise
        """
        bvs = self.get_records()['birdview']
        if stack:
            bvs = torch.stack(bvs, dim=-4)
        return bvs

    def save_gif(self, filename: str, batch_index: int = 0, fps: float = 10) -> None:
        """
        Saves a GIF to disk using all birdviews for a selected example from the batch.
        """
        bvs = self.get_birdviews()
        if os.path.dirname(filename) != '':
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        imageio.mimsave(
            filename, [bv[batch_index].floor().cpu().numpy().astype(np.uint8).transpose(1, 2, 0) for bv in bvs],
            format="GIF-PIL", fps=fps
        )
        try:
            from pygifsicle import optimize
            optimize(filename, options=['--no-warnings'])
        except ImportError:
            logger.info("You can install pygifsicle for gif compression and optimization.")

    def save_png(self, filename: str, frame: int,  batch_index: int = 0) -> None:
        """
        Saves the given frame as a PNG to disk.
        """
        bvs = self.get_birdviews()
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        imageio.imsave(
            filename, bvs[frame][batch_index].cpu().numpy().astype(np.uint8).transpose(1, 2, 0))


class TrajectoryVisualizationWrapper(BirdviewRecordingWrapper):
    """
    Records birdviews with additional trajectory visualizations.
    Those visualizations will be visible to agents relying on rendered observations,
    so typically this wrapper is used with replay states.
    The wrapper visualizes one ground truth trajectory and S predictions for each agent.
    Visualized trajectories should be concatenated across the agent dimension.

    Args:
        ground_truth: BxAxTxSt tensor with ground truth trajectories for all agents
        predictions: BxSxAxTxSt tensor with predicted trajectories
        gt_present_masks: BxSxAxT boolean tensor, indicating which points in predictions should be visualized
        predictions_present_masks: BxAxT boolean tensor, indicating which points in ground truth should be visualized
    """
    def __init__(self, simulator: SimulatorInterface, ground_truth: Tensor, predictions: Tensor,
                 gt_present_masks: Tensor, predictions_present_masks: Tensor,
                 trajectory_present_masks: Tensor, res: Resolution = Resolution(1024, 1024), fov: float = 100,
                 camera_xy: Optional[Tensor] = None, camera_psi: Optional[Tensor] = None, to_cpu: bool = False):
        assert_equal(ground_truth.shape[-3:], predictions.shape[-3:])
        assert len(ground_truth.shape) == 4
        assert len(predictions.shape) == 5
        assert ground_truth.shape[0] == predictions.shape[0]

        super().__init__(simulator, res, fov, camera_xy, camera_psi, to_cpu=to_cpu)

        # dummy state for agents not present so that they're not visible
        absent_state = torch.cat([self.get_world_center() + 1000,
                                  torch.zeros_like(self.get_world_center())], dim=-1)  # BxS
        # ground truth is only rendered when predictions are made and ground truth agent is present
        ground_truth = ground_truth.where(
            gt_present_masks.logical_and(predictions_present_masks)[..., None].expand_as(ground_truth),
            absent_state[:, None, None, :].expand_as(ground_truth)
        )
        # predictions are rendered even when ground truth agent is absent
        predictions = predictions.where(
            trajectory_present_masks[..., None].expand_as(predictions),
            absent_state[:, None, None, None, :].expand_as(predictions)
        )

        innermost_simulator = self.get_innermost_simulator()
        prediction_meshes = []
        for i in range(predictions.shape[1]):
            prediction_mesh = generate_trajectory_mesh(predictions[:, i], category='prediction')
            prediction_meshes.append(prediction_mesh)
        ground_truth_mesh = generate_trajectory_mesh(ground_truth, category='ground_truth')
        trajectory_meshes = [ground_truth_mesh, *prediction_meshes]
        innermost_simulator.renderer.add_static_meshes(trajectory_meshes)


class SelectiveWrapper(SimulatorWrapper):
    """
    Only exposes a subset of agents from the inner simulator.
    The selection may change over time, but there is a fixed number of agent slots available.
    Which slots are actually used is indicated by the presence mask.
    The base class exposes the first E agents and the
    children should override `update_exposed_agents` and `get_present_mask` to customize it.
    Typically, hidden agents are either absent or their state is set externally,
    but by default the `default_action` is executed for them.

    Args:
        simulator: existing simulator to wrap
        exposed_agent_limit: denoted as E in comments, sets the number of slots for exposed agents
        default_action: a functor of BxAxAc tensors, defining the default action to use for non-exposed agents
    """

    def __init__(self, simulator: SimulatorInterface, exposed_agent_limit: int, default_action: Tensor):
        super().__init__(simulator)
        self.default_action = default_action
        self.exposed_agent_limit = exposed_agent_limit
        self.exposed_agents: Tensor = None  # functor of BxE int tensor of indices of exposed agents
        self.update_exposed_agents()

    def update_exposed_agents(self) -> None:
        """
        There should always be E exposed agents, although some of them may be marked as absent.
        """
        device = self.get_world_center().device
        self.exposed_agents = torch.arange(self.exposed_agent_limit, dtype=torch.long, device=device).expand((self.batch_size, self.exposed_agent_limit))

    def get_exposed_agents(self) -> Tensor:
        """
        Returns:
            a functor of BxE int tensors of indices of exposed agents
        """
        return self.exposed_agents

    def is_exposed(self) -> Tensor:
        """
        Returns:
            a functor of BxA boolean tensors indicating which agents are exposed
        """
        exposed = torch.stack([isin(torch.arange(self.inner_simulator.agent_count).to(self.exposed_agents.device), self.exposed_agents[b])
                               for b in range(self.batch_size)])
        return exposed

    def _restrict_tensor(self, tensor: Tensor, agent_dim: int) -> Tensor:
        """
        Selects exposed agents from a given tensor, along the supplied agent dimension.
        """
        if agent_dim == -1:
            return tensor.gather(index=self.get_exposed_agents(), dim=agent_dim)
        elif agent_dim == -2:
            return tensor.gather(index=self.get_exposed_agents().unsqueeze(-1).expand(self.batch_size, self.exposed_agent_limit, tensor.shape[-1]),
                                dim=agent_dim)
        elif agent_dim == -3:
            return tensor.gather(index=self.get_exposed_agents().unsqueeze(-1).unsqueeze(-1).expand(
                self.batch_size, self.exposed_agent_limit, tensor.shape[-2], tensor.shape[-1]), dim=agent_dim)
        else:
            raise NotImplementedError

    def _extend_tensor(self, tensor: Tensor, padding: Tensor) -> Tensor:
        """
        Given a tensor of exposed agents, constructs a tensor of all agents, using supplied padding.
        Agent dimension should be the penultimate one.
        """
        selection = self.get_exposed_agents()
        extended = padding.clone()
        batch_idx = torch.arange(extended.shape[0]).unsqueeze(-1).repeat(1, selection.shape[-1])
        extended[batch_idx, selection] = tensor
        return extended

    def to(self, device) -> Self:
        self.default_action = self.default_action.to(device)
        if self.exposed_agents is not None:
            self.exposed_agents = self.exposed_agents.to(device)
        return super().to(device)

    def copy(self):
        inner_copy = self.inner_simulator.copy()
        other = self.__class__(inner_copy, exposed_agent_limit=self.agent_count, default_action=self.default_action)
        other.exposed_agents = self.exposed_agents
        return other

    def extend(self, n, in_place=True):
        extended = super().extend(n, in_place=in_place)
        enlarge = lambda x: x.unsqueeze(1).expand((x.shape[0], n) + x.shape[1:]).reshape((n * x.shape[0],) +
                                                                                         x.shape[1:])
        extended.default_action = enlarge(self.default_action)
        extended.exposed_agents = enlarge(self.exposed_agents)
        return extended

    def select_batch_elements(self, idx, in_place=True):
        other = super().select_batch_elements(idx, in_place=in_place)
        other.default_action = other.default_action[idx]
        if other.exposed_agents is not None:
            other.exposed_agents = other.exposed_agents[idx]
        return other

    def get_state(self):
        return self._restrict_tensor(self.inner_simulator.get_state(), agent_dim=-2)

    def get_present_mask(self):
        return torch.ones_like(self.get_exposed_agents(), dtype=torch.bool)

    def get_agent_size(self):
        return self._restrict_tensor(self.inner_simulator.get_agent_size(), agent_dim=-2)

    def get_agent_type(self):
        return self._restrict_tensor(self.inner_simulator.get_agent_type(), agent_dim=-1)

    def get_waypoints(self):
        waypoints = self.inner_simulator.get_waypoints()
        if waypoints is not None:
            waypoints = self._restrict_tensor(waypoints, agent_dim=-3)
        return waypoints

    def get_waypoints_state(self):
        waypoints_state = self.inner_simulator.get_waypoints_state()
        if waypoints_state is not None:
            waypoints_state = self._restrict_tensor(waypoints_state, agent_dim=-2)
        return waypoints_state

    def get_waypoints_mask(self):
        masks = self.inner_simulator.get_waypoints_mask()
        if masks is not None:
            masks = self._restrict_tensor(masks, agent_dim=-2)
        return masks

    def get_all_agents_absolute(self):
        return self._restrict_tensor(self.inner_simulator.get_all_agents_absolute(), agent_dim=-2)

    def get_all_agents_relative(self, exclude_self=True):
        return self._restrict_tensor(self.inner_simulator.get_all_agents_relative(exclude_self=exclude_self), agent_dim=-3)

    def set_state(self, agent_state, mask=None):
        full_state = self._extend_tensor(agent_state, self.inner_simulator.get_state())
        if mask is None:
            full_mask = None
        else:
            full_mask = self._extend_tensor(mask, self.is_exposed())
        self.inner_simulator.set_state(full_state, mask=full_mask)
        self.update_exposed_agents()

    def fit_action(self, future_state, current_state=None):
        extended_future_state = self._extend_tensor(future_state, padding=self.inner_simulator.get_state())
        if current_state is None:
            extended_current_state = None
        else:
            extended_current_state = self._extend_tensor(current_state, padding=self.inner_simulator.get_state())
        extended_action = self.inner_simulator.fit_action(extended_future_state, extended_current_state)
        action = self._restrict_tensor(extended_action, agent_dim=-2)
        return action

    def render(self, camera_xy, camera_psi, res=None, rendering_mask=None, fov=None, waypoints=None,
               waypoints_rendering_mask=None, custom_agent_colors=None):
        if rendering_mask is not None:
            rd_mask = rendering_mask.permute(0, 2, 1)
            pad_tensor = torch.zeros(rd_mask.shape[0], self.is_exposed().shape[1], rd_mask.shape[1], device=rd_mask.device)
            rendering_mask = self._extend_tensor(rd_mask, pad_tensor).permute(0, 2, 1)
        return self.inner_simulator.render(camera_xy, camera_psi, res, rendering_mask, fov=fov, waypoints=waypoints,
                                           waypoints_rendering_mask=waypoints_rendering_mask, custom_agent_colors=custom_agent_colors)

    def step(self, action):
        extended_action = self._extend_tensor(action, padding=self.default_action)
        super().step(extended_action)
        self.update_exposed_agents()

    def update_present_mask(self, *args, **kwargs):
        raise NotImplementedError("Updating present mask for SelectiveWrapper")

    def _compute_collision_of_multi_agents(self, mask=None):
        exposed_mask = self.is_exposed().reshape(self.get_innermost_simulator().batch_size, self.is_exposed().shape[-1])
        if mask is not None:
            exposed_mask = exposed_mask * mask
        return self.inner_simulator._compute_collision_of_multi_agents(exposed_mask)


class BoundedRegionWrapper(SelectiveWrapper):
    """
    A variant of `SelectiveWrapper` that exposes agents contained within a given polygon defining the area of interest.
    It additionally provides a method to determine which of the exposed agents have been "warmed-up",
    which means that they have been exposed for a certain number of time steps, but that is separate
    from the main functionality.

    Args:
        warmup_timesteps: after how many steps of being exposed are the agents considered "warmed-up"
        cutoff_polygon_verts: vertices defining the bounding convex polygon, in either clockwise or counter-closkwise
            order, provided as a functor of BxNx2 tensors, so that each agent type can use a different polygon
    """

    def __init__(self, simulator: SimulatorInterface, exposed_agent_limit: int, default_action: Tensor,
                 warmup_timesteps: int, cutoff_polygon_verts: Tensor):
        super().__init__(simulator, exposed_agent_limit, default_action)
        # Optional BxNx2 Tensor Collection for the polygon region to restrict agents,
        # the order can be either clockwise or anti-clockwise
        self.cutoff_polygon_verts = cutoff_polygon_verts
        # Scaling factor for the polygon region
        # After how many steps the agents are warmed-up
        self.warmup_timesteps = warmup_timesteps
        # Track for how many timesteps each agent has been in range
        self.proximal_timesteps = torch.zeros_like(
            self.inner_simulator.get_present_mask(), dtype=torch.long, device=simulator.get_world_center().device
        )  # BxA

        self.update_exposed_agents()

    def update_exposed_agents(self):
        if self.exposed_agents is None:
            super().update_exposed_agents()
            return  # for first run only

        location = self.inner_simulator.get_state()[..., :2]
        is_proximal = self.inner_simulator.get_present_mask()
        if self.cutoff_polygon_verts.shape[-2] > 0:
            is_proximal = is_proximal.logical_and(is_inside_polygon(location, self.cutoff_polygon_verts))
        self.proximal_timesteps = is_proximal * (is_proximal + self.proximal_timesteps)

        exposed = self.exposed_agents.clone()  # BxE
        is_proximal = self.proximal_timesteps > 0  # BxA
        is_to_expose = is_proximal.logical_and(self.is_exposed().logical_not())  # BxA
        for batch_idx in range(self.batch_size):
            # I couldn't find a way to avoid iterating over the batch dimension
            to_expose = is_to_expose[0].nonzero(as_tuple=False)
            available_slots = self.get_present_mask()[0].logical_not().nonzero(as_tuple=False)
            if to_expose.shape[0] > available_slots.shape[0]:
                logger.warning("More proximal agents than we can fit into exposed slots")
                to_expose = to_expose[:available_slots.shape[0]]
            exposed[batch_idx, available_slots[:to_expose.shape[0]]] = to_expose
        self.exposed_agents = exposed

    def get_present_mask(self):
        return self.proximal_timesteps.gather(dim=-1, index=self.exposed_agents) > 0

    def is_warmed_up(self):
        """
        Returns:
            a functor of BxA boolean tensors indicating which agents are warmed-up
        """
        return self.is_exposed().logical_and(self.proximal_timesteps > self.warmup_timesteps)

    def to(self, device) -> Self:
        self.proximal_timesteps = self.proximal_timesteps.to(device)
        if self.cutoff_polygon_verts is not None:
            self.cutoff_polygon_verts = self.cutoff_polygon_verts.to(device)
        return super().to(device)

    def copy(self):
        inner_copy = self.inner_simulator.copy()
        other = self.__class__(
            inner_copy, exposed_agent_limit=self.agent_count, default_action=self.default_action,
            warmup_timesteps=self.warmup_timesteps, cutoff_polygon_verts=self.cutoff_polygon_verts
        )
        other.exposed_agents = self.exposed_agents
        other.warmup_timesteps = self.warmup_timesteps
        other.proximal_timesteps = self.proximal_timesteps.clone()
        return other

    def extend(self, n, in_place=True):
        extended = super().extend(n, in_place=in_place)
        enlarge = lambda x: x.unsqueeze(1).expand((x.shape[0], n) + x.shape[1:]).\
            reshape((n * x.shape[0],) + x.shape[1:])
        extended.proximal_timesteps = enlarge(self.proximal_timesteps)
        if extended.cutoff_polygon_verts is not None:
            extended.cutoff_polygon_verts = enlarge(self.cutoff_polygon_verts)
        return extended

    def select_batch_elements(self, idx, in_place=True):
        other = super().select_batch_elements(idx, in_place=in_place)
        other.proximal_timesteps = other.proximal_timesteps[idx]
        if other.cutoff_polygon_verts is not None:
            other.cutoff_polygon_verts = other.cutoff_polygon_verts[idx]
        return other


class NoReentryBoundedRegionWrapper(BoundedRegionWrapper):
    """
    A variant of `BoundedRegionWrapper` that does not allow reentry.
    This means that if an agent is not exposed at a certain time, it will never be exposed in the future.
    However, the agents that can stop being exposed if they exit the area of interest.
    """

    def __init__(self, simulator: SimulatorInterface, exposed_agent_limit, default_action, warmup_timesteps,
                 cutoff_polygon_verts: Optional[Tensor] = None):
        self.previous_present_mask = simulator.get_present_mask()
        self.proximal_timesteps = None
        super().__init__(simulator, exposed_agent_limit, default_action,
                 warmup_timesteps, cutoff_polygon_verts,)

    def to(self, device) -> Self:
        self.previous_present_mask = self.previous_present_mask.to(device)
        if self.proximal_timesteps is not None:
            self.proximal_timesteps = self.proximal_timesteps.to(device)
        return super().to(device)

    def extend(self, n, in_place=True):
        extended = super().extend(n, in_place=in_place)
        enlarge = lambda x: x.unsqueeze(1).expand((x.shape[0], n) + x.shape[1:]).reshape((n * x.shape[0],) +
                                                                                         x.shape[1:])
        extended.previous_present_mask = enlarge(self.previous_present_mask)
        return extended

    def select_batch_elements(self, idx, in_place=True):
        other = super().select_batch_elements(idx, in_place=in_place)
        other.previous_present_mask = other.previous_present_mask[idx]
        return other

    def update_exposed_agents(self):
        super().update_exposed_agents()
        if self.proximal_timesteps is None:
            return  # for first run only
        self.proximal_timesteps = self.proximal_timesteps.mul(self.previous_present_mask)
        present_mask = self.get_present_mask()
        self.inner_simulator.update_present_mask(present_mask)
        self.previous_present_mask = present_mask.logical_and(self.previous_present_mask)

    def update_present_mask(self, present_mask):
        self.inner_simulator.update_present_mask(present_mask)
        self.previous_present_mask = present_mask

    def get_present_mask(self):
        present_mask = super().get_present_mask()

        return present_mask.logical_and(self.previous_present_mask)
