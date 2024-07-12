import abc
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from itertools import accumulate
from typing import Optional, Union, Dict, List, Iterable, Callable, Any

import imageio
from typing_extensions import Self

import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import pad

import torchdrivesim.rendering.pytorch3d
from torchdrivesim.goals import WaypointGoal
from torchdrivesim.kinematic import KinematicModel
from torchdrivesim.lanelet2 import LaneletMap
from torchdrivesim.mesh import generate_trajectory_mesh, BirdviewMesh
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


# the type system in Python is not sufficiently expressive to parameterize those types,
# so the following is a sensible compromise
TensorPerAgentType = Union[Tensor, Dict[str, Tensor]]
IntPerAgentType = Union[int, Dict[str, int]]


class AgentTypeFunctor(abc.ABC):
    """
    Lifts functions operating on individual agent type to functions operating across all agent types.
    Children of this class specify how the collection of agents is represented.
    """
    @abc.abstractmethod
    def fmap(self, f, *args):
        """
        Applies a given function of any number of arguments to all agent types.
        """
        pass

    def to_device(self, tensor: TensorPerAgentType, device: torch.device) -> TensorPerAgentType:
        """
        Applies `.to` to each element.
        """
        return self.fmap(lambda x: x.to(device), tensor)

    def __call__(self, f, *args):
        return self.fmap(f, *args)


class SingletonAgentTypeFunctor(AgentTypeFunctor):
    """
    Trivial functor where only one agent type is used.
    No packaging is used for the arguments.
    """
    def fmap(self, f, *args):
        return f(*args)


class DictAgentTypeFunctor(AgentTypeFunctor):
    """
    Arguments are packaged as dictionaries, with keys being agent type names.
    """
    def __init__(self, agent_types: List[str]):
        super().__init__()
        self.agent_types = agent_types

    def fmap(self, f, *args):
        result = dict()
        for key in self.agent_types:
            try:
                result[key] = f(*[d[key] for d in args])
            except KeyError:
                logger.debug(f"Missing agent type {key} - ingored")
        return result


class SimulatorInterface(metaclass=abc.ABCMeta):
    """
    Abstract interface for a 2D differentiable driving simulator.
    """
    @property
    @abc.abstractmethod
    def agent_functor(self) -> AgentTypeFunctor:
        """
        Defines how to apply functions across agent types.
        """
        pass

    @property
    @abc.abstractmethod
    def agent_types(self) -> Optional[List[str]]:
        """
        List of agent types used by this simulator, or `None` if only one type used.
        """
        pass

    def across_agent_types(self, f, *args):
        """
        Applies a given per-agent-type operation across all agent types.
        Subsequent arguments should be given in the format of multiple agent
        collection that this class uses, which is typically a dictionary
        with keys being agent type names.
        """
        return self.agent_functor(f, *args)

    @property
    @abc.abstractmethod
    def action_size(self) -> IntPerAgentType:
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
    def agent_count(self) -> IntPerAgentType:
        """
        How many agents of each type are there in the simulation.
        This counts the available slots, not taking present masks into consideration.
        """
        return self.across_agent_types(lambda s: s.shape[-2], self.get_agent_size())

    @abc.abstractmethod
    def get_world_center(self) -> Tensor:
        """
        Returns a Bx2 tensor with the coordinates of the map center.
        """
        pass

    @abc.abstractmethod
    def get_state(self) -> TensorPerAgentType:
        """
        Returns a functor of BxAxSt tensors representing current agent states.
        """
        pass

    @abc.abstractmethod
    def get_agent_size(self) -> TensorPerAgentType:
        """
        Returns a functor of BxAx2 tensors representing agent length and width.
        """
        pass

    @abc.abstractmethod
    def get_agent_type(self) -> TensorPerAgentType:
        """
        Returns a functor of BxA long tensors containing agent type indexes relative to the list containing all agent types
            as returned by `SimulatorInterface.agent_types`.
        """
        pass

    @abc.abstractmethod
    def get_present_mask(self) -> TensorPerAgentType:
        """
        Returns a functor of BxA boolean tensors indicating which agents are currently present in the simulation.
        """
        pass

    @abc.abstractmethod
    def get_all_agents_absolute(self) -> TensorPerAgentType:
        """
        Returns a functor of BxAx6 tensors,
        where the last dimension contains the following information: x, y, psi, length, width, present.
        Typically used to implement non-visual observation modalities.
        """
        pass

    @abc.abstractmethod
    def get_all_agents_relative(self, exclude_self: bool = True) -> TensorPerAgentType:
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
    def get_waypoints(self) -> TensorPerAgentType:
        """
        Returns a functor of BxAxMx2 tensors representing current agent waypoints.
        """
        pass

    @abc.abstractmethod
    def get_waypoints_state(self) -> TensorPerAgentType:
        """
        Returns a functor of BxAx1 tensors representing current agent waypoints state.
        """
        pass

    @abc.abstractmethod
    def get_waypoints_mask(self) -> TensorPerAgentType:
        """
        Returns a functor of BxAxM boolean tensors representing current agent waypoints present mask.
        """
        pass

    @abc.abstractmethod
    def step(self, agent_action: TensorPerAgentType) -> None:
        """
        Runs the simulation for one step with given agent actions.
        Input is a functor of BxAxAc tensors, where Ac is determined by the kinematic model.
        """
        pass

    @abc.abstractmethod
    def set_state(self, agent_state: TensorPerAgentType, mask: Optional[TensorPerAgentType] = None) -> None:
        """
        Arbitrarily set the state of the agents, without advancing the simulation.
        The change is effective immediately, without waiting for the next step.

        Args:
            agent_state: a functor of BxAx4 tensors with agent states
            mask: a functor of BxA boolean tensors, deciding which agent states to update; all by default
        """
        pass

    @abc.abstractmethod
    def update_present_mask(self, present_mask: TensorPerAgentType) -> None:
        """
        Sets the present mask of agents to the provided value.

        Args:
            present_mask: a functor of BxA boolean tensors
        """
        pass

    @abc.abstractmethod
    def fit_action(self, future_state: TensorPerAgentType, current_state: Optional[TensorPerAgentType] = None)\
            -> TensorPerAgentType:
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
               rendering_mask: Optional[TensorPerAgentType] = None, fov: Optional[float] = None,
               waypoints: Optional[Tensor] = None, waypoints_rendering_mask: Optional[Tensor] = None) -> Tensor:
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
        Returns:
             BxNxCxHxW tensor of resulting RGB images for each camera
        """
        pass

    def render_egocentric(self, ego_rotate: bool = True, res: Optional[Resolution] = None, fov: Optional[float] = None)\
            -> TensorPerAgentType:
        """
        Renders the world using cameras placed on each agent.

        Args:
            ego_rotate: whether to orient the cameras such that the ego agent faces up in the image
            res: desired image resolution (only square resolutions are supported; by default use value from config)
            fov: the field of view of the resulting image in meters (by default use value from config)
        Returns:
             a functor of BxAxCxHxW tensors of resulting RGB images for each agent.
        """
        # compute camera positions
        agent_pos = self.across_agent_types(lambda s: (s[..., :2], s[..., 2:3]), self.get_state())
        agent_count = self.agent_count
        singleton = isinstance(self.agent_functor, SingletonAgentTypeFunctor)
        agent_pos_dict = dict(agent=agent_pos) if singleton else agent_pos
        agent_count_dict = dict(agent=agent_count) if singleton else agent_count
        camera_xy = torch.cat([s[0] for s in agent_pos_dict.values()], dim=-2)
        camera_psi = torch.cat([s[1] for s in agent_pos_dict.values()], dim=-2)
        waypoints = self.get_waypoints()
        if waypoints is not None:
            waypoints_mask = self.get_waypoints_mask()
            waypoints_dict = dict(agent=waypoints) if singleton else waypoints
            waypoints_mask_dict = dict(agent=waypoints_mask) if singleton else waypoints_mask
            waypoints = torch.cat(list(waypoints_dict.values()), dim=-3)
            waypoints_mask = torch.cat(list(waypoints_mask_dict.values()), dim=-2)
        else:
            waypoints, waypoints_mask = None, None
        if not ego_rotate:
            camera_psi = torch.ones_like(camera_psi) * (np.pi / 2)
        # render birdview and split resulting tensor across agent types
        rendering_mask = None
        if self.get_innermost_simulator().cfg.single_agent_rendering:
            agent_types_starting_idx = list(accumulate([0] + list(agent_count_dict.values())))[:-1]
            agent_rel_starting_idx = {agent_type: agent_types_starting_idx[i]
                                      for i, agent_type in enumerate(agent_count_dict.keys())}
            if singleton:
                agent_rel_starting_idx = 0
            rendering_mask = self.across_agent_types(
                lambda a_pos, a_count, a_st_idx:
                pad(
                    torch.eye(a_pos[0].shape[1]).to(camera_xy.device),
                    (0, 0, a_st_idx, camera_xy.shape[1]-a_st_idx-a_count)
                ).unsqueeze(0).expand(a_pos[0].shape[0], -1, -1)
                if a_count > 0 else
                torch.zeros(a_pos[0].shape[0], camera_xy.shape[1], 0).to(camera_xy.device),
                agent_pos, agent_count, agent_rel_starting_idx)  # BxNcxA where Nc=A

        bv = self.render(camera_xy, camera_psi, rendering_mask=rendering_mask, res=res, fov=fov,
                         waypoints=waypoints, waypoints_rendering_mask=waypoints_mask)
        chunks = [0] + list(np.cumsum(list(agent_count_dict.values())))
        total_agents = chunks[-1]
        bv = bv.reshape((bv.shape[0] // total_agents, total_agents) + bv.shape[1:])
        bv_dict = {agent_type: bv[..., chunks[i]:chunks[i+1], :, :, :]
                   for (i, agent_type) in enumerate(agent_pos_dict.keys())}
        if singleton:
            bv_dict = bv_dict['agent']
        return bv_dict

    @abc.abstractmethod
    def compute_offroad(self) -> TensorPerAgentType:
        """
        Offroad metric for each agent, defined as the distance to the road mesh.
        See `torchdrivesim.infractions.offroad_infraction_loss` for details.

        Returns:
            a functor of BxA tensors
        """
        pass

    @abc.abstractmethod
    def compute_wrong_way(self) -> TensorPerAgentType:
        """
        Wrong-way metric for each agent, based on the inner product between the agent and lane direction.
        See `torchdrivesim.infractions.lanelet_orientation_loss` for details.

        Returns:
            a functor of BxA tensors
        """
        pass

    def compute_traffic_lights_violations(self) -> TensorPerAgentType:
        """
        Boolean value indicating whether each agent is committing a traffic light violation.
        See `torchdrivesim.infractions.traffic_controls.TrafficLightControl.compute_violations` for details.

        Returns:
            a functor of BxA tensors
        """
        if self.get_traffic_controls() is not None and 'traffic_light' in self.get_traffic_controls():
            violation = self.across_agent_types(
                lambda state, lenwid, mask: self.get_traffic_controls()['traffic_light'].compute_violation(
                    torch.cat([state[..., :2], lenwid, state[..., 2:3]], dim=-1)
                ) * mask.to(state.dtype),
                self.get_state(), self.get_agent_size(), self.get_present_mask(),
            )
        else:
            violation = self.across_agent_types(
                lambda state, lenwid, mask: torch.zeros(state.shape[0], state.shape[1],
                                                        dtype=torch.bool, device=state.device),
                self.get_state(), self.get_agent_size(), self.get_present_mask(),
            )
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
    def _compute_collision_of_multi_agents(self, mask: Optional[Tensor] = None) -> TensorPerAgentType:
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

    def compute_collision(self, agent_types: Optional[List[str]] = None) -> TensorPerAgentType:
        """
        Compute the collision metric for agents exposed through the interface of this class.
        Includes collisions with agents not exposed through the interface.
        Collisions are defined as overlap of agents' bounding boxes, with details determined
        by the specific method chosen in the config.

        Args:
            agent_types: An optional list of specific agent types for computing collisions with. Not supported by
                the collision metrics `nograd` and `nograd-pytorch3d`.
        Returns:
            a functor of BxA tensors
        """
        innermost_simulator = self.get_innermost_simulator()
        if innermost_simulator.cfg.collision_metric in [CollisionMetric.nograd, CollisionMetric.nograd_pytorch3d]:
            assert agent_types is None, 'The argument `agent_types` is not supported by the selected collision metric.'
            agent_collisions = self._compute_collision_of_multi_agents()
        else:
            def f(box, box_type):
                agent_count = box.shape[-2]
                if agent_count == 0:
                    return torch.zeros_like(box[..., 0])
                else:
                    # TODO: batch across agent dimension
                    collisions = []
                    for i in range(box.shape[-2]):
                        remove_self_overlap = None
                        if agent_types is not None:
                            remove_self_overlap = torch.tensor([innermost_simulator._agent_types[a_type_idx] in agent_types
                                                            for a_type_idx in box_type[..., i].flatten()], device=box_type.device)
                            remove_self_overlap = remove_self_overlap.reshape(box_type[..., i].shape)
                        collision = innermost_simulator._compute_collision_of_single_agent(box[..., i, :],
                            remove_self_overlap=remove_self_overlap, agent_types=agent_types)
                        collisions.append(collision)
                    return torch.stack(collisions, dim=-1)

            agent_box = self.across_agent_types(
                lambda state, size: torch.cat([state[..., :2], size, state[..., 2:3]], dim=-1),
                self.get_state(), self.get_agent_size()
            )
            agent_collisions = self.across_agent_types(
                f, agent_box, self.get_agent_type()
            )

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
    """

    def __init__(self, road_mesh: BirdviewMesh, kinematic_model: Dict[str, KinematicModel],
                 agent_size: Dict[str, Tensor], initial_present_mask: Dict[str, Tensor],
                 cfg: TorchDriveConfig, renderer: Optional[BirdviewRenderer] = None,
                 lanelet_map: Optional[List[Optional[LaneletMap]]] = None, recenter_offset: Optional[Tensor] = None,
                 internal_time: int = 0, traffic_controls: Optional[Dict[str, BaseTrafficControl]] = None,
                 waypoint_goals: Optional[WaypointGoal] = None):
        self.road_mesh = road_mesh
        self.lanelet_map = lanelet_map
        self.recenter_offset = recenter_offset
        self.kinematic_model = kinematic_model
        self.agent_size = agent_size
        self.present_mask = initial_present_mask

        self._agent_types = list(self.kinematic_model.keys())
        self._batch_size = self.road_mesh.batch_size
        self.agent_type = {a_type: torch.ones_like(a_size[..., 0]).long()*idx for idx, (a_type, a_size) in enumerate(agent_size.items())}

        self.validate_agent_types()
        self.validate_tensor_shapes()

        self.cfg: TorchDriveConfig = cfg
        if renderer is None:
            cfg.renderer.left_handed_coordinates = cfg.left_handed_coordinates
            self.renderer: BirdviewRenderer = renderer_from_config(
                cfg=cfg.renderer, static_mesh=self.road_mesh
            )
        else:
            #  We assume the provided renderer has all static meshes already added to avoid increasing the size of the
            #  static mesh with duplicate additional static meshes (e.g. lane markings).
            self.renderer = renderer

        self.traffic_controls = traffic_controls
        self.waypoint_goals = waypoint_goals

        if cfg.left_handed_coordinates:
            def set_left_handed(kin):
                kin.left_handed = cfg.left_handed_coordinates

            self.across_agent_types(set_left_handed, self.kinematic_model)

        self.warned_no_lanelet = False
        self.internal_time = internal_time

    @property
    def agent_functor(self) -> DictAgentTypeFunctor:
        return DictAgentTypeFunctor(agent_types=self.agent_types)

    @property
    def agent_types(self):
        return self._agent_types

    @property
    def action_size(self) -> IntPerAgentType:
        return self.across_agent_types(
            lambda kin: kin.action_size, self.kinematic_model
        )

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def to(self, device):
        self.road_mesh = self.road_mesh.to(device)
        self.recenter_offset = self.recenter_offset.to(device) if self.recenter_offset is not None else None
        self.agent_size = self.agent_functor.to_device(self.agent_size, device)
        self.agent_type = self.agent_functor.to_device(self.agent_type, device)
        self.present_mask = self.agent_functor.to_device(self.present_mask, device)

        self.kinematic_model = self.agent_functor.to_device(self.kinematic_model, device)  # type: ignore
        self.traffic_controls = {k: v.to(device) for (k, v) in self.traffic_controls.items()} if self.traffic_controls is not None else None
        self.waypoint_goals = self.waypoint_goals.to(device) if self.waypoint_goals is not None else None
        self.renderer = self.renderer.to(device)

        return self

    def copy(self):
        other = self.__class__(
            road_mesh=self.road_mesh, kinematic_model=self.across_agent_types(lambda k: k.copy(), self.kinematic_model),
            agent_size=self.agent_size, initial_present_mask=self.present_mask,
            cfg=self.cfg, renderer=self.renderer.copy(), lanelet_map=self.lanelet_map,
            recenter_offset=self.recenter_offset, internal_time=self.internal_time,
            traffic_controls={k: v.copy() for k, v in self.traffic_controls.items()} if self.traffic_controls is not None else None,
            waypoint_goals=self.waypoint_goals.copy() if self.waypoint_goals is not None else None
        )
        return other

    def extend(self, n, in_place=True):
        if not in_place:
            other = self.copy()
            other.extend(n, in_place=True)
            return other

        self.road_mesh = self.road_mesh.expand(n)
        enlarge = lambda x: x.unsqueeze(1).expand((x.shape[0], n) + x.shape[1:]).reshape((n * x.shape[0],) + x.shape[1:])
        self.agent_size = self.across_agent_types(enlarge, self.agent_size)
        self.agent_type = self.across_agent_types(enlarge, self.agent_type)
        self.present_mask = self.across_agent_types(enlarge, self.present_mask)
        self.recenter_offset = enlarge(self.recenter_offset) if self.recenter_offset is not None else None
        self.lanelet_map = [lanelet_map for lanelet_map in self.lanelet_map for _ in range(n)] if self.lanelet_map is not None else None

        # kinematic models are expanded in place
        def expand_kinematic(kin):
            kin.map_param(enlarge)
            kin.set_state(enlarge(kin.get_state()))

        self.across_agent_types(expand_kinematic, self.kinematic_model)
        self._batch_size *= n
        self.renderer = self.renderer.expand(n)
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
        self.agent_size = self.across_agent_types(
            lambda x: x[idx], self.agent_size
        )
        self.agent_type = self.across_agent_types(
            lambda x: x[idx], self.agent_type
        )
        self.present_mask = self.across_agent_types(
            lambda x: x[idx], self.present_mask
        )

        # kinematic models are modified in place
        def select_batch_kinematic(kin):
            kin.map_param(lambda x: x[idx])
            kin.set_state(kin.get_state()[idx])
        self.across_agent_types(
            select_batch_kinematic, self.kinematic_model
        )

        self._batch_size = len(idx)
        self.renderer = self.renderer.select_batch_elements(idx)
        if self.traffic_controls is not None:
            self.traffic_controls={k: v.select_batch_elements(idx) for k, v in self.traffic_controls.items()}
        if self.waypoint_goals is not None:
            self.waypoint_goals = self.waypoint_goals.select_batch_elements(idx)
        return self

    def validate_agent_types(self):
        # check that all dicts have the same keys and iterate in the same order
        assert list(self.kinematic_model.keys()) == self.agent_types
        assert list(self.agent_size.keys()) == self.agent_types
        assert list(self.agent_type.keys()) == self.agent_types
        assert list(self.present_mask.keys()) == self.agent_types

    def validate_tensor_shapes(self):
        # check that tensors have the expected number of dimensions
        self.across_agent_types(lambda kin: assert_equal(len(kin.get_state().shape), 3), self.kinematic_model)
        self.across_agent_types(lambda s: assert_equal(len(s.shape), 3), self.agent_size)
        self.across_agent_types(lambda s: assert_equal(len(s.shape), 2), self.agent_type)
        self.across_agent_types(lambda m: assert_equal(len(m.shape), 2), self.present_mask)

        # check that batch size is the same everywhere
        b = self.batch_size
        assert_equal(self.road_mesh.batch_size, b)
        self.across_agent_types(lambda kin: assert_equal(kin.get_state().shape[0], b), self.kinematic_model)
        self.across_agent_types(lambda s: assert_equal(s.shape[0], b), self.agent_size)
        self.across_agent_types(lambda s: assert_equal(s.shape[0], b), self.agent_type)
        self.across_agent_types(lambda m: assert_equal(m.shape[0], b), self.present_mask)

        # check that the number of agents is the same everywhere
        self.validate_agent_count(self.across_agent_types(
            lambda kin: kin.get_state().shape[-2], self.kinematic_model))
        self.validate_agent_count(self.across_agent_types(lambda s: s.shape[-2], self.agent_size))
        self.validate_agent_count(self.across_agent_types(lambda s: s.shape[-1], self.agent_type))
        self.validate_agent_count(self.across_agent_types(lambda m: m.shape[-1], self.present_mask))

    def validate_agent_count(self, count_dict):
        self.across_agent_types(assert_equal, count_dict, self.agent_count)

    def get_world_center(self):
        return self.renderer.world_center

    def get_state(self):
        return self.across_agent_types(lambda kin: kin.get_state(), self.kinematic_model)

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
            return self.across_agent_types(
                lambda state, mask: lanelet_orientation_loss(
                    self.lanelet_map, state, self.recenter_offset,
                    direction_angle_threshold=self.cfg.wrong_way_angle_threshold,
                    lanelet_dist_tolerance=self.cfg.lanelet_inclusion_tolerance,
                ) * mask,
                self.get_state(), self.get_present_mask()
            )
        else:
            if not self.warned_no_lanelet:
                logger.debug("No lanelet map is provided. Returning zeros for wrong_way losses.")
                self.warned_no_lanelet = True
            return self.across_agent_types(
                lambda state: torch.zeros(state.shape[0], state.shape[1]).to(state.device), self.get_state())

    def get_agent_size(self):
        return self.agent_size

    def get_agent_type(self):
        return self.agent_type

    def get_present_mask(self):
        return self.present_mask

    def get_all_agents_absolute(self):
        return self.across_agent_types(
            lambda state, size, present: torch.cat([state[..., :3], size, present.unsqueeze(-1)], dim=-1),
            self.get_state(), self.get_agent_size(), self.get_present_mask()
        )

    def get_all_agents_relative(self, exclude_self=True):
        abs_agent_pos = self.get_all_agents_absolute()
        # concatenate absolute agent states across all agent types
        all_abs_agent_pos = torch.cat(list(abs_agent_pos.values()), dim=-2)
        cum_agent_count = list(np.cumsum(list(self.agent_count.values())))
        all_agent_count = cum_agent_count[-1]
        # compute where each agent type starts in the concatenated tensor
        agent_starts = dict(zip(self.agent_count.keys(), [0] + cum_agent_count[:-1]))

        def make_relative(abs_pos, agent_start, agent_count):
            xy, psi = abs_pos[..., :2], abs_pos[..., 2:3]  # current agent type
            all_xy, all_psi = all_abs_agent_pos[..., :2], all_abs_agent_pos[..., 2:3]  # all agent types
            # compute relative position of all agents w.r.t. each agent from of current type
            rel_xy, rel_psi = relative(origin_xy=xy.unsqueeze(-2), origin_psi=psi.unsqueeze(-2),
                                       target_xy=all_xy.unsqueeze(-3), target_psi=all_psi.unsqueeze(-3))
            rel_state = torch.cat([rel_xy, rel_psi], dim=-1)
            # insert the info that doesn't vary with the coordinate frame
            rel_pos = torch.cat([rel_state, all_abs_agent_pos[..., 3:].unsqueeze(-3).expand_as(rel_state)], dim=-1)
            if exclude_self:
                # remove the diagonal of the current agent type
                to_keep = torch.eye(agent_count, dtype=torch.bool, device=rel_pos.device).logical_not()
                # pad to keep all agents of other types
                to_keep = pad(to_keep, (agent_start, all_agent_count - agent_start - agent_count), value=True)
                # need to flatten to index two dimensions simultaneously
                to_keep = torch.flatten(to_keep)
                rel_pos = rel_pos.flatten(start_dim=-3, end_dim=-2)
                rel_pos = rel_pos[..., to_keep, :]
                # the result has one less agent in the penultimate dimension
                rel_pos = rel_pos.reshape((*rel_pos.shape[:-2], agent_count, all_agent_count - 1, 6))
            return rel_pos

        return self.across_agent_types(
            make_relative, abs_agent_pos, agent_starts, self.agent_count
        )

    def get_innermost_simulator(self) -> Self:
        return self

    def step(self, agent_action):
        self.internal_time += 1
        # validate agent types
        assert list(agent_action.keys()) == self.agent_types
        # validate tensor shape lengths
        self.across_agent_types(lambda s: assert_equal(len(s.shape), 3), agent_action)
        # validate batch size
        self.across_agent_types(lambda s: assert_equal(s.shape[0], self.batch_size), agent_action)
        # validate agent numbers
        self.validate_agent_count(self.across_agent_types(lambda s: s.shape[-2], agent_action))

        self.across_agent_types(lambda kin, act: kin.step(act), self.kinematic_model, agent_action)

        if self.traffic_controls is not None:
            for traffic_control_type, traffic_control in self.traffic_controls.items():
                traffic_control.step(self.internal_time)
        if self.waypoint_goals is not None:
            self.waypoint_goals.step(self.get_state(), self.internal_time, threshold=self.cfg.waypoint_removal_threshold)

    def set_state(self, agent_state, mask=None):
        if mask is None:
            mask = self.across_agent_types(lambda states: torch.ones_like(states[..., 0], dtype=torch.bool),
                                           agent_state)

        # validate agent types
        assert list(agent_state.keys()) == self.agent_types
        assert list(mask.keys()) == self.agent_types
        # validate tensor shape lengths
        self.across_agent_types(lambda s: assert_equal(len(s.shape), 3), agent_state)
        self.across_agent_types(lambda m: assert_equal(len(m.shape), 2), mask)
        # validate batch size
        b = self.batch_size
        self.across_agent_types(lambda s: assert_equal(s.shape[0], b), agent_state)
        self.across_agent_types(lambda m: assert_equal(m.shape[0], b), mask)
        # validate agent numbers
        self.validate_agent_count(self.across_agent_types(lambda s: s.shape[-2], agent_state))
        self.validate_agent_count(self.across_agent_types(lambda m: m.shape[-1], mask))

        def set_new_state(kinematic, state, mask):
            state_from_kinematic = kinematic.get_state()
            state_size, state_from_kinematic_size = state.shape[-1], state_from_kinematic.shape[-1]
            assert state_size <= state_from_kinematic_size
            if state_size < state_from_kinematic_size:
                state = torch.cat([state, state_from_kinematic[..., (state_size-state_from_kinematic_size):]], dim=-1)
            new_state = state.where(mask.unsqueeze(-1).expand_as(state), kinematic.get_state())
            kinematic.set_state(new_state)
            return kinematic

        self.across_agent_types(set_new_state, self.kinematic_model, agent_state, mask)

    def update_present_mask(self, present_mask):
        self.across_agent_types(lambda m: assert_equal(len(m.shape), 2), present_mask)
        self.across_agent_types(lambda m: assert_equal(m.shape[0], self.batch_size), present_mask)
        self.validate_agent_count(self.across_agent_types(lambda m: m.shape[-1], present_mask))

        self.present_mask = present_mask

    def fit_action(self, future_state, current_state=None):
        if current_state is None:
            current_state = {agent_type: None for agent_type in self.agent_types}

        action = self.across_agent_types(
            lambda kin, f, c: kin.fit_action(future_state=f, current_state=c),
            self.kinematic_model, future_state, current_state
        )

        return action

    def render(self, camera_xy, camera_psi, res=None, rendering_mask=None, fov=None,
               waypoints=None, waypoints_rendering_mask=None):
        camera_sc = torch.cat([torch.sin(camera_psi), torch.cos(camera_psi)], dim=-1)
        if len(camera_xy.shape) == 2:
            # Reshape from Bx2 to Bx1x2
            camera_xy = camera_xy.unsqueeze(1)
            camera_sc = camera_sc.unsqueeze(1)
        n_cameras = camera_xy.shape[-2]
        present_mask = {k: v.unsqueeze(-2).expand(v.shape[:-1] + (n_cameras,) + v.shape[-1:])
                        for k, v in self.get_present_mask().items()}
        rendering_mask = {
            k: v if rendering_mask is None else v.logical_and(rendering_mask[k])
            for k, v in present_mask.items()
        }
        return self.renderer.render_frame(
            self.get_state(), self.get_agent_size(), camera_xy, camera_sc, rendering_mask, res=res, fov=fov,
            waypoints=waypoints, waypoints_rendering_mask=waypoints_rendering_mask,
            traffic_controls={k: v.copy() for k, v in self.traffic_controls.items()}
                if self.traffic_controls is not None else None
        )

    def compute_offroad(self):
        offroad = self.across_agent_types(
            lambda state, lenwid, mask: offroad_infraction_loss(
                state, lenwid, self.road_mesh, threshold=self.cfg.offroad_threshold
            ) * mask.to(state.dtype),
            self.get_state(), self.agent_size, self.get_present_mask(),
        )
        return offroad

    def _compute_collision_of_single_agent(self, box, remove_self_overlap=None, agent_types=None):
        assert len(box.shape) == 2
        assert box.shape[0] == self.batch_size
        assert box.shape[-1] == 5
        flattened = HomogeneousWrapper(self)

        def f(x, agent_dim):
            out = x
            if agent_types is not None:
                out = flattened.agent_split(out, agent_dim=agent_dim)
                out = {k: out[k] for k in agent_types if k in out}
                if out:
                    out = flattened.agent_concat(out, agent_dim=agent_dim)
                else:
                    new_shape = list(x.shape)
                    new_shape[agent_dim] = 0
                    out = torch.zeros(new_shape, dtype=x.dtype, device=x.device)
            return out

        states = f(flattened.get_state(), agent_dim=-2)
        if states.shape[-2] == 0:
            return torch.zeros_like(box[..., 0])
        sizes = f(flattened.get_agent_size(), agent_dim=-2)
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
        overlap = overlap * f(flattened.get_present_mask(), agent_dim=-1).to(overlap.dtype)
        collision = overlap.sum(dim=-1)
        if remove_self_overlap is None:
            remove_self_overlap = torch.ones_like(collision)
        collision = collision - overlap.max(dim=-1)[0] * remove_self_overlap.to(collision.dtype)  # self-overlap is always highest
        return collision

    def _compute_collision_of_multi_agents(self, mask=None):
        collision_mask = self.get_present_mask() if mask is None else mask  # BxA
        # Need to flatten and get all agents boxes for calculating each agent's collision
        flattened = HomogeneousWrapper(self)
        collision_mask = flattened.agent_concat(collision_mask, -1)
        states = flattened.get_state()
        sizes = flattened.get_agent_size()
        present_mask = flattened.get_present_mask()
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
        collision = flattened.agent_split(collision, agent_dim=-1)
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
    def agent_functor(self) -> AgentTypeFunctor:
        return self.inner_simulator.agent_functor

    @property
    def agent_types(self):
        return self.inner_simulator.agent_types

    @property
    def action_size(self) -> IntPerAgentType:
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
               waypoints_rendering_mask=None):
        return self.inner_simulator.render(camera_xy, camera_psi, res=res, rendering_mask=rendering_mask, fov=fov,
                                           waypoints=waypoints, waypoints_rendering_mask=waypoints_rendering_mask)


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
    def __init__(self, simulator: SimulatorInterface, npc_mask: TensorPerAgentType):
        super().__init__(simulator)
        self.npc_mask = npc_mask

    def _update_npc_present_mask(self) -> TensorPerAgentType:
        """
        Computes updated present masks for NPCs, with arbitrary padding for the remaining agents.
        By default, leaves present masks unchanged.

        Returns:
            a functor of BxA boolean tensors, where A is the number of agents in the inner simulator
        """
        return self.inner_simulator.get_present_mask()

    def _get_npc_action(self) -> TensorPerAgentType:
        """
        Computes the actions for NPCs, with arbitrary padding for actions of the remaining agents.
        By default, the actions are all zeros, but subclasses can implement more intelligent behavior.

        Returns:
            a functor of BxAxAc tensors, where A is the number of agents in the inner simulator
        """
        return self.across_agent_types(
            lambda s, a: torch.zeros(s.shape[:-1] + (a,), dtype=s.dtype, device=s.device),
            self.inner_simulator.get_state(), self.action_size
        )

    def _npc_teleport_to(self) -> Optional[TensorPerAgentType]:
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
        self.npc_mask = self.agent_functor.to_device(self.npc_mask, device)
        return super().to(device)

    def copy(self):
        inner_copy = self.inner_simulator.copy()
        other = self.__class__(inner_copy, npc_mask=self.npc_mask)
        return other

    def validate_agent_count(self, count_dict):
        self.across_agent_types(assert_equal, count_dict, self.agent_count)

    def get_state(self):
        states = self.across_agent_types(
            lambda x, k: x[..., k.logical_not(), :], self.inner_simulator.get_state(), self.npc_mask
        )
        return states

    def get_waypoints(self):
        waypoints = self.inner_simulator.get_waypoints()
        if waypoints is not None:
            waypoints = self.across_agent_types(
                lambda x, k: x[..., k.logical_not(), :, :], waypoints, self.npc_mask
            )
        return waypoints

    def get_waypoints_state(self):
        waypoints_state = self.inner_simulator.get_waypoints_state()
        if waypoints_state is not None:
            waypoints_state = self.across_agent_types(
                lambda x, k: x[..., k.logical_not(), :], waypoints_state, self.npc_mask
            )
        return waypoints_state

    def get_waypoints_mask(self):
        masks = self.inner_simulator.get_waypoints_mask()
        if masks is not None:
            masks = self.across_agent_types(
                lambda x, k: x[..., k.logical_not(), :], masks, self.npc_mask
            )
        return masks

    def get_agent_size(self):
        sizes = self.across_agent_types(
            lambda x, k: x[..., k.logical_not(), :], self.inner_simulator.get_agent_size(), self.npc_mask
        )
        return sizes

    def get_agent_type(self):
        sizes = self.across_agent_types(
            lambda x, k: x[..., k.logical_not()], self.inner_simulator.get_agent_type(), self.npc_mask
        )
        return sizes

    def get_present_mask(self):
        present_mask = self.across_agent_types(
            lambda x, k: x[..., k.logical_not()], self.inner_simulator.get_present_mask(), self.npc_mask
        )
        return present_mask

    def get_all_agents_relative(self, exclude_self=True):
        agent_info = self.inner_simulator.get_all_agents_relative(exclude_self=exclude_self)
        agent_info = self.across_agent_types(
            lambda x, k: x[..., k.logical_not(), :, :], agent_info, self.npc_mask
        )
        return agent_info

    def set_state(self, agent_state, mask=None):
        if mask is None:
            mask = self.across_agent_types(
                lambda states: torch.ones_like(states[..., 0], dtype=torch.bool), agent_state
            )

        # validate agent types
        assert list(agent_state.keys()) == self.agent_types
        assert list(mask.keys()) == self.agent_types
        # validate tensor shape lengths
        self.across_agent_types(lambda s: assert_equal(len(s.shape), 3), agent_state)
        self.across_agent_types(lambda m: assert_equal(len(m.shape), 2), mask)
        # validate batch shape
        b = self.batch_size
        self.across_agent_types(lambda s: assert_equal(s.shape[0], b), agent_state)
        self.across_agent_types(lambda m: assert_equal(m.shape[0], b), mask)
        # validate agent numbers
        self.validate_agent_count(self.across_agent_types(lambda s: s.shape[-2], agent_state))
        self.validate_agent_count(self.across_agent_types(lambda m: m.shape[-1], mask))

        def set_masked_state(old_state, mask, new_state):
            with_padding = torch.zeros_like(old_state)
            with_padding[..., torch.torch.logical_not(mask), :] = new_state  # I *think* autodiff handles this correctly
            selection_mask = self.extend_tensor(mask, old_state.shape[:-2]).unsqueeze(-1).expand_as(old_state)
            updated_state = old_state.where(selection_mask, with_padding)
            return updated_state

        def make_full_mask(npc_mask, current_mask):
            non_replay_mask = npc_mask.logical_not()
            full_mask = torch.zeros_like(non_replay_mask, dtype=torch.bool)
            full_mask = NPCWrapper.extend_tensor(full_mask, current_mask.shape[:-1]).clone()
            full_mask[..., non_replay_mask] = current_mask
            return full_mask

        states = self.across_agent_types(
            set_masked_state, self.inner_simulator.get_state(), self.npc_mask, agent_state
        )
        # Only set state for non-replay agents
        full_mask = self.across_agent_types(make_full_mask, self.npc_mask, mask)

        self.inner_simulator.set_state(states, mask=full_mask)

    def step(self, action):
        # validate tensor shape lengths
        self.across_agent_types(lambda s: assert_equal(len(s.shape), 3), action)
        # validate batch shape
        self.across_agent_types(lambda s: assert_equal(s.shape[0], self.batch_size), action)
        # validate agent numbers
        self.validate_agent_count(self.across_agent_types(lambda s: s.shape[-2], action))

        def make_full_action(replay_action, mask, given_action):
            with_padding = torch.zeros_like(replay_action)
            with_padding[..., torch.logical_not(mask), :] = given_action
            selection_mask = self.extend_tensor(mask, replay_action.shape[:-2]).unsqueeze(-1).expand_as(replay_action)
            full_action = replay_action.where(selection_mask, with_padding)
            return full_action

        # step all agents, with dummy action for replay agents
        npc_action = self._get_npc_action()
        full_actions = self.across_agent_types(make_full_action, npc_action, self.npc_mask, action)
        self.inner_simulator.step(full_actions)

        # set target state for replay vehicles
        npc_state = self._npc_teleport_to()
        if npc_state is not None:
            full_npc_mask = self.across_agent_types(
                lambda r, p: r[None, :].expand_as(p), self.npc_mask, self.inner_simulator.get_present_mask()
            )
            self.inner_simulator.set_state(npc_state, mask=full_npc_mask)

        # update presence mask of NPCs in case it changed
        non_replay_present_mask = self.across_agent_types(
            lambda x, k: x[..., torch.logical_not(k)], self.inner_simulator.get_present_mask(), self.npc_mask
        )
        self.update_present_mask(non_replay_present_mask)

    def update_present_mask(self, present_mask):
        self.across_agent_types(lambda m: assert_equal(len(m.shape), 2), present_mask)
        self.across_agent_types(lambda m: assert_equal(m.shape[0], self.batch_size), present_mask)
        self.validate_agent_count(self.across_agent_types(lambda m: m.shape[-1], present_mask))

        replay_present_mask = self._update_npc_present_mask()

        def make_present_mask(recorded_present_mask, replay_mask, non_replay_present_mask):
            new_present_mask = recorded_present_mask.clone()
            new_present_mask[..., torch.logical_not(replay_mask)] = non_replay_present_mask
            return new_present_mask

        new_present_mask = self.across_agent_types(
            make_present_mask, replay_present_mask, self.npc_mask, present_mask
        )
        self.inner_simulator.update_present_mask(new_present_mask)

    def render(self, camera_xy, camera_psi, res=None, rendering_mask=None, fov=None, waypoints=None,
               waypoints_rendering_mask=None):
        if rendering_mask is not None:
            def pad_rendering_mask(rd_mask, rpl_mask):
                new_mask = torch.zeros(rd_mask.shape[0], rd_mask.shape[1], rpl_mask.shape[0], device=rd_mask.device)
                new_mask[..., torch.logical_not(rpl_mask)] = rd_mask
                return new_mask
            rendering_mask = self.across_agent_types(
                pad_rendering_mask, rendering_mask, self.npc_mask
            )
        return self.inner_simulator.render(camera_xy, camera_psi, res, rendering_mask, fov=fov, waypoints=waypoints,
                                           waypoints_rendering_mask=waypoints_rendering_mask)

    def fit_action(self, future_state, current_state=None):
        full_state = self.inner_simulator.get_state()

        def augment_state(full, partial, mask):
            result = full.clone()
            result[..., mask.logical_not(), :] = partial
            return result

        full_future_state = self.across_agent_types(
            augment_state, full_state, future_state, self.npc_mask
        )
        if current_state is None:
            full_current_state = None
        else:
            full_current_state = self.across_agent_types(
                augment_state, full_state, current_state, self.npc_mask
            )

        full_action = self.inner_simulator.fit_action(full_future_state, full_current_state)
        action = self.across_agent_types(
            lambda x, m: x[..., m.logical_not(), :], full_action, self.npc_mask
        )
        return action

    def compute_offroad(self):
        offroad = self.across_agent_types(
            lambda state, lenwid, mask: offroad_infraction_loss(
                state, lenwid, self.get_innermost_simulator().road_mesh, threshold=self.cfg.offroad_threshold
            ) * mask.to(state.dtype),
            self.get_state(), self.get_agent_size(), self.get_present_mask(),
        )
        return offroad

    def compute_wrong_way(self):
        innermost_simulator = self.get_innermost_simulator()
        if innermost_simulator.lanelet_map is not None:
            if isinstance(innermost_simulator.lanelet_map, Iterable) and None in innermost_simulator.lanelet_map \
                    and not innermost_simulator.warned_no_lanelet:
                idx_no_map = [i for i, item in enumerate(innermost_simulator.lanelet_map) if item is None]
                logger.debug(f"Batches {idx_no_map} have no lanelet map. Returning zeros for wrong_way losses.")
                innermost_simulator.warned_no_lanelet = True
            return self.across_agent_types(lambda state, mask: lanelet_orientation_loss(
                    innermost_simulator.lanelet_map, state, innermost_simulator.recenter_offset,
                ) * mask, self.get_state(), self.get_present_mask())
        else:
            if not innermost_simulator.warned_no_lanelet:
                logger.debug("No lanelet map is provided. Returning zeros for wrong_way losses.")
                innermost_simulator.warned_no_lanelet = True
            return self.across_agent_types(
                lambda state: torch.zeros(state.shape[0], state.shape[1]).to(state.device), self.get_state())

    def _compute_collision_of_multi_agents(self, mask=None):
        batched_non_replay_mask = self.across_agent_types(
            lambda k: ~k.expand(self.get_innermost_simulator().batch_size, -1), self.npc_mask
        )
        if mask is not None:
            batched_non_replay_mask = self.across_agent_types(
                lambda bnrm, m: bnrm * m, batched_non_replay_mask, mask
            )
        collision = self.across_agent_types(
            lambda c, k: c[..., k.logical_not()],
            self.inner_simulator._compute_collision_of_multi_agents(batched_non_replay_mask), self.npc_mask
        )
        return collision

    @staticmethod
    def extend_tensor(x, batch_dims):
        # add specified dimensions to the front of the tensor
        x_dims = x.shape
        for d in batch_dims:
            x.unsqueeze(0)
        return x.expand(batch_dims + x_dims)


class HomogeneousWrapper(SimulatorWrapper):
    """
    Removes distinction between agent types, replacing collections with single tensors
    concatenated across the agent dimension. Only safe to use when all agents share the same
    kinematic model.
    """
    def __init__(self, simulator: SimulatorInterface):
        super().__init__(simulator)

    @property
    def agent_functor(self) -> SingletonAgentTypeFunctor:
        return SingletonAgentTypeFunctor()

    @property
    def agent_types(self):
        return None

    @property
    def action_size(self) -> IntPerAgentType:
        action_sizes = self.inner_simulator.action_size
        if isinstance(action_sizes, int):
            return action_sizes
        else:
            return max([action_size for action_size in action_sizes.values()])

    def agent_concat(self, d, agent_dim, check_agent_count=True):
        """
        Concatenates a collection of agent tensors along the agent dimension.
        """
        if check_agent_count:
            self.inner_simulator.across_agent_types(
                lambda x, n: assert_equal(x.shape[agent_dim], n), d, self.inner_simulator.agent_count
            )
        if isinstance(self.inner_simulator.agent_functor, DictAgentTypeFunctor):
            x = torch.cat(list(d.values()), dim=agent_dim)
        else:
            x = d
        return x

    def agent_split(self, x, agent_dim):
        """
        Reverse of `agent_concat`, splits a tensor into a collection of agent tensors.
        """
        assert x.shape[agent_dim] == self.agent_count
        if isinstance(self.inner_simulator.agent_functor, DictAgentTypeFunctor):
            length = list(self.inner_simulator.agent_count.values())
            offset = [0] + list(np.cumsum(length))
            split = {agent_type: x.narrow(dim=agent_dim, start=offset[i], length=length[i])
                     for (i, agent_type) in enumerate(self.inner_simulator.agent_types)}
        else:
            split = x
        return split

    def get_state(self):
        return self.agent_concat(self.inner_simulator.get_state(), agent_dim=-2)

    def get_agent_size(self):
        return self.agent_concat(self.inner_simulator.get_agent_size(), agent_dim=-2)

    def get_agent_type(self):
        return self.agent_concat(self.inner_simulator.get_agent_type(), agent_dim=-1)

    def get_present_mask(self):
        return self.agent_concat(self.inner_simulator.get_present_mask(), agent_dim=-1)

    def get_waypoints(self):
        waypoints = self.inner_simulator.get_waypoints()
        if waypoints is not None:
            waypoints = self.agent_concat(waypoints, agent_dim=-3)
        return waypoints

    def get_waypoints_state(self):
        waypoints_state = self.inner_simulator.get_waypoints_state()
        if waypoints_state is not None:
            waypoints_state = self.agent_concat(waypoints_state, agent_dim=-2)
        return waypoints_state

    def get_waypoints_mask(self):
        masks = self.inner_simulator.get_waypoints_mask()
        if masks is not None:
            masks = self.agent_concat(masks, agent_dim=-2)
        return masks

    def get_all_agents_absolute(self):
        agent_info = self.inner_simulator.get_all_agents_absolute()
        return self.agent_concat(agent_info, agent_dim=-2, check_agent_count=False)

    def get_all_agents_relative(self, exclude_self=True):
        agent_info = self.inner_simulator.get_all_agents_relative(exclude_self=exclude_self)
        return self.agent_concat(agent_info, agent_dim=-3)

    def set_state(self, agent_state, mask=None):
        inner_mask = self.agent_split(mask, agent_dim=-1) if mask is not None else None
        self.inner_simulator.set_state(
            self.agent_split(agent_state, agent_dim=-2), inner_mask
        )

    def step(self, action):
        self.inner_simulator.step(
            self.agent_split(action, agent_dim=-2)
        )

    def update_present_mask(self, present_mask):
        self.inner_simulator.update_present_mask(
            self.agent_split(present_mask, agent_dim=-1)
        )

    def fit_action(self, future_state, current_state=None):
        inner_current = self.agent_split(current_state, agent_dim=-2) if current_state is not None else None
        inner_action = self.inner_simulator.fit_action(
            self.agent_split(future_state, agent_dim=-2), inner_current
        )
        action = self.agent_concat(inner_action, agent_dim=-2)
        return action

    def compute_offroad(self):
        offroad = self.inner_simulator.compute_offroad()
        offroad = self.agent_concat(offroad, agent_dim=-1)
        return offroad

    def compute_wrong_way(self):
        return self.agent_concat(self.inner_simulator.compute_wrong_way(), agent_dim=-1)

    def _compute_collision_of_multi_agents(self, mask=None):
        if mask is not None:
            mask = self.agent_split(mask, agent_dim=-1)
        return self.agent_concat(self.inner_simulator._compute_collision_of_multi_agents(mask), agent_dim=-1)

    def render(self, camera_xy, camera_psi, res=None, rendering_mask=None, fov=None, waypoints=None,
               waypoints_rendering_mask=None):
        rendering_mask = self.agent_split(rendering_mask, -1) if rendering_mask is not None else None
        return self.inner_simulator.render(camera_xy, camera_psi, res, rendering_mask, fov=fov, waypoints=waypoints,
                                           waypoints_rendering_mask=waypoints_rendering_mask)


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
            waypoints = s.get_waypoints()
            singleton = isinstance(s.agent_functor, SingletonAgentTypeFunctor)
            if waypoints is not None:
                waypoints_mask = s.get_waypoints_mask()
                waypoints_dict = dict(agent=waypoints) if singleton else waypoints
                waypoints_mask_dict = dict(agent=waypoints_mask) if singleton else waypoints_mask
                waypoints = torch.cat(list(waypoints_dict.values()), dim=-3).flatten(1,2).unsqueeze(dim=1)
                waypoints_mask = torch.cat(list(waypoints_mask_dict.values()), dim=-2).flatten(1,2).unsqueeze(dim=1)
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

    def __init__(self, simulator: SimulatorInterface, exposed_agent_limit: int, default_action: TensorPerAgentType):
        super().__init__(simulator)
        self.default_action = default_action
        self.exposed_agent_limit = exposed_agent_limit
        self.exposed_agents: TensorPerAgentType = None  # functor of BxE int tensor of indices of exposed agents
        self.update_exposed_agents()

    def update_exposed_agents(self) -> None:
        """
        There should always be E exposed agents, although some of them may be marked as absent.
        """
        device = self.get_world_center().device
        self.exposed_agents = self.across_agent_types(
            lambda n: torch.arange(n, dtype=torch.long, device=device).expand((self.batch_size, n)),
            self.exposed_agent_limit
        )

    def get_exposed_agents(self) -> TensorPerAgentType:
        """
        Returns:
            a functor of BxE int tensors of indices of exposed agents
        """
        return self.exposed_agents

    def is_exposed(self) -> TensorPerAgentType:
        """
        Returns:
            a functor of BxA boolean tensors indicating which agents are exposed
        """
        return self.across_agent_types(
            lambda exposed, agent_count, m: torch.stack([isin(torch.arange(agent_count).to(exposed.device),
                                                 exposed[b, m[b]]) for b in range(exposed.shape[0])]),
            self.exposed_agents, self.inner_simulator.agent_count, self.get_present_mask()
        )

    def _restrict_tensor(self, tensor: TensorPerAgentType, agent_dim: int) -> TensorPerAgentType:
        """
        Selects exposed agents from a given tensor, along the supplied agent dimension.
        """
        if agent_dim == -1:
            return self.across_agent_types(
                lambda x, i: x.gather(index=i, dim=agent_dim), tensor, self.get_exposed_agents()
            )
        elif agent_dim == -2:
            return self.across_agent_types(
                lambda x, i: x.gather(index=i.unsqueeze(-1).expand(i.shape + x.shape[-1:]), dim=agent_dim),
                tensor, self.get_exposed_agents()
            )
        elif agent_dim == -3:
            return self.across_agent_types(
                lambda x, i: x.gather(index=i[..., None, None].expand(i.shape + x.shape[-2:]), dim=agent_dim),
                tensor, self.get_exposed_agents()
            )
        else:
            raise NotImplementedError

    def _extend_tensor(self, tensor: TensorPerAgentType, padding: TensorPerAgentType) -> TensorPerAgentType:
        """
        Given a tensor of exposed agents, constructs a tensor of all agents, using supplied padding.
        Agent dimension should be the penultimate one.
        """

        def extend(x, pad, selection):
            extended = pad.clone()
            batch_idx = torch.arange(extended.shape[0]).unsqueeze(-1).repeat(1, selection.shape[-1])
            extended[batch_idx, selection] = x
            return extended

        return self.across_agent_types(extend, tensor, padding, self.get_exposed_agents())

    def to(self, device) -> Self:
        self.default_action = self.agent_functor.to_device(self.default_action, device)
        if self.exposed_agents is not None:
            self.exposed_agents = self.agent_functor.to_device(self.exposed_agents, device)
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
        extended.default_action = self.across_agent_types(enlarge, self.default_action)
        self.exposed_agents = self.across_agent_types(enlarge, self.exposed_agents)
        return extended

    def select_batch_elements(self, idx, in_place=True):
        other = super().select_batch_elements(idx, in_place=in_place)
        other.default_action = other.across_agent_types(lambda x: x[idx], other.default_action)
        if other.exposed_agents is not None:
            other.exposed_agents = other.across_agent_types(lambda x: x[idx], other.exposed_agents)
        return other

    def get_state(self):
        return self._restrict_tensor(self.inner_simulator.get_state(), agent_dim=-2)

    def get_present_mask(self):
        return self.across_agent_types(
            lambda x: torch.ones_like(x, dtype=torch.bool), self.get_exposed_agents()
        )

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
               waypoints_rendering_mask=None):
        if rendering_mask is not None:
            def padding_rendering_mask(rd_mask, expose_mask):
                rd_mask = rd_mask.permute(0, 2, 1)
                pad_tensor = torch.zeros(rd_mask.shape[0], expose_mask.shape[1], rd_mask.shape[1], device=rd_mask.device)
                return self._extend_tensor(rd_mask, pad_tensor).permute(0, 2, 1)
            rendering_mask = self.across_agent_types(padding_rendering_mask, rendering_mask, self.is_exposed())
        return self.inner_simulator.render(camera_xy, camera_psi, res, rendering_mask, fov=fov, waypoints=waypoints,
                                           waypoints_rendering_mask=waypoints_rendering_mask)

    def step(self, action):
        extended_action = self._extend_tensor(action, padding=self.default_action)
        super().step(extended_action)
        self.update_exposed_agents()

    def update_present_mask(self, *args, **kwargs):
        raise NotImplementedError("Updating present mask for SelectiveWrapper")

    def _compute_collision_of_multi_agents(self, mask=None):
        exposed_mask = self.across_agent_types(
            lambda exposed: exposed.reshape(self.get_innermost_simulator().batch_size, exposed.shape[-1]),
            self.is_exposed()
        )
        if mask is not None:
            exposed_mask = self.across_agent_types(lambda em, m: em * m, exposed_mask, mask)
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

    def __init__(self, simulator: SimulatorInterface, exposed_agent_limit: int, default_action: TensorPerAgentType,
                 warmup_timesteps: int, cutoff_polygon_verts: TensorPerAgentType):
        super().__init__(simulator, exposed_agent_limit, default_action)
        # Optional BxNx2 Tensor Collection for the polygon region to restrict agents,
        # the order can be either clockwise or anti-clockwise
        self.cutoff_polygon_verts = cutoff_polygon_verts
        # Scaling factor for the polygon region
        # After how many steps the agents are warmed-up
        self.warmup_timesteps = warmup_timesteps
        # Track for how many timesteps each agent has been in range
        self.proximal_timesteps = self.across_agent_types(
            lambda x: torch.zeros_like(x, dtype=torch.long, device=simulator.get_world_center().device),  # BxA
            self.inner_simulator.get_present_mask()
        )

        self.update_exposed_agents()

    def update_exposed_agents(self):
        if self.exposed_agents is None:
            super().update_exposed_agents()
            return  # for first run only

        def compute_proximal(state, proximal_timesteps, present, cutoff_boundary):
            location = state[..., :2]
            is_proximal = present
            if cutoff_boundary.shape[-2] > 0:
                is_proximal = is_proximal.logical_and(is_inside_polygon(location, cutoff_boundary))
            proximal_timesteps = is_proximal * (is_proximal + proximal_timesteps)
            return proximal_timesteps

        self.proximal_timesteps = self.across_agent_types(
            compute_proximal, self.inner_simulator.get_state(), self.proximal_timesteps,
            self.inner_simulator.get_present_mask(), self.cutoff_polygon_verts
        )

        def compute_exposed(prev_exposed, prev_is_exposed, proximal_timesteps, slot_taken):
            exposed = prev_exposed.clone()  # BxE
            is_proximal = proximal_timesteps > 0  # BxA
            is_to_expose = is_proximal.logical_and(prev_is_exposed.logical_not())  # BxA
            for batch_idx in range(self.batch_size):
                # I couldn't find a way to avoid iterating over the batch dimension
                to_expose = is_to_expose[0].nonzero(as_tuple=False)
                available_slots = slot_taken[0].logical_not().nonzero(as_tuple=False)
                if to_expose.shape[0] > available_slots.shape[0]:
                    logger.warning("More proximal agents than we can fit into exposed slots")
                    to_expose = to_expose[:available_slots.shape[0]]
                exposed[batch_idx, available_slots[:to_expose.shape[0]]] = to_expose
            return exposed

        self.exposed_agents = self.across_agent_types(
            compute_exposed, self.exposed_agents, self.is_exposed(), self.proximal_timesteps, self.get_present_mask()
        )

    def get_present_mask(self):
        def is_present(exposed, proximal_timesteps):
            return proximal_timesteps.gather(index=exposed, dim=-1) > 0

        return self.across_agent_types(
            is_present, self.exposed_agents, self.proximal_timesteps
        )

    def is_warmed_up(self):
        """
        Returns:
            a functor of BxA boolean tensors indicating which agents are warmed-up
        """
        return self.across_agent_types(
            lambda x, t: x.logical_and(t > self.warmup_timesteps),
            self.is_exposed(), self.proximal_timesteps
        )

    def to(self, device) -> Self:
        self.proximal_timesteps = self.agent_functor.to_device(self.proximal_timesteps, device)
        if self.cutoff_polygon_verts is not None:
            self.cutoff_polygon_verts = self.agent_functor.to_device(self.cutoff_polygon_verts, device)
        return super().to(device)

    def copy(self):
        inner_copy = self.inner_simulator.copy()
        other = self.__class__(
            inner_copy, exposed_agent_limit=self.agent_count, default_action=self.default_action,
            warmup_timesteps=self.warmup_timesteps, cutoff_polygon_verts=self.cutoff_polygon_verts
        )
        other.exposed_agents = self.exposed_agents
        other.warmup_timesteps = self.warmup_timesteps
        other.proximal_timesteps = self.across_agent_types(
            lambda x: x.clone(), self.proximal_timesteps
        )
        return other

    def extend(self, n, in_place=True):
        extended = super().extend(n, in_place=in_place)
        enlarge = lambda x: x.unsqueeze(1).expand((x.shape[0], n) + x.shape[1:]).\
            reshape((n * x.shape[0],) + x.shape[1:])
        extended.proximal_timesteps = self.across_agent_types(
            enlarge, self.proximal_timesteps
        )
        if extended.cutoff_polygon_verts is not None:
            extended.cutoff_polygon_verts = self.across_agent_types(
                enlarge, self.cutoff_polygon_verts
            )
        return extended

    def select_batch_elements(self, idx, in_place=True):
        other = super().select_batch_elements(idx, in_place=in_place)
        other.proximal_timesteps = other.across_agent_types(lambda x: x[idx], other.proximal_timesteps)
        if other.cutoff_polygon_verts is not None:
            other.cutoff_polygon_verts = other.across_agent_types(
                lambda x: x[idx], other.cutoff_polygon_verts
            )
        return other


class NoReentryBoundedRegionWrapper(BoundedRegionWrapper):
    """
    A variant of `BoundedRegionWrapper` that does not allow reentry.
    This means that if an agent is not exposed at a certain time, it will never be exposed in the future.
    However, the agents that can stop being exposed if they exit the area of interest.
    """

    def __init__(self, simulator: SimulatorInterface, exposed_agent_limit, default_action, warmup_timesteps,
                 cutoff_polygon_verts: Optional[TensorPerAgentType] = None):
        self.previous_present_mask = simulator.get_present_mask()
        self.proximal_timesteps = None
        super().__init__(simulator, exposed_agent_limit, default_action,
                 warmup_timesteps, cutoff_polygon_verts,)

    def to(self, device) -> Self:
        self.previous_present_mask = self.agent_functor.to_device(self.previous_present_mask, device)
        if self.proximal_timesteps is not None:
            self.proximal_timesteps = self.agent_functor.to_device(self.proximal_timesteps, device)
        return super().to(device)

    def extend(self, n, in_place=True):
        extended = super().extend(n, in_place=in_place)
        enlarge = lambda x: x.unsqueeze(1).expand((x.shape[0], n) + x.shape[1:]).reshape((n * x.shape[0],) +
                                                                                         x.shape[1:])
        extended.previous_present_mask = self.across_agent_types(
            enlarge, self.previous_present_mask
        )
        return extended

    def select_batch_elements(self, idx, in_place=True):
        other = super().select_batch_elements(idx, in_place=in_place)
        other.previous_present_mask = other.across_agent_types(lambda x: x[idx], other.previous_present_mask)
        return other

    def update_exposed_agents(self):
        super().update_exposed_agents()
        if self.proximal_timesteps is None:
            return  # for first run only
        self.proximal_timesteps = self.across_agent_types(
            lambda x, y: x.mul(y), self.proximal_timesteps, self.previous_present_mask
        )
        present_mask = self.get_present_mask()
        self.inner_simulator.update_present_mask(present_mask)
        self.previous_present_mask = self.agent_functor(torch.logical_and, present_mask, self.previous_present_mask)

    def update_present_mask(self, present_mask):
        self.inner_simulator.update_present_mask(present_mask)
        self.previous_present_mask = present_mask

    def get_present_mask(self):
        present_mask = super().get_present_mask()

        return self.agent_functor(torch.logical_and, present_mask, self.previous_present_mask)
