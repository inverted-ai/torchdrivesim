from dataclasses import dataclass
import json
import random
from typing import List, Dict
from typing_extensions import Self
from functools import reduce, lru_cache
from enum import auto, Enum
import logging
import torch

from torchdrivesim.traffic_controls import TrafficLightControl

logger = logging.getLogger(__name__)


class TrafficLightState(Enum):
    none = auto()
    green = auto()
    yellow = auto()
    red = auto()


ActorStates = Dict[str, TrafficLightState]


@dataclass(eq=True)
class TrafficLightGroupState:
    """
    Representing a single state of a group of traffic lights.
    """
    actor_states: ActorStates
    sequence_number: int
    duration: float # in seconds
    next_state: int


class TrafficLightStateMachine:
    def __init__(self, group_states: List[TrafficLightGroupState]):
        """
        TrafficLightStateMachine that traverse a list of TrafficLightGroupStates.
        
        Args:
            group_states (List[TrafficLightGroupState]): A list of TrafficLightGroupState objects.

        """
        self._states = group_states
        self._time_remaining, self._current_state, self._duration = None, None, None
        self.reset()

    @classmethod
    def from_json(cls, json_file_path: str) -> Self:
        """
        Current json format:
        [
            {
                "actor_states": {
                    "4411": "red",
                    "3411": "red",
                    .........
                },
                "state": "0",
                "duration": 10,
                "next_state": "1"
            },
            ...
        ]
        """
        with open(json_file_path, "rb") as f:
            items = json.load(f)
        try:
            return cls(
                [
                    TrafficLightGroupState(
                        actor_states={
                            k: TrafficLightState[v]
                            for k, v in item["actor_states"].items()
                        },
                        sequence_number=int(item["state"]),
                        duration=float(item["duration"]),
                        next_state=int(item["next_state"]),
                    )
                    for item in items
                ]
            )
        except KeyError as e:
            raise ValueError(f"KeyError: {e} in {json_file_path}")

    def to_json(self) -> str:
        return json.dumps(
            [
                {
                    "actor_states": {k: v.name for k, v in state.actor_states.items()},
                    "state": str(state.sequence_number),
                    "duration": state.duration,
                    "next_state": str(state.next_state),
                }
                for state in self._states
            ]
        )

    def reset(self):
        state = random.randint(0, len(self._states) - 1)
        self.set_to(state, self._states[state].duration)

    def set_to(self, state_index: int, time_remaining: float):
        """
        Set the TrafficLightStateMachine to a specific state.
        """
        state = state_index
        if state_index < 0:
            state = 0
        elif state_index >= len(self._states):
            state = len(self._states) - 1
        self._current_state = self._states[state]
        self._duration = self._current_state.duration
        self._time_remaining = (
            time_remaining if time_remaining <= self._duration else self._duration
        )

    def tick(self, dt: float):
        """
        Update the time remaining for the current state.
        """
        self._time_remaining -= dt
        while self._time_remaining <= 0:
            next_state = self._current_state.next_state
            next_duration = self._states[next_state].duration
            if self._time_remaining == 0:
                self.set_to(next_state, next_duration)
                break
            elif self._time_remaining + next_duration > 0:
                self._time_remaining += next_duration
                self.set_to(next_state, self._time_remaining)
                break
            else:
                self._time_remaining += next_duration
                self._current_state = self._states[next_state]

    @property
    def states(self) -> List[TrafficLightGroupState]:
        return self._states

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def current_state(self) -> TrafficLightGroupState:
        return self._current_state

    @property
    def time_remaining(self) -> float:
        return self._time_remaining

    def get_current_actor_states(self) -> ActorStates:
        return self.current_state.actor_states


class TrafficLightController:
    def __init__(self, traffic_fsms: List[TrafficLightStateMachine]):
        """
        TrafficLightController that steps a group of TrafficLightStateMachines.
        """
        self.traffic_fsms = traffic_fsms
        self._time_remaining, self._current_state, self._state_per_machine = (
            None,
            None,
            None,
        )
        self.reset()

    @classmethod
    def from_json(cls, json_file_path: str) -> Self:
        """
        Current json format:
        [
            [
                {
                    "actor_states": {
                        "4411": "red",
                        "3411": "red",
                        .........
                    },
                    "state": "0",
                    "duration": 10,
                    "next_state": "1"
                },
                ...
            ],
            ...
        ]
        """
        with open(json_file_path, "rb") as f:
            items = json.load(f)
        try:
            return cls(
                [
                    TrafficLightStateMachine(
                        group_states=[
                            TrafficLightGroupState(
                                actor_states={
                                    k: TrafficLightState[v]
                                    for k, v in gs["actor_states"].items()
                                },
                                sequence_number=int(gs["state"]),
                                duration=float(gs["duration"]),
                                next_state=int(gs["next_state"]),
                            )
                            for gs in sm
                        ]
                    )
                    for sm in items
                ]
            )
        except KeyError as e:
            raise ValueError(f"KeyError: {e} in {json_file_path}")

    def to_json(self) -> str:
        return json.dumps(
            [
                [
                    {
                        "actor_states": {
                            k: v.name for k, v in state.actor_states.items()
                        },
                        "state": str(state.sequence_number),
                        "duration": state.duration,
                        "next_state": str(state.next_state),
                    }
                    for state in fsm.states
                ]
                for fsm in self.traffic_fsms
            ]
        )

    def tick(self, dt):
        for fsm in self.traffic_fsms:
            fsm.tick(dt)
        self.update_current_state_and_time()

    def set_to(self, light_states: List[List[float]]):
        for i, (state, time_remaining) in enumerate(light_states):
            fsm = self.traffic_fsms[i]
            fsm.set_to(int(state), time_remaining)
        self.update_current_state_and_time()

    def reset(self):
        for fsm in self.traffic_fsms:
            fsm.reset()
        self.update_current_state_and_time()

    def update_current_state_and_time(self):
        self._current_state = self.collect_all_current_light_states()
        self._state_per_machine = [
            fsm.current_state.sequence_number for fsm in self.traffic_fsms
        ]
        self._time_remaining = [fsm.time_remaining for fsm in self.traffic_fsms]

    @property
    def current_state(self):
        return self._current_state

    @property
    def current_state_with_name(self):
        return {k: v.name for k, v in self._current_state.items()}

    @property
    def state_per_machine(self):
        return self._state_per_machine

    @property
    def time_remaining(self):
        return self._time_remaining

    @lru_cache
    def get_number_of_light_groups(self):
        return len(self.traffic_fsms)

    def collect_all_current_light_states(self):
        return reduce(
            lambda x, y: {**x, **y},
            [fsm.get_current_actor_states() for fsm in self.traffic_fsms],
            {},
        )


def current_light_state_tensor_from_controller(
        traffic_light_controller: TrafficLightController, traffic_light_ids: List[int]
) -> torch.Tensor:
    return torch.tensor([
        TrafficLightControl._default_allowed_states().index(traffic_light_controller.current_state[str(_id)].name)
        for _id in traffic_light_ids]
    )
