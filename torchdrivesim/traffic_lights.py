import json
import random
from typing import List
from functools import reduce, lru_cache
from enum import auto, Enum
import logging

logger = logging.getLogger(__name__)



class TrafficLightState(Enum):
    none = auto()
    green = auto()
    yellow = auto()
    red = auto()


class TrafficLightStateMachine:
    def __init__(self, json_file_path: str, tick_to_second_ratio: float = 0.1):
        self._tick_to_second_ratio = tick_to_second_ratio
        self._states = self.load_from_json(json_file_path)
        for state in self._states:
            state['duration'] = float(state['duration'])
        self._time_remaining, self._current_state, self._duration = None, None, None
        self.reset()

    def load_from_json(self, json_file_path: str):
        with open(json_file_path, 'rb') as f:
            light_state_machine = json.load(f)
        return light_state_machine

    def reset(self):
        state = random.randint(0, len(self._states) - 1)
        self.set_to(state, self._states[state]['duration'])

    def set_to(self, state_index: int, time_remaining: float):
        state = state_index
        if state_index < 0:
            state = 0
        elif state_index >= len(self._states):
            state = len(self._states) - 1
        self._current_state = self._states[state]
        self._duration = self._current_state['duration']
        self._time_remaining = time_remaining if time_remaining <= self._duration else self._duration

    def tick(self, dt: float):
        self._time_remaining -= (dt * self._tick_to_second_ratio)
        while self._time_remaining <= 0:
            next_state = int(self._current_state['next_state'])
            next_duration = self._states[next_state]['duration']
            self.set_to(next_state, next_duration)

    @property
    def states(self):
        return self._states

    @property
    def duration(self):
        return self._duration

    @property
    def current_state(self):
        return self._current_state

    @property
    def time_remaining(self):
        return self._time_remaining

    def get_current_actor_states(self):
        return self.current_state['actor_states']


class TrafficLightController:
    def __init__(self, traffic_fsms: List[TrafficLightStateMachine]):
        self.traffic_fsms = traffic_fsms
        self._time_remaining, self._current_state, self._state_per_machine = None, None, None
        self.reset()

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
        self._state_per_machine = [fsm.current_state["state"] for fsm in self.traffic_fsms]
        self._time_remaining = [
            fsm.time_remaining for fsm in self.traffic_fsms]


    @property
    def current_state(self):
        return self._current_state
    
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
        return reduce(lambda x, y: {**x, **y}, [fsm.get_current_actor_states() for fsm in self.traffic_fsms], {})
