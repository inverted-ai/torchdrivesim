import random

import torch

from torchdrivesim.infractions import collision_detection_with_discs
from torchdrivesim.behavior.common import InitializationFailedError
from torchdrivesim.lanelet2 import pick_random_point_and_orientation


def heuristic_initialize(lanelet_map, agent_num, min_speed=0, max_speed=10, num_attempts_per_agent=500):
    length = 4.97
    width = 2.04
    lr = 1.96

    longitudinal_gap = 1
    lateral_gap = 0.2
    gap = torch.tensor([longitudinal_gap, lateral_gap])

    agent_attributes = []
    agent_states = []
    for i in range(agent_num):
        for _ in range(num_attempts_per_agent):
            x, y, orientation = pick_random_point_and_orientation(lanelet_map)
            speed = random.uniform(min_speed, max_speed)

            if len(agent_states) == 0:
                collides_with_others = False
            else:
                others_boxes = torch.stack([
                    torch.cat([st[:2], at[:2] + gap, st[2:3]], dim=-1)
                    for (at, st) in zip(agent_attributes, agent_states)
                ], dim=-2).unsqueeze(0)
                self_boxes = torch.tensor(
                    [x, y, length, width, orientation]
                ).expand_as(others_boxes)
                collides_with_others = collision_detection_with_discs(self_boxes, others_boxes).bool().any().item()

            if not collides_with_others:
                agent_attributes.append(torch.tensor([
                    length, width, lr
                ]))
                agent_states.append(torch.tensor([
                    x, y, orientation, speed
                ]))
                break

        if not len(agent_states) > i:
            raise InitializationFailedError()

    if agent_num > 0:
        return torch.stack(agent_attributes, dim=-2).unsqueeze(0), torch.stack(agent_states, dim=-2).unsqueeze(0)
    else:
        return torch.zeros(1, 0, 4), torch.zeros(1, 0, 3)
