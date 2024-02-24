"""
Miscellaneous utilities, including for geometric operations on tensors.
"""
import collections
from functools import reduce
from typing import Tuple, List, Dict
from pydantic import BaseModel
from enum import Enum

import os
import math
import random
import numpy as np
import torch
from torch import Tensor

Resolution = collections.namedtuple('Resolution', ['width', 'height'])
RECURRENT_SIZE = 132


def isin(x: Tensor, y: Tensor) -> Tensor:
    """
    Checks whether elements of tensor x are contained in tensor y.
    This function is built-in in torch >= 1.10
    and will be removed from here in the future.

    Args:
        x: any tensor
        y: a one-dimensional tensor
    Returns:
        a boolean tensor with the same shape as x
    """
    assert len(y.shape) == 1
    return (x[..., None] == y).any(-1)


def normalize_angle(angle):
    """
    Normalize to <-pi, pi) range by shifting by a multiple of 2*pi.
    Works with floats, numpy arrays, and torch tensors.
    """
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return angle


def rotation_matrix(theta: Tensor) -> Tensor:
    """
    Counterclockwise rotation matrix in 2D.

    Args:
        theta: tensor of shape Sx1 with rotation angle in radians
    Returns:
        Sx2x2 tensor with the rotation matrix.
    """
    rot_mat = torch.stack([
        torch.cat([torch.cos(theta), - torch.sin(theta)], dim=-1),
        torch.cat([torch.sin(theta), torch.cos(theta)], dim=-1)
    ], dim=-2)
    return rot_mat


def rotate(v: Tensor, angle: Tensor) -> Tensor:
    """
    Rotate the vector counterclockwise (from x towards y).
    Works correctly in batch mode.

    Args:
        v: tensor of shape Sx2 representing points
        angle: tensor of shape Sx1 representing rotation angle
    Returns:
        Sx2 tensor of rotated points
    """
    rot_mat = rotation_matrix(angle)
    rotated = torch.matmul(rot_mat, v.unsqueeze(-1)).squeeze(-1)
    return rotated


def relative(origin_xy: Tensor, origin_psi: Tensor, target_xy: Tensor, target_psi: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Computes position and orientation of the target relative to origin.
    Points are represented as Sx2 tensors of coordinates and Sx1 tensors of orientations in radians.
    """
    rel_xy = rotate(target_xy - origin_xy, - origin_psi)
    rel_psi = normalize_angle(target_psi - origin_psi)
    return rel_xy, rel_psi


def is_inside_polygon(point: Tensor, polygon: Tensor) -> Tensor:
    """
    Checks whether a given point is inside a given convex polygon.
    B and P can be zero or more batch dimensions, the former being batch and the latter points.

    Args:
        point: BxPx2 tensor of x-y coordinates
        polygon: BxNx2 tensor of points specifying a convex polygon in either clockwise or counter-clockwise fashion.
    Returns:
        boolean tensor of shape BxP indicating whether the point is inside the polygon
    """
    batch_dims = len(polygon.shape) - 2
    assert batch_dims >= 0
    assert polygon.shape[:batch_dims] == point.shape[:batch_dims]
    for _ in point.shape[batch_dims:-1]:
        polygon = polygon.unsqueeze(-3)
    edges = torch.stack([polygon, polygon.roll(-1, dims=-2)], dim=-2)
    a = edges[..., 1, 1] - edges[..., 0, 1]
    b = edges[..., 0, 0] - edges[..., 1, 0]
    c = - a * edges[..., 0, 0] - b * edges[..., 0, 1]
    is_right = a * point[..., None, 0] + b * point[..., None, 1] + c >= 0
    all_right = torch.all(is_right, dim=-1)
    all_left = torch.all(is_right.logical_not(), dim=-1)
    return torch.logical_or(all_right, all_left)


def merge_dicts(ds: List[Dict]) -> Dict:
    """
    Merges a sequence of dictionaries, giving preference to entries earlier in the sequence.
    """
    def f(x, y):
        x.update(y)
        return x
    return reduce(f, ds, dict())


def assert_equal(x, y):
    assert x == y


def save_video(imgs, filename, batch_index=0, fps=10, web_browser_friendly=False):
    import cv2
#    os.makedirs(os.path.dirname(filename), exist_ok=True)
    img_stack = [cv2.cvtColor(
        img[batch_index].cpu().numpy().astype(
            np.uint8).transpose(1, 2, 0), cv2.COLOR_RGB2BGR
    )
        for img in imgs]

    w = img_stack[0].shape[0]
    h = img_stack[0].shape[1]
    output_format = cv2.VideoWriter_fourcc(*'mp4v')

    vid_out = cv2.VideoWriter(filename=filename,
                              fourcc=output_format,
                              fps=fps,
                              frameSize=(w, h))

    for frame in img_stack:
        vid_out.write(frame)

    vid_out.release()

    if web_browser_friendly:
        import uuid
        temp_filename = os.path.join(os.path.dirname(
            filename), str(uuid.uuid4()) + '.mp4')
        os.rename(filename, temp_filename)
        os.system(
            f"ffmpeg -y -i {temp_filename} -hide_banner -loglevel error -vcodec libx264 -f mp4 {filename}")
        os.remove(temp_filename)


def set_seeds(seed, logger=None):
    if seed is None:
        seed = np.random.randint(low=0, high=2**32 - 1)
    if logger is not None:
        logger.info(f"seed: {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


class InvertedAIError(Exception):
    """
    Base exception class for Inverted AI Python-API error handling.
    """

    def __init__(
        self,
        message=None,
        http_body=None,
        http_status=None,
        json_body=None,
        headers=None,
        code=None,
    ):
        super(InvertedAIError, self).__init__(message)

        if http_body and hasattr(http_body, "decode"):
            try:
                http_body = http_body.decode("utf-8")
            except BaseException:
                http_body = (
                    "<Could not decode body as utf-8. "
                    "Please report to info@inverted.ai>"
                )

        self._message = message
        self.http_body = http_body
        self.http_status = http_status
        self.json_body = json_body
        self.headers = headers or {}
        self.code = code

    def __str__(self):
        msg = self._message or "<empty message>"
        return msg

    @property
    def user_message(self):
        return self._message

    def __repr__(self):
        return "%s(message=%r, http_status=%r, request_id=%r)" % (
            self.__class__.__name__,
            self._message,
            self.http_status,
        )



class InvalidInput(InvertedAIError):
    """
    Invalid Python API input.
    """
    pass


class Point(BaseModel):
    """
    2D coordinates of a point in a given location.
    Each location comes with a canonical coordinate system, where
    the distance units are meters.
    """

    x: float
    y: float

    @classmethod
    def fromlist(cls, l):
        x, y = l
        return cls(x=x, y=y)

    def __sub__(self, other):
        return math.sqrt((abs(self.x - other.x)**2) + (abs(self.y - other.y)**2))


class TrafficLightState(str, Enum):
    """
    Dynamic state of a traffic light.

    See Also
    --------
    StaticMapActor
    """

    none = "none"  #: The light is off and will be ignored.
    green = "green"
    yellow = "yellow"
    red = "red"


class AgentAttributes(BaseModel):
    """
    Static attributes of the agent, which don't change over the course of a simulation.
    We assume every agent is a rectangle obeying a kinematic bicycle model.

    See Also
    --------
    AgentState
    """

    length: float  #: Longitudinal extent of the agent, in meters.
    width: float  #: Lateral extent of the agent, in meters.
    #: Distance from the agent's center to its rear axis in meters. Determines motion constraints.
    rear_axis_offset: float

    @classmethod
    def fromlist(cls, l):
        length, width, rear_axis_offset = l
        return cls(length=length, width=width, rear_axis_offset=rear_axis_offset)

    def tolist(self):
        """
        Convert AgentAttributes to a flattened list of agent attributes
        in this order: [length, width, rear_axis_offset]
        """
        return [self.length, self.width, self.rear_axis_offset]


class AgentState(BaseModel):
    """
    The current or predicted state of a given agent at a given point.

    See Also
    --------
    AgentAttributes
    """

    center: Point  #: The center point of the agent's bounding box.
    #: The direction the agent is facing, in radians with 0 pointing along x and pi/2 pointing along y.
    orientation: float
    speed: float  #: In meters per second, negative if the agent is reversing.

    def tolist(self):
        """
        Convert AgentState to flattened list of state attributes in this order: [x, y, orientation, speed]
        """
        return [self.center.x, self.center.y, self.orientation, self.speed]

    @classmethod
    def fromlist(cls, l):
        """
        Build AgentState from a list with this order: [x, y, orientation, speed]
        """
        x, y, psi, v = l
        return cls(center=Point(x=x, y=y), orientation=psi, speed=v)


class RecurrentState(BaseModel):
    """
    Recurrent state used in :func:`iai.drive`.
    It should not be modified, but rather passed along as received.
    """

    packed: List[float] = [0.0] * RECURRENT_SIZE
    #: Internal representation of the recurrent state.

    @classmethod
    def check_recurrentstate(cls, values):
        if len(values.get("packed")) == RECURRENT_SIZE:
            return values
        else:
            raise InvalidInput("Incorrect Recurrentstate Size.")

    @classmethod
    def fromval(cls, val):
        return cls(packed=val)
