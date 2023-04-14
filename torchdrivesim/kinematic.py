"""
Kinematic models define action spaces for agents and optionally constrain their motion.
The constraints from kinematic models are computed independently for different agents.
We currently provide the kinematic bicycle model and an unconstrained kinematic model,
both with various parameterizations of the action space.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple
import logging

import numpy as np
import torch
from torch import Tensor

from torchdrivesim.utils import rotate

logger = logging.getLogger(__name__)


class KinematicModel(ABC):
    """
    A generic kinematic base class. Minimum subclass needs to define `action_size`, `step` and `fit_action`.
    Designed to be used in batch mode. Subclasses may include additional model parameters with arbitrary names.

    Args:
        dt: Default time unit in seconds.
    """

    state_size: int = 4  #: length St of the state vector, by default x,y,orientation,speed
    action_size: int = 4  #: length Ac of the action vector, by default matches state

    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self.state = None

    @abstractmethod
    def step(self, action: Tensor, dt: Optional[float] = None) -> None:
        """
        Calculates and sets the next state given the current action.

        Args:
            action: BxAc tensor
            dt: time delta, if not specified use object default
        Returns:
            BxSt tensor representing the new state at time t+dt
        """
        raise NotImplementedError

    @abstractmethod
    def fit_action(self, future_state: Tensor, current_state: Optional[Tensor] = None,
                   dt: Optional[float] = None) -> Tensor:
        """
        Produces an action that applied to the current state would produce the given state,
        or some approximation thereof.

        Args:
            future_state: BxSt tensor representing state to achieve
            current_state: BxSt tensor, if not specified use object's state
            dt: time step in seconds, if different from default
        Returns:
            BxAc action tensor
        """
        raise NotImplementedError

    def copy(self, other=None):
        """
        Returns a shallow copy of the current object.
        Optionally takes a target which will be copied into.
        """
        if other is None:
            other = self.__class__(dt=self.dt)
        other.set_params(**self.get_params())
        other.set_state(self.get_state())
        return other

    def to(self, device: torch.device):
        """
        Moves all tensors to a given device in place.
        """
        if self.state is not None:
            self.state = self.state.to(device)
        self.map_param(lambda x: x.to(device))
        return self

    def set_state(self, state: Tensor) -> None:
        self.state = state

    def get_state(self) -> Tensor:
        return self.state

    def get_params(self) -> Dict[str, Tensor]:
        """
        Returns a dictionary of model parameters.
        """
        return dict()

    def set_params(self, **kwargs) -> None:
        """
        Set custom parameters of the model, specified as tensors.
        """
        pass

    def flattening(self, batch_shape) -> None:
        """
        Flattens batch dimensions for model parameters in place.
        """
        pass

    def unflattening(self, batch_shape) -> None:
        """
        Reverse of `flattening`.
        """
        pass

    def map_param(self, f) -> None:
        """
        Apply a function to all the parameters of the model.
        """
        pass

    def normalize_action(self, action: Tensor) -> Tensor:
        """
        Normalizes the action to fit it in [-1,1] interval.
        Typically used to train neural policies.
        """
        return action

    def denormalize_action(self, action: Tensor) -> Tensor:
        """
        Reverse of `normalize_action`.
        """
        return action

    @staticmethod
    def pack_state(x: Tensor, y: Tensor, psi: Tensor, speed: Tensor) -> Tensor:
        """
        Packs the given state components as a BxSt tensor. Inputs should have shape (B,).
        """
        return torch.stack([x, y, psi, speed], dim=-1)

    @staticmethod
    def unpack_state(state: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Reverse of `pack_state`.
        """
        return state[..., 0], state[..., 1], state[..., 2], state[..., 3]


class TeleportingKinematicModel(KinematicModel):
    """
    A trivial kinematic model where the action is the next state.
    """
    def step(self, action, dt=None):
        self.set_state(action)

    def fit_action(self, future_state, current_state=None, dt=None):
        return future_state


class SimpleKinematicModel(KinematicModel):
    """
    A simple kinematic model where the action is the gradient of the state vector with respect to time.
    The action is specified in units of constructor arguments.

    Args:
        max_dx: Normalization factor for action in x and y.
        max_dpsi: Normalization factor for action in orientation.
        max_dv: Normalization factor for action in speed.
    """
    def __init__(self, max_dx=20, max_dpsi=10*np.pi, max_dv=5, dt=0.1):
        super().__init__(dt=dt)
        self.max_dx = max_dx
        self.max_dpsi = max_dpsi
        self.max_dv = max_dv
        self._normalization_factor = torch.tensor([self.max_dx, self.max_dx, self.max_dpsi, self.max_dv])

    def copy(self, other=None):
        if other is None:
            other = self.__class__(max_dx=self.max_dx, max_dv=self.max_dv, dt=self.dt)
        return super().copy(other)

    def normalize_action(self, action):
        return action / self._normalization_factor.to(action.device).to(action.dtype)

    def denormalize_action(self, action):
        return action * self._normalization_factor.to(action.device).to(action.dtype)

    def step(self, action, dt=None):
        if dt is None:
            dt = self.dt
        assert action.shape[-1] == self.action_size
        action = self.denormalize_action(action)
        self.set_state(self.get_state() + action * dt)

    def fit_action(self, future_state, current_state=None, dt=None):
        if dt is None:
            dt = self.dt
        if current_state is None:
            current_state = self.get_state()
        action = (future_state - current_state) / dt
        action = self.normalize_action(action)
        return action


class OrientedKinematicModel(SimpleKinematicModel):
    """
    Just like `SimpleKinematicModel`, but the action coordinate frame rotates with the agent,
    so that the x-axis of the action space always points forward.
    """
    def step(self, action, dt=None):
        assert action.shape[-1] == self.action_size
        psi = self.get_state()[..., 2:3]
        xy = rotate(action[..., :2], psi)
        action = torch.cat([xy, action[..., 2:]], dim=-1)
        super().step(action, dt=dt)

    def fit_action(self, future_state, current_state=None, dt=None):
        parent_action = super().fit_action(future_state, current_state=current_state, dt=dt)
        if current_state is None:
            current_state = self.get_state()
        current_psi = current_state[..., 2:3]
        xy = rotate(parent_action[..., :2], - current_psi)
        return torch.cat([xy, parent_action[...,2:]], dim=-1)


class KinematicBicycle(KinematicModel):
    """
    A kinematic bicycle model where steering is applied to the geometric center directly as beta.
    The only parameter is the distance between the center and the rear axis, called 'lr'.
    The action space is (acceleration, steering), in the units specified by constructor arguments.
    By default, steering is constrained to a right angle, so negative speed is needed to reverse.

    Args:
        max_acceleration: Normalization factor for acceleration.
        max_steering: Normalization factor for steering.
        dt: Default time step length.
        left_handed: Set if using a left-handed coordinate system for portability to right-handed coordinates.
    """
    action_size: int = 2

    def __init__(self, max_acceleration=5, max_steering=np.pi/2, dt=0.1, left_handed=False):
        super().__init__(dt=dt)
        self.max_acceleration = max_acceleration
        self.max_steering = max_steering
        self.left_handed = left_handed
        self._normalization_factor = torch.tensor([self.max_acceleration, self.max_steering])

        self.lr = None

    def copy(self, other=None):
        if other is None:
            other = self.__class__(max_acceleration=self.max_acceleration, dt=self.dt, left_handed=self.left_handed)
        return super().copy(other)

    def get_params(self):
        params = super().get_params()
        params['lr'] = self.lr
        return params

    def set_params(self,  **kwargs):
        assert 'lr' in kwargs
        self.lr = kwargs['lr']

    def flattening(self, batch_shape):
        assert self.lr is not None
        self.lr = self.lr.reshape((int(np.prod(batch_shape)),))

    def unflattening(self, batch_shape):
        assert self.lr is not None
        self.lr = self.lr.reshape(batch_shape)

    def map_param(self, f):
        assert self.lr is not None
        self.lr = f(self.lr)

    def normalize_action(self, action):
        return action / self._normalization_factor.to(action.device).to(action.dtype)

    def denormalize_action(self, action):
        return action * self._normalization_factor.to(action.device).to(action.dtype)

    def step(self, action, dt=None):
        assert action.shape[-1] == 2, "The bicycle model takes as input only acceleration and steering"
        action = self.denormalize_action(action)
        a, beta = action[..., 0], action[..., 1]
        if self.left_handed:
            beta = - beta  # Flip steering angle when using left-hand coordinate system
        if dt is None:
            dt = self.dt
        x, y, psi, v = self.unpack_state(self.get_state())
        v = v + a * dt
        x = x + v * torch.cos(psi + beta) * dt
        y = y + v * torch.sin(psi + beta) * dt
        psi = psi + (v / self.lr) * torch.sin(beta) * dt
        # psi = (np.pi + psi) % (2 * np.pi) - np.pi # Normalize angle between -pi and pi

        self.set_state(self.pack_state(x, y, psi, v))

    def fit_action(self, future_state, current_state=None, dt=None):
        if dt is None:
            dt = self.dt
        f_x, f_y, f_psi, f_v = self.unpack_state(future_state)
        if current_state is not None:
            c_x, c_y, c_psi, c_v = self.unpack_state(current_state)
        else:
            c_x, c_y, c_psi, c_v = self.unpack_state(self.get_state())

        vx = (f_x - c_x) / dt
        vy = (f_y - c_y) / dt
        v = torch.sqrt(vx ** 2 + vy ** 2)
        #The following two lines of code calculate beta such that it is modulated to
        #-pi pi, and the calculated steering angle is set to 0
        #if the velocity is set to 0
        beta = torch.atan2(vy, vx) - c_psi * torch.sign(torch.abs(v))
        beta = torch.remainder(beta + np.pi, 2*np.pi) - np.pi
        reversing = torch.sign(torch.cos(beta)) == -1  # ensures reversing=1 when beta=+-pi/2
        v = torch.sqrt(vx ** 2 + vy ** 2) * torch.where(reversing, -1, 1)
        beta = torch.where(reversing, beta - np.pi * torch.sign(beta), beta)
        a = (v - c_v) / dt
        if self.left_handed:
            beta = - beta  # Flip steering angle when using left-hand coordinate system

        action = torch.stack([a, beta], dim=-1)
        action = self.normalize_action(action)

        return action


class BicycleNoReversing(KinematicBicycle):
    """
    Modified bicycle model that brings a vehicle to full stop when it attempts to reverse.
    """
    def step(self, action, dt=None):
        if dt is None:
            dt = self.dt
        action = self.denormalize_action(action)
        acc, beta = action[..., 0], action[..., 1]
        _, _, _, v = self.unpack_state(self.get_state())
        reversing = v + acc * dt < 0
        modified_acc = torch.where(reversing, - v / dt, acc)
        modified_action = torch.stack([modified_acc, beta], dim=-1)
        modified_action = self.normalize_action(modified_action)
        super().step(modified_action, dt=dt)


class BicycleByDisplacement(KinematicBicycle):
    """
    Similar to `SimpleKinematicModel`, but the action space is directed velocity.
    """
    def __init__(self, max_dx=20, dt=0.1):
        super().__init__(dt=dt)
        self.max_dx = max_dx
        self._xy_normalization_tensor = torch.tensor([self.max_dx, self.max_dx])

    def copy(self, other=None):
        if other is None:
            other = self.__class__(max_dx=self.max_dx, dt=self.dt)
        return super().copy(other)

    def step(self, action, dt=None):
        assert action.shape[-1] == 2  # x and y displacement
        self.step_from_xy(action[..., :2], dt=dt)

    def step_from_xy(self, xy, dt=None):
        if dt is None:
            dt = self.dt
        action = xy * self._xy_normalization_tensor.to(xy.device).to(xy.dtype)
        dx, dy = action[..., 0], action[..., 1]
        # implicitly using the fact that bicycle model ignores psi and v when fitting the action
        x, y, psi, v = self.unpack_state(self.get_state())
        bicycle_action = super().fit_action(self.pack_state(x + dx * dt, y + dy * dt, psi, v))
        super().step(bicycle_action, dt=dt)

    def fit_action(self, future_state, current_state=None, dt=None):
        if dt is None:
            dt = self.dt
        xf, yf, psif, vf = self.unpack_state(future_state)
        if current_state is None:
            xp, yp, psip, vp = self.unpack_state(self.get_state())
        else:
            xp, yp, psip, vp = self.unpack_state(current_state)
        action = torch.stack([(xf - xp) / dt, (yf - yp) / dt], dim=-1)
        action = action / self._xy_normalization_tensor.to(action.device).to(action.dtype)
        return action


class BicycleByOrientedDisplacement(BicycleByDisplacement):
    """
    A combination of `BicycleByDisplacement` and `OrientedKinematicModel`.
    """
    def step_from_xy(self, xy, dt=None):
        psi = self.get_state()[..., 2:3]
        xy = rotate(xy, psi)
        super().step_from_xy(xy, dt=dt)

    def fit_action(self, future_state, current_state=None, dt=None):
        action = super().fit_action(future_state, current_state=current_state, dt=dt)
        if current_state is None:
            current_state = self.get_state()
        psi = current_state[..., 2:3]
        return rotate(action[..., :2], - psi)
