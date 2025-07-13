"""
Kinematic models define action spaces for agents and optionally constrain their motion.
The constraints from kinematic models are computed independently for different agents.
We currently provide the kinematic bicycle model and an unconstrained kinematic model,
both with various parameterizations of the action space.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Tuple
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

    @property
    def batch_size(self) -> int:
        return len(self.get_state()[..., 0].flatten())

    @abstractmethod
    def step(self, action: Tensor, dt: Optional[float] = None) -> None:
        """
        Calculates and sets the next state given the current action.

        Args:
            action: BxAc tensor
            dt: time delta, if not specified use object default
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
    
    def extend(self, n: int):
        enlarge = lambda x: x.unsqueeze(1).expand((x.shape[0], n) + x.shape[1:]).reshape((n * x.shape[0],) + x.shape[1:])
        self.map_param(enlarge)
        self.set_state(enlarge(self.get_state()))

    def select_batch_elements(self, idx):
        self.map_param(lambda x: x[idx])
        self.set_state(self.get_state()[idx])


class CompoundKinematicModel(KinematicModel):
    """
    A class that allows combining multiple kinematic models by splitting the batch,
    applying the models, then reassembling the tensor before returning.
    It does not allow duplicate parameter names across models.

    Args:
        models: kinematic models to combine
        batch_assignments: tensor of integers assigning each element of the batch to a model in the list above
    """
    def __init__(self, models: List[KinematicModel], model_assignments: Tensor, dt: float = 0.1):
        super().__init__(dt=dt)
        self.models = models
        self.model_assignments = model_assignments

        self.state_size = max([model.state_size for model in self.models])
        self.action_size = max([model.action_size for model in self.models])

        batch_sizes = [m.batch_size for m in self.models]
        batch_selections = [(self.batch_assignments == i).sum().item() for i in range(len(self.models))]
        if batch_sizes != batch_selections:
            raise ValueError(f"Batch sizes of models do not match how many elements are assigned to them: {batch_sizes} vs {batch_selections}")
        
        _ = self.get_params()  # Check for duplicate parameter names

    @property
    def batch_assignments(self) -> Tensor:
        return self.model_assignments.flatten()

    @property
    def batch_size(self) -> int:
        return len(self.batch_assignments)
    
    @property
    def batch_shape(self) -> torch.Size:
        return self.model_assignments.shape

    def step(self, action: Tensor, dt: Optional[float] = None) -> None:
        action_flattened = action.flatten(0, -2)
        action_splits = [action_flattened[self.batch_assignments == i, :model.action_size] for (i, model) in enumerate(self.models)]
        for model, action_split in zip(self.models, action_splits):
            model.step(action_split, dt=dt)

    def fit_action(self, future_state: Tensor, current_state: Optional[Tensor] = None,
                   dt: Optional[float] = None) -> Tensor:
        if current_state is None:
            current_state = self.get_state()
        future_state_flattened = future_state.flatten(0, -2)
        current_state_flattened = current_state.flatten(0, -2)
        future_state_splits = [future_state_flattened[self.batch_assignments == i, :model.state_size] for (i, model) in enumerate(self.models)]
        current_state_splits = [current_state_flattened[self.batch_assignments == i, :model.state_size] for (i, model) in enumerate(self.models)]
        action_splits = [model.fit_action(f, c, dt=dt) for model, f, c in zip(self.models, future_state_splits, current_state_splits)]
        padded_action_splits = [torch.nn.functional.pad(a, (0, self.action_size - a.shape[-1])) for a in action_splits]
        action_flattened = torch.zeros_like(torch.cat(padded_action_splits, dim=0))
        for i, action_split in enumerate(padded_action_splits):
            action_flattened[self.batch_assignments == i] = action_split
        action = action_flattened.reshape(future_state.shape[:-1] + (self.action_size,))
        return action

    def copy(self, other=None):
        if other is None:
            other = self.__class__(models=[m.copy() for m in self.models], model_assignments=self.model_assignments, dt=self.dt)
        other.set_params(**self.get_params())
        other.set_state(self.get_state())
        return other
    
    def to(self, device):
        other = super().to(device)
        for model in other.models:
            model.to(device)
        other.model_assignments = other.model_assignments.to(device)
        return other
    
    def extend(self, n: int):
        enlarge = lambda x: x.unsqueeze(1).expand((x.shape[0], n) + x.shape[1:]).reshape((n * x.shape[0],) + x.shape[1:])
        state = self.get_state()
        self.model_assignments = enlarge(self.model_assignments)
        self.map_param(enlarge)
        self.set_state(enlarge(state))

    def select_batch_elements(self, idx):
        self.model_assignments = self.model_assignments[idx]
        # raise NotImplementedError()  # TODO
        # self.map_param(lambda x: x[idx])
        # self.set_state(self.get_state()[idx])

    def set_state(self, state: Tensor) -> None:
        state_flattened = state.flatten(0, -2)
        state_splits = [state_flattened[self.batch_assignments == i, :model.state_size] for (i, model) in enumerate(self.models)]
        for model, state_split in zip(self.models, state_splits):
            model.set_state(state_split)

    def get_state(self) -> Tensor:
        state_splits = [model.get_state() for model in self.models]
        padded_state_splits = [torch.nn.functional.pad(s, (0, self.state_size - s.shape[-1])) for s in state_splits]
        state_flattened = torch.zeros_like(torch.cat(padded_state_splits, dim=0))
        for i, state_split in enumerate(padded_state_splits):
            state_flattened[self.batch_assignments == i] = state_split
        state = state_flattened.reshape(self.batch_shape + (self.state_size,))
        return state

    def get_params(self) -> Dict[str, Tensor]:
        params_splits = [model.get_params() for model in self.models]
        if len(set(key for params in params_splits for key in params.keys())) != sum(len(params) for params in params_splits):
            all_names = [p for params in params_splits for p in params.keys()]
            duplicate_names = set([name for name in all_names if all_names.count(name) > 1])
            raise ValueError(f"Duplicate parameter names in CompoundKinematicModel: {duplicate_names}")
        padded_param_splits = [{k: torch.zeros((self.batch_size,) + v.shape[1:], dtype=v.dtype, device=v.device) for k, v in params.items()}
                               for params in params_splits]
        for i, (padding, params) in enumerate(zip(padded_param_splits, params_splits)):
            for name, value in params.items():
                padding[name][self.batch_assignments == i] = value
        params = {name: value.reshape(self.batch_shape) for params in padded_param_splits for name, value in params.items()}
        return params


    def set_params(self, **kwargs) -> None:
        for i, model in enumerate(self.models):
            model_params = model.get_params()
            matching_kwargs = {k: v.flatten()[self.batch_assignments == i] for k, v in kwargs.items() if k in model_params}
            model.set_params(**matching_kwargs)

    def flattening(self, batch_shape) -> None:
        for model in self.models:
            model.flattening(batch_shape)

    def unflattening(self, batch_shape) -> None:
        for model in self.models:
            model.unflattening(batch_shape)

    def map_param(self, f) -> None:
        for model in self.models:
            model.map_param(f)

    def normalize_action(self, action: Tensor) -> Tensor:
        action_flattened = action.flatten(0, -2)
        action_splits = [action_flattened[self.batch_assignments == i, :model.action_size] for (i, model) in enumerate(self.models)]
        normalized_action_splits = [model.normalize_action(a) for model, a in zip(self.models, action_splits)]
        padded_normalized_action_splits = [torch.nn.functional.pad(a, (0, self.action_size - a.shape[-1])) for a in normalized_action_splits]
        normalized_action_flattened = torch.zeros_like(torch.cat(padded_normalized_action_splits, dim=0))
        for i, normalized_action_split in enumerate(padded_normalized_action_splits):
            normalized_action_flattened[self.batch_assignments == i] = normalized_action_split
        normalized_action = normalized_action_flattened.reshape(action.shape)
        return normalized_action

    def denormalize_action(self, action: Tensor) -> Tensor:
        action_flattened = action.flatten(0, -2)
        action_splits = [action_flattened[self.batch_assignments == i, :model.action_size] for (i, model) in enumerate(self.models)]
        denormalized_action_splits = [model.denormalize_action(a) for model, a in zip(self.models, action_splits)]
        padded_denormalized_action_splits = [torch.nn.functional.pad(a, (0, self.action_size - a.shape[-1])) for a in denormalized_action_splits]
        denormalized_action_flattened = torch.zeros_like(torch.cat(padded_denormalized_action_splits, dim=0))
        for i, denormalized_action_split in enumerate(padded_denormalized_action_splits):
            denormalized_action_flattened[self.batch_assignments == i] = denormalized_action_split
        denormalized_action = denormalized_action_flattened.reshape(action.shape)
        return denormalized_action


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
        other._normalization_factor = self._normalization_factor.clone()
        return super().copy(other)

    def to(self, device):
        other = super().to(device)
        other._normalization_factor = other._normalization_factor.to(device)
        return other

    def normalize_action(self, action):
        return action / self._normalization_factor

    def denormalize_action(self, action):
        return action * self._normalization_factor

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
        other._normalization_factor = self._normalization_factor.clone()
        return super().copy(other)
    
    def to(self, device):
        other = super().to(device)
        other._normalization_factor = other._normalization_factor.to(device)
        return other

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
        return action / self._normalization_factor

    def denormalize_action(self, action):
        return action * self._normalization_factor

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
        other._xy_normalization_tensor = self._xy_normalization_tensor.clone()
        return super().copy(other)
    
    def to(self, device):
        other = super().to(device)
        other._xy_normalization_tensor = other._xy_normalization_tensor.to(device)
        return other

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
        action = action / self._xy_normalization_tensor
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
