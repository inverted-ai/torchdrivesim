"""
Custom classes for representing various triangular meshes as tensors.
"""
import copy
import dataclasses
import json
import logging
import math
import os
import pickle
import pickle as default_pickle_module
from dataclasses import dataclass
from typing import Union, Tuple, List, Dict, Optional, Any
from typing_extensions import Self

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from torchdrivesim import assert_pytorch3d_available
from torchdrivesim.traffic_controls import BaseTrafficControl, TrafficLightControl
from torchdrivesim.utils import is_inside_polygon, merge_dicts, rotate, transform

logger = logging.getLogger(__name__)

Color = Union[Tensor
, Tuple[int, int, int]]


def tensor_color(color: Color, device: torch.device = 'cpu', dtype=torch.float) -> Tensor\
        :
    """
    Converts all supported color representations to the tensor representation.

    Args:
        color: RGB color as either an int 3-tuple in [0,255] range or tensor of shape (3,) in [0,1] range
        device: device to place the resulting tensor on, ignored if color is already a tensor
        dtype: element type for the resulting tensor, ignored if color is already a tensor
    Returns:
        RGB tensor of shape (3,) in [0,1] range
    """
    if not isinstance(color, Tensor
                      ):
        color = torch.tensor(color, device=device, dtype=dtype) / 255.0
    return color


class BadMeshFormat(RuntimeError):
    """
    The mesh received had the wrong format, usually indicating loading from old cache.
    """
    pass


@dataclass
class BaseMesh:
    """
    Basic triangle mesh in a space of arbitrary dimensions Dim.
    Only specifies triangles, with no additional properties.
    Always includes exactly one batch dimension.
    """
    verts: Tensor
    #: BxVxDim
    faces: Tensor
    #: BxFx3 indexing into verts
    _verts_fill: float = dataclasses.field(default=0.0, init=False)  # padding value for vertices
    _faces_fill: int = dataclasses.field(default=0, init=False)  # pytorch3d uses -1, but requires padding at the end

    def __post_init__(self):
        # self._validate_input()  # disabled for efficiency
        pass

    def _validate_input(self):
        if len(self.verts.shape) == 2:
            self.verts = self.verts.unsqueeze(0)  # add batch dimension
        assert len(self.verts.shape) == 3
        if len(self.faces.shape) == 2:
            self.faces = self.faces.unsqueeze(0)
        assert len(self.faces.shape) == 3
        assert self.faces.shape[-1] == 3
        batch_size = self.batch_size
        if batch_size > 1:
            if self.verts.shape[0] == 1:
                self.verts = self.verts.expand((batch_size, -1, -1))
            elif self.verts.shape[0] != batch_size:
                raise ValueError
            if self.faces.shape[0] == 1:
                self.faces = self.verts.expand((batch_size, -1, -1))
            elif self.faces.shape[0] != batch_size:
                raise ValueError

    @property
    def dim(self) -> int:
        """
        Dimension of the space in which the mesh lives, usually 2 or 3.
        """
        return self.verts.shape[-1]

    @property
    def batch_size(self) -> int:
        return max(self.verts.shape[0], self.faces.shape[0])

    @property
    def verts_count(self) -> int:
        return self.verts.shape[-2]

    @property
    def faces_count(self) -> int:
        return self.faces.shape[-2]

    @property
    def device(self) -> torch.device:
        return self.verts.device

    @property
    def center(self) -> Tensor:
        """
        A BxDim center of the mesh, calculated as the average of min and max vertex coordinates for each dimension.
        Note that if the vertices are padded, the padding value may distort this result.
        """
        if self.verts_count > 0:
            return (self.verts.max(dim=-2).values + self.verts.min(dim=-2).values) / 2
        else:
            return torch.zeros((self.batch_size, 2), dtype=self.verts.dtype, device=self.device)

    def to(self, device: torch.device):
        """
        Moves all tensors to a given device, returning a new mesh object.
        Does not modify the existing mesh object.
        """
        return dataclasses.replace(self, verts=self.verts.to(device), faces=self.faces.to(device))

    def clone(self):
        """
        Deep copy of the mesh.
        """
        return copy.deepcopy(self)

    def expand(self, size: int):
        """
        Expands batch dimension on the right by the given factor, returning a new mesh.
        """
        f = lambda x: x.unsqueeze(1).expand((self.batch_size, size, *x.shape[1:])).flatten(0, 1)
        return dataclasses.replace(self, verts=f(self.verts), faces=f(self.faces))

    def select_batch_elements(self, idx: Tensor):
        """
        Selects given indices from batch dimension, possibly with repetitions, returning a new mesh
        """
        f = lambda x: x[idx]
        return dataclasses.replace(self, verts=f(self.verts), faces=f(self.faces))

    def translate(self, xy: torch.Tensor, inplace: bool = True) -> Self:
        """
        Shifts the mesh by the given coordinate so that it is at the origin (0,0).

        Args:
            xy: Bx2 tensor specifying the camera position
            inplace: whether to modify the mesh in place
        """
        shifted_mesh = self if inplace else self.clone()
        shifted_mesh.verts = shifted_mesh.verts.clone()
        shifted_mesh.verts[..., :2] += xy.unsqueeze(1)
        return shifted_mesh

    def __getitem__(self, item):  # square bracket syntax for batch element selection
        return self.select_batch_elements(item)

    @classmethod
    def collate(cls, meshes):
        """
        Batches a collection of meshes with appropriate padding.
        All input meshes must have a singleton batch dimension.
        """
        verts = pad_sequence(
            [m.verts.squeeze(0) for m in meshes], batch_first=True, padding_value=cls._verts_fill
        )
        faces = pad_sequence(
            [m.faces.squeeze(0) for m in meshes], batch_first=True, padding_value=cls._faces_fill
        )
        return cls(verts=verts, faces=faces)

    @classmethod
    def concat(cls, meshes):
        """
        Concatenates multiple meshes to form a single scene.
        """
        # if -1 was used for padding faces, pytorch3d would require padding at the back
        verts = torch.cat([m.verts for m in meshes], dim=-2)
        face_offsets = [0] + list(np.cumsum([m.verts_count for m in meshes]))[:-1]
        faces = torch.cat([m.faces + offset for (m, offset) in zip(meshes, face_offsets)], dim=-2)
        return cls(verts=verts, faces=faces)

    def merge(self, other):
        """
        Merges the current mesh with another to form a single scene, returning a new mesh.
        """
        return self.concat([self, other])

    def offset(self, offset: Tensor):
        """
        Shifts the mesh by a given distance, returning a new mesh.
        Missing dimensions in the argument are padded with zeros if needed.
        """
        if offset.shape[-1] < self.dim:
            offset = torch.cat(
                [offset, torch.zeros(offset.shape[:-1] + (self.dim - offset.shape[-1],),
                dtype=offset.dtype, device=offset.device)]
            )
        return dataclasses.replace(self, verts=self.verts + offset)

    def pytorch3d(self, include_textures=True) -> "pytorch3d.structures.Meshes":
        """
        Converts the mesh to a PyTorch3D one.
        For the base class there are no textures, but subclasses may include them.
        Empty meshes are augmented with a single degenerate face on conversion,
        since PyTorch3D does not handle empty meshes correctly.
        """
        assert_pytorch3d_available()
        import pytorch3d
        assert self.dim in [2, 3]
        if self.faces_count == 0:
            verts = torch.zeros((self.batch_size, 1, self.dim), device=self.device, dtype=self.verts.dtype)
            faces = torch.zeros((self.batch_size, 1, 3), device=self.device, dtype=self.faces.dtype)
        else:
            verts = self.verts
            faces = self.faces
        if self.dim == 2:
            verts = torch.cat([verts, torch.zeros_like(verts[..., :1])], dim=-1)
        return pytorch3d.structures.Meshes(verts=verts, faces=faces)

    def pickle(self, mesh_file_path: str):
        """
        Store this mesh to a given file.
        """
        if not os.path.exists(os.path.dirname(mesh_file_path)):
            os.makedirs(os.path.dirname(mesh_file_path))
        with open(mesh_file_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def unpickle(cls, mesh_file_path: str, pickle_module: Any = default_pickle_module):
        """
        Load a mesh of this type from the given file.
        """
        with open(mesh_file_path, 'rb') as f:
            road_mesh = pickle_module.Unpickler(f).load()
        if isinstance(road_mesh, BaseMesh):
            return road_mesh
        else:
            raise BadMeshFormat

    def serialize(self):
        return {
            'verts': self.verts.tolist(),
            'faces': self.faces.tolist()
        }

    def save(self, file_save_path: str):
        """
        Save the attributes of the mesh object in a json where tensors are converted to lists.
        """
        directory = os.path.dirname(file_save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(os.path.dirname(file_save_path), exist_ok=True)
        data = self.serialize()
        with open(file_save_path, 'w') as file:
            json.dump(data, file)

    @classmethod
    def _deserialize_tensors(cls, data: Dict) -> Dict:
        """
        Convert list attributes to tensor attributes if there is any tensor attribute
        """
        new_data = data.copy()
        new_data.update(verts=torch.tensor(data['verts']), faces=torch.tensor(data['faces']))
        return new_data

    @classmethod
    def deserialize(cls, data: Dict) -> Self:
        return cls(**cls._deserialize_tensors(data))

    @classmethod
    def load(cls, filepath):
        try:
            with open(filepath, 'r') as file:
                data = json.load(file)
            return cls.deserialize(data)
        except Exception as e:
            logger.error(e)
            raise BadMeshFormat

    @classmethod
    def empty(cls, dim: int = 2, batch_size: int = 1):
        """
        Create empty mesh.
        """
        verts = torch.zeros((batch_size, 0, dim), dtype=torch.float)
        faces = torch.zeros((batch_size, 0, 3), dtype=torch.int)
        return cls(verts=verts, faces=faces)

    def _trim_and_return_verts_and_faces(self, vertices_to_keep: Tensor, trim_face_only=False):
        assert len(vertices_to_keep.shape) == 2, "Batch dimension needed for polygon tensor."
        mesh_verts, mesh_faces = self.verts, self.faces.long()
        selected_faces_idx = torch.gather(
            vertices_to_keep.long().unsqueeze(1).expand(-1, mesh_faces.shape[1], -1),
            index=mesh_faces, dim=-1).any(dim=-1)  # BxF
        padded_selected_faces = pad_sequence(
            [mesh_faces[i, selected] for i, selected in enumerate(selected_faces_idx)],
            batch_first=True
        )  # BxFsx3
        trimmed_mesh_faces = padded_selected_faces
        if trim_face_only:
            trimmed_mesh_verts = mesh_verts
            inside_polygon_idx_with_outside_dependent = None
        else:
            if trimmed_mesh_faces.shape[1] < 1:
                inside_polygon_idx_with_outside_dependent = torch.zeros(
                    trimmed_mesh_faces.shape[0], 0,
                    device=trimmed_mesh_faces.device)
            else:
                inside_polygon_idx_with_outside_dependent = pad_sequence(
                    [x.unique() for x in trimmed_mesh_faces.flatten(start_dim=1)],
                    batch_first=True
                )  # BxVs
            trimmed_mesh_verts = torch.gather(
                mesh_verts, 1,
                inside_polygon_idx_with_outside_dependent.unsqueeze(-1).expand(-1, -1, mesh_verts.shape[-1])
            )  # BxVsx3
            new_verts_idx = torch.zeros(mesh_verts.shape[:2], dtype=torch.long, device=mesh_verts.device)
            new_verts_idx.scatter_(
                1,
                inside_polygon_idx_with_outside_dependent,
                torch.arange(
                    0,
                    inside_polygon_idx_with_outside_dependent.shape[1],
                    device=inside_polygon_idx_with_outside_dependent.device
                ).unsqueeze(0).expand(new_verts_idx.shape[0], -1)
            )
            offset_idx = new_verts_idx.unsqueeze(-1).expand(-1, -1, 3)
            trimmed_mesh_faces = torch.gather(offset_idx, dim=1, index=trimmed_mesh_faces)
        return trimmed_mesh_verts, trimmed_mesh_faces, inside_polygon_idx_with_outside_dependent

    def trim(self, polygon: Tensor, trim_face_only: bool = False):
        """
        Crops the mesh to a given 2D convex polygon, returning a new mesh.
        Faces where all vertices are outside the polygon are removed,
        even if the triangle intersects with the polygon.
        Vertices are removed if they are not used by any face.

        Args:
            polygon: BxPx2 tensor specifying a convex polygon in either clockwise or counterclockwise fashion
            trim_face_only: whether to keep all vertices, including those unused by any faces
        """
        if not self.dim == 2:
            raise NotImplementedError(f"Trimming mesh to a polygon in {self.dim} dimensions")
        trimmed_mesh_verts, trimmed_mesh_faces, _ = self._trim_and_return_verts_and_faces(
            is_inside_polygon(self.verts, polygon), trim_face_only
        )
        trimmed_base_mesh = BaseMesh(trimmed_mesh_verts, trimmed_mesh_faces)
        mesh_with_trims = dataclasses.replace(trimmed_base_mesh, verts=trimmed_mesh_verts, faces=trimmed_mesh_faces)

        return mesh_with_trims


@dataclass
class AttributeMesh(BaseMesh):
    """
    Endows each vertex with an attribute.
    An attribute is a vector of arbitrary dimension Attr, usually a color or something similar.
    Typically, in any given face all vertices have the same attribute values.
    """
    attrs: Tensor  #: BxVxAttr_dim specifying attributes for all vertices
    _attrs_fill: float = dataclasses.field(default=0.0, init=False)

    def _validate_input(self):
        super()._validate_input()
        if len(self.attrs.shape) == 2:
            self.attrs = self.attrs.unsqueeze(0)
        assert len(self.attrs.shape) == 3
        assert self.attrs.shape[-2] == self.verts.shape[-2]
        batch_size = self.batch_size
        if batch_size > 1:
            if self.attrs.shape[0] == 1:
                self.attrs = self.attrs.expand((batch_size, -1, -1))
            elif self.attrs.shape[0] != batch_size:
                raise ValueError

    @property
    def attr_dim(self) -> int:
        """
        Size of the attribute dimension.
        """
        return self.attrs.shape[-1]

    @classmethod
    def set_attr(cls, mesh: BaseMesh, attr: Tensor):
        """
        Sets a given attribute value for all vertices in a given mesh.
        """
        assert len(attr.shape) == 1
        attrs = attr.expand(mesh.verts.shape[:-1] + attr.shape)
        return cls(verts=mesh.verts, faces=mesh.faces, attrs=attrs)

    def to(self, device: torch.device):
        return dataclasses.replace(
            self, verts=self.verts.to(device), faces=self.faces.to(device), attrs=self.attrs.to(device)
        )

    def expand(self, size: int):
        f = lambda x: x.unsqueeze(1).expand((self.batch_size, size, *x.shape[1:])).flatten(0, 1)
        return dataclasses.replace(self, verts=f(self.verts), faces=f(self.faces), attrs=f(self.attrs))

    def select_batch_elements(self, idx: int):
        f = lambda x: x[idx]
        return dataclasses.replace(self, verts=f(self.verts), faces=f(self.faces), attrs=f(self.attrs))

    @classmethod
    def concat(cls, meshes):
        base_concat = BaseMesh.concat(meshes)
        attrs = torch.cat([m.attrs for m in meshes], dim=-2)
        return cls(verts=base_concat.verts, faces=base_concat.faces, attrs=attrs)

    @classmethod
    def collate(cls, meshes):
        base_collated = BaseMesh.collate(meshes)
        attrs = pad_sequence(
            [m.attrs.squeeze(0) for m in meshes], batch_first=True, padding_value=cls._attrs_fill
        )
        return cls(verts=base_collated.verts, faces=base_collated.faces, attrs=attrs)

    def pytorch3d(self, include_textures=True) -> "pytorch3d.structures.Meshes":
        """
        PyTorch3D uses per-face textures, which are obtained by averaging attributes of the face.
        The resulting texture for each face is constant.
        """
        assert_pytorch3d_available()
        import pytorch3d
        assert self.dim in [2, 3]
        if not include_textures:
            return super().pytorch3d(include_textures=False)

        if self.faces_count == 0:
            verts = torch.zeros((self.batch_size, 1, 3), device=self.device, dtype=self.verts.dtype)
            faces = torch.zeros((self.batch_size, 1, 3), device=self.device, dtype=self.faces.dtype)
            attrs = torch.zeros((self.batch_size, 1, self.attr_dim), device=self.device, dtype=self.attrs.dtype)
        else:
            if self.dim == 2:
                verts = torch.cat([self.verts, torch.zeros_like(self.verts[..., :1])], dim=-1)
            else:
                verts = self.verts
            faces = self.faces
            attrs = self.attrs
        return pytorch3d.structures.Meshes(
            verts=verts, faces=faces,
            textures=pytorch3d.renderer.TexturesVertex(attrs)
        )

    @classmethod
    def unpickle(cls, mesh_file_path: str, pickle_module: Any = default_pickle_module):
        with open(mesh_file_path, 'rb') as f:
            mesh = pickle_module.Unpickler(f).load()
        if isinstance(mesh, RGBMesh):
            return mesh
        else:
            raise BadMeshFormat

    def serialize(self):
        data = super().serialize()
        data.update({"attrs": self.attrs.tolist()})
        return data

    @classmethod
    def _deserialize_tensors(cls, data: Dict) -> Dict:
        new_data = super()._deserialize_tensors(data)
        new_data.update(attrs=torch.tensor(data['attrs']))
        return new_data

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'r') as file:
            data = json.load(file)
        return cls(
            verts=torch.tensor(data['verts']),
            faces=torch.tensor(data['faces']),
            attrs=torch.tensor(data['attrs'])
        )

    @classmethod
    def empty(cls, dim=2, batch_size=1, attr_dim=3):
        verts = torch.zeros((batch_size, 0, dim), dtype=torch.float)
        faces = torch.zeros((batch_size, 0, 3), dtype=torch.int)
        attrs = torch.zeros((batch_size, 0, attr_dim), dtype=torch.float)
        return cls(verts=verts, faces=faces, attrs=attrs)

    def trim(self, polygon: Tensor, trim_face_only: bool = False):
        trimmed_mesh_verts, trimmed_mesh_faces, trimmed_verts_idx = self._trim_and_return_verts_and_faces(
            is_inside_polygon(self.verts, polygon), trim_face_only
        )
        if trimmed_verts_idx is not None:
            trimmed_mesh_attrs = torch.gather(
                self.attrs, 1,
                trimmed_verts_idx.unsqueeze(-1)
                .expand(-1, -1, self.attrs.shape[-1])
            )  # BxVsxAttr_dim
        else:
            trimmed_mesh_attrs = self.attrs
        mesh_with_trims = dataclasses.replace(
            self, verts=trimmed_mesh_verts, faces=trimmed_mesh_faces, attrs=trimmed_mesh_attrs
        )
        return mesh_with_trims


class RGBMesh(AttributeMesh):
    """
    AttributeMesh where the attribute is an RGB color in [0,1] range.
    """
    def _validate_input(self):
        super()._validate_input()
        assert self.attr_dim == 3

    @classmethod
    def set_color(cls, mesh: BaseMesh, color: Color):
        """
        Sets a constant color for all vertices in a given mesh.
        """
        color = tensor_color(color, device=mesh.device, dtype=mesh.verts.dtype)
        return cls.set_attr(mesh=mesh, attr=color)


@dataclass
class BirdviewMesh(BaseMesh):
    """
    2D mesh with vertices and faces assigned to discrete categories, to facilitate rendering simple 2D worlds.
    Category assignment is stored per vertex, and faces should not mix vertices from different categories.
    For each category there is a color and a rendering priority z (lower renders on top), but those
    don't need to be specified until the mesh is converted to a different representation.
    """
    categories: List[str]  #: types of mesh elements included, e.g. road, vehicle, etc.
    colors: Dict[str, Tensor]  #: color of each category, 1d float tensor with 3 elements in [0,1] range RGB
    zs: Dict[str, float]  # rendering priority of each category (lower is on top)
    vert_category: Tensor  #: BxV tensor of indices into categories list
    _cat_fill: int = 0  #: padding for vert_category

    def _validate_input(self):
        assert self.verts.shape[-1] == 2
        super()._validate_input()
        if len(self.vert_category.shape) == 1:
            self.vert_category = self.vert_category.unsqueeze(0)  # add batch dimension
        assert self.vert_category.shape == (self.batch_size, self.verts_count)

    @property
    def num_categories(self) -> int:
        return len(self.categories)

    @classmethod
    def set_properties(cls, mesh: BaseMesh, category: str, color: Optional[Color] = None, z: Optional[float] = None):
        """
        Lifts a BaseMesh into a BirdviewMesh with a single category.
        """
        vert_category = torch.zeros(
            (mesh.batch_size, mesh.verts_count), dtype=mesh.faces.dtype, device=mesh.device
        )
        if color is not None:
            colors = {category: tensor_color(color)}
        else:
            colors = {}
        if z is not None:
            zs = {category: z}
        else:
            zs = {}

        return cls(
            verts=mesh.verts, faces=mesh.faces, categories=[category],
            vert_category=vert_category, colors=colors, zs=zs
        )

    def to(self, device):
        return dataclasses.replace(
            self, verts=self.verts.to(device), faces=self.faces.to(device),
            vert_category=self.vert_category.to(device)
        )

    def expand(self, size):
        f = lambda x: x.unsqueeze(1).expand((self.batch_size, size, *x.shape[1:])).flatten(0, 1)
        return dataclasses.replace(
            self, verts=f(self.verts), faces=f(self.faces),
            vert_category=f(self.vert_category)
        )

    def select_batch_elements(self, idx):
        f = lambda x: x[idx]
        return dataclasses.replace(
            self, verts=f(self.verts), faces=f(self.faces),
            vert_category=f(self.vert_category)
        )

    @classmethod
    def unify(cls, meshes):
        """
        Generates meshes equivalent to input meshes, only all sharing the same category definitions.
        """
        device = meshes[0].device if meshes else 'cpu'
        categories = list(set().union(*[set(m.categories) for m in meshes]))
        category_maps = [
            torch.tensor([categories.index(c) for c in m.categories], dtype=torch.int, device=device)
            for m in meshes
        ]
        colors = merge_dicts([m.colors for m in meshes])
        zs = merge_dicts([m.zs for m in meshes])
        unified = [
            dataclasses.replace(
                m, categories=categories, vert_category=cat_map[m.vert_category.to(torch.int64)],
                colors=colors, zs=zs
            )
            for (m, cat_map) in zip(meshes, category_maps)
        ]
        return unified

    @classmethod
    def concat(cls, meshes):
        meshes = cls.unify(meshes)
        base_concat = BaseMesh.concat(meshes)
        categories = meshes[0].categories if meshes else []
        colors = meshes[0].colors if meshes else {}
        zs = meshes[0].zs if meshes else {}
        vert_category = torch.cat([m.vert_category.to(torch.int64) for m in meshes], dim=-1)
        return cls(
            verts=base_concat.verts, faces=base_concat.faces, categories=categories,
            vert_category=vert_category, colors=colors, zs=zs

        )

    @classmethod
    def collate(cls, meshes):
        meshes = cls.unify(meshes)
        base_collated = BaseMesh.collate(meshes)
        categories = meshes[0].categories if meshes else []
        colors = meshes[0].colors if meshes else {}
        zs = meshes[0].zs if meshes else {}
        vert_category = pad_sequence(
            [m.vert_category.squeeze(0).to(torch.int64) for m in meshes],
            batch_first=True, padding_value=cls._cat_fill
        )
        return cls(
            verts=base_collated.verts, faces=base_collated.faces, categories=categories,
            vert_category=vert_category, colors=colors, zs=zs
        )

    def fill_attr(self) -> RGBMesh:
        """
        Computes explicit color for each vertex and augments vertices with z values corresponding to rendering priority.
        """
        missing_colors = [c for c in self.categories if c not in self.colors]
        if missing_colors:
            raise RuntimeError(f'Missing color values for the following categories: {missing_colors}')
        missing_zs = [c for c in self.categories if c not in self.zs]
        if missing_zs:
            raise RuntimeError(f'Missing z values for the following categories: {missing_zs}')
        zs = torch.tensor([self.zs[k] for k in self.categories], dtype=self.verts.dtype, device=self.device)
        zs = zs[self.vert_category.to(torch.int64)].unsqueeze(-1)
        color_list = [self.colors[k] for k in self.categories]
        if color_list:
            colors = torch.stack(color_list).to(self.verts.dtype).to(self.device)
            colors = colors[self.vert_category.to(torch.int64)]
        else:
            colors = torch.zeros((self.batch_size, 0, 3), dtype=self.verts.dtype, device=self.device)
        verts = torch.cat([self.verts[..., :2], zs], dim=-1)
        attrs = colors
        return RGBMesh(verts=verts, faces=self.faces, attrs=attrs)

    def pytorch3d(self, include_textures=True) -> "pytorch3d.structures.Meshes":
        if include_textures:
            return self.fill_attr().pytorch3d(include_textures=True)
        else:
            return super().pytorch3d(include_textures=False)

    @classmethod
    def unpickle(cls, mesh_file_path: str, pickle_module: Any = default_pickle_module):
        with open(mesh_file_path, 'rb') as f:
            lane_mesh = pickle_module.Unpickler(f).load()
        if isinstance(lane_mesh, BirdviewMesh):
            return lane_mesh
        else:
            raise BadMeshFormat

    def serialize(self):
        data = super().serialize()
        data.update({
            'categories': self.categories,
            'colors': {k: v.tolist() for k, v in self.colors.items()},
            'zs': self.zs,
            'vert_category': self.vert_category.tolist(),
            '_cat_fill': self._cat_fill
        })
        return data

    @classmethod
    def _deserialize_tensors(cls, data: Dict) -> Dict:
        new_data = super()._deserialize_tensors(data)
        new_data.update(categories=data['categories'],
                        colors={k: torch.tensor(v) for k, v in data['colors'].items()},
                        zs=data['zs'],
                        vert_category=torch.tensor(data['vert_category']),
                        _cat_fill=data['_cat_fill'])
        return new_data

    @classmethod
    def empty(cls, dim=2, batch_size=1):
        verts = torch.zeros((batch_size, 0, dim), dtype=torch.float)
        faces = torch.zeros((batch_size, 0, 3), dtype=torch.int)
        vert_category = torch.zeros([batch_size, 0], dtype=torch.int)
        categories = []
        colors = dict()
        zs = dict()

        return cls(
            verts=verts, faces=faces, vert_category=vert_category,
            categories=categories, colors=colors, zs=zs
        )

    def trim(self, polygon: Tensor, trim_face_only=False):
        trimmed_mesh_verts, trimmed_mesh_faces, trimmed_verts_idx = self._trim_and_return_verts_and_faces(
            is_inside_polygon(self.verts, polygon), trim_face_only
        )
        if trimmed_verts_idx is not None:
            trimmed_vert_category = torch.gather(self.vert_category, 1, trimmed_verts_idx)  # BxVs
        else:
            trimmed_vert_category = self.vert_category
        mesh_with_trims = dataclasses.replace(self, verts=trimmed_mesh_verts,
                                              faces=trimmed_mesh_faces,
                                              vert_category=trimmed_vert_category)
        return mesh_with_trims

    def separate_by_category(self) -> Dict[str, BaseMesh]:
        """
        Splits the mesh into meshes representing different categories.
        """
        meshes = dict()
        for (i, category) in enumerate(self.categories):
            verts, faces, _ = self._trim_and_return_verts_and_faces(
                self.vert_category == i, trim_face_only=False
            )
            meshes[category] = BaseMesh(verts=verts, faces=faces)
        return meshes


class BirdviewRGBMeshGenerator:
    """
    A generator for constructing simulation birdview representations. All meshes (backround mesh, agent mesh and
    traffic control mesh) are constructed as templates once at the creation of the generator. Calling the generate
    function will transform each mesh to a new position/state and return a single RGB mesh per camera for rendering.
    """
    def __init__(self, background_mesh: BirdviewMesh, color_map: Dict[str, Tuple[int, int, int]],
                 rendering_levels: Dict[str, float], world_center: Optional[Tensor] = None,
                 agent_attributes: Optional[Tensor] = None, agent_types: Optional[Tensor] = None,
                 agent_type_names: Optional[List[str]] = None, render_agent_direction: bool = True,
                 traffic_controls: Optional[Dict[str, BaseTrafficControl]] = None,
                 waypoint_radius: float = 2.0, waypoint_num_triangles: int = 10):

        self.color_map = color_map
        self.rendering_levels = rendering_levels

        self.initialize_background_mesh(background_mesh, world_center)
        self.initialize_waypoint_mesh(waypoint_radius, waypoint_num_triangles)

        self.actor_mesh = None
        if agent_attributes is not None:
            assert agent_types is not None
            assert agent_type_names is not None
            self.initialize_actors_mesh(agent_attributes, agent_types, agent_type_names, render_agent_direction)

        self.static_traffic_controls_mesh = None
        self.traffic_lights_mesh = None
        self.traffic_light_colors = None
        if traffic_controls is not None:
            self.initialize_traffic_controls_mesh(traffic_controls)

    def to(self, device: torch.device):
        """
        Moves the renderer to another device in place.
        """
        self.background_mesh = self.background_mesh.to(device)
        self.world_center = self.world_center.to(device)
        self.waypoint_mesh = self.waypoint_mesh.to(device)
        if self.actor_mesh is not None:
            self.actor_mesh = self.actor_mesh.to(device)
        if self.static_traffic_controls_mesh is not None:
            self.static_traffic_controls_mesh = self.static_traffic_controls_mesh.to(device)
        if self.traffic_lights_mesh is not None:
            self.traffic_lights_mesh = self.traffic_lights_mesh.to(device)
        if self.traffic_light_colors is not None:
            self.traffic_light_colors = self.traffic_light_colors.to(device)
        return self

    def copy(self):
        return self.expand(1)

    def expand(self, n: int):
        """
        Adds another dimension with size n on the right of the batch dimension and flattens them.
        Returns a new renderer, without modifying the current one.
        """
        expand = lambda x: x.unsqueeze(1).expand((x.shape[0], n) + x.shape[1:]).reshape((n * x.shape[0],) + x.shape[1:])\
            if x is not None else None
        other = self.__class__(background_mesh=BirdviewMesh.empty(self.background_mesh.batch_size*n),
                               color_map=self.color_map.copy(), rendering_levels=self.rendering_levels.copy())
        other.background_mesh = self.background_mesh.expand(n)
        other.waypoint_mesh = self.waypoint_mesh.expand(n)
        other.world_center = expand(self.world_center)
        if self.actor_mesh is not None:
            other.actor_mesh = self.actor_mesh.expand(n)
        if self.static_traffic_controls_mesh is not None:
            other.static_traffic_controls_mesh = self.static_traffic_controls_mesh.expand(n)
        if self.traffic_lights_mesh is not None:
            other.traffic_lights_mesh = self.traffic_lights_mesh.expand(n)
        if self.traffic_light_colors is not None:
            other.traffic_light_colors = expand(self.traffic_light_colors)
        other.waypoint_radius = self.waypoint_radius
        other.waypoint_num_triangles = self.waypoint_num_triangles
        if hasattr(self, 'render_agent_direction'):
            other.render_agent_direction = self.render_agent_direction
        return other

    def select_batch_elements(self, idx: Tensor):
        """
        Selects given elements from the batch, potentially with repetitions.
        Returns a new renderer, without modifying the current one.

        Args:
            idx: one-dimensional integer tensor
        """
        other = self.copy()
        other.background_mesh = other.background_mesh[idx]
        other.world_center = other.world_center[idx]
        other.waypoint_mesh = other.waypoint_mesh[idx]
        if other.actor_mesh is not None:
            other.actor_mesh = other.actor_mesh[idx]
        if other.static_traffic_controls_mesh is not None:
            other.static_traffic_controls_mesh = other.static_traffic_controls_mesh[idx]
        if other.traffic_lights_mesh is not None:
            other.traffic_lights_mesh = other.traffic_lights_mesh[idx]
        if other.traffic_light_colors is not None:
            other.traffic_light_colors = other.traffic_light_colors[idx]
        return other

    def initialize_background_mesh(self, background_mesh: BirdviewMesh, world_center: Optional[Tensor] = None):
        if world_center is None:
            if hasattr(background_mesh, 'categories') and 'road' in background_mesh.categories:
                world_center = background_mesh.separate_by_category()['road'].center
            else:
                world_center = background_mesh.center
        self.world_center = world_center.to(background_mesh.device)
        self.background_mesh = set_colors_with_defaults(background_mesh.clone(), color_map=self.color_map,
                                                        rendering_levels=self.rendering_levels)

    def add_static_meshes(self, meshes: List[BirdviewMesh]) -> None:
        """
        Includes additional static elements to background mesh.
        """
        self.add_static_rgb_meshes([set_colors_with_defaults(m.clone(), color_map=self.color_map,
                                                             rendering_levels=self.rendering_levels) for m in meshes])

    def add_static_rgb_meshes(self, meshes: List[RGBMesh]) -> None:
        """
        Includes additional static rgb elements to background mesh.
        """
        self.background_mesh = self.background_mesh.concat(
            [self.background_mesh] + meshes
        )

    def initialize_waypoint_mesh(self, waypoint_radius: float = 2.0, waypoint_num_triangles: int = 10):
        self.waypoint_radius = waypoint_radius
        self.waypoint_num_triangles = waypoint_num_triangles
        waypoint_mesh = self._make_waypoint_mesh(self.background_mesh.batch_size, radius=self.waypoint_radius,
                                                 num_triangles=self.waypoint_num_triangles,
                                                 device=self.background_mesh.device)
        self.waypoint_mesh = set_colors_with_defaults(waypoint_mesh, color_map=self.color_map,
                                                      rendering_levels=self.rendering_levels)

    @classmethod
    def _make_waypoint_mesh(cls, batch_size: int, radius: float = 2.0, num_triangles: int = 10,
                            device: torch.device = torch.device('cpu')) -> BirdviewMesh:
        """
        Create a mesh of the given waypoints.

        Args:
            batch_size: int number of batch waypoint meshes to create
            radius: float radius of the disc
            num_triangles: int number of triangles used for the disc
            device: torch.device to create the mesh
        """
        disc_verts, disc_faces = generate_disc_mesh(device=device, radius=radius, num_triangles=num_triangles)
        disc_verts = disc_verts[None, ...].expand(batch_size, *disc_verts.shape).clone()
        disc_faces = disc_faces[None, ...].expand(batch_size, *disc_faces.shape).clone()
        return rendering_mesh(BaseMesh(verts=disc_verts, faces=disc_faces), 'goal_waypoint')

    @classmethod
    def _make_direction_mesh(cls, lenwid: Tensor, size: float = 0.3, device: torch.device = torch.device('cpu')) -> BaseMesh:
        """
        Create a mesh indicating the direction of each agent.

        Args:
            lenwid: BxAx2 tensor specifying length and width of the agents
            size: determines the size of the triangle indicating direction
        """
        batch_size = lenwid.shape[0]
        n_actors = lenwid.shape[-2]
        corners = torch.stack([
            F.pad( lenwid[..., 0:1] * size, (1, 0), value=0.0),
            F.pad( lenwid[..., 1:2] * 0.5,  (0, 1), value=0.0),
            F.pad(-lenwid[..., 1:2] * 0.5,  (0, 1), value=0.0),
        ], dim=-2).flip([-1])
        offset = torch.cat([
            lenwid[..., 0:1]*(0.5 - size),
            torch.zeros_like(lenwid[..., 1:2])
        ], dim=-1).unsqueeze(-2)
        corners = corners + offset
        verts = corners.reshape(batch_size, n_actors * 3, 2)
        faces = torch.tensor(
            [[[0,  1,  2]]], dtype=torch.long, device=device
        ).expand(batch_size, n_actors, 3)
        faces_offset = 3 * torch.arange(
            start=0, end=n_actors, dtype=torch.long, device=device
        ).reshape(1, n_actors, 1).expand_as(faces)
        faces = faces + faces_offset
        return BaseMesh(verts=verts, faces=faces)

    @classmethod
    def _make_actors_mesh(cls, agent_attributes: Tensor, agent_types: Tensor, agent_type_names: List[str],
                          render_agent_direction: bool = True, device: torch.device = torch.device('cpu')) -> BirdviewMesh:
        lenwid = agent_attributes
        n_actors = lenwid.shape[-2]
        length, width = lenwid[..., 0], lenwid[..., 1]
        corners = torch.stack([
            torch.stack([x, y], dim=-1) for (x, y) in
            [(length, width), (length, - width), (- length, - width), (- length, width)]
        ], dim=-2) * 0.5
        batch_size = lenwid.size()[0]
        actor_verts = corners.reshape(batch_size, n_actors * 4, 2)

        actor_faces = torch.tensor([[0, 1, 3], [1, 3, 2]], dtype=torch.long, device=device)
        actor_faces = actor_faces.expand(batch_size, n_actors, 2, 3)
        offsets = 4 * torch.arange(start=0, end=n_actors, dtype=torch.long,
                                device=device).reshape(n_actors, 1, 1).expand_as(actor_faces)
        actor_faces = actor_faces + offsets
        actor_faces = actor_faces.reshape(batch_size, n_actors * 2, 3)

        if render_agent_direction:
            direction_mesh = cls._make_direction_mesh(lenwid=lenwid, device=device)
            # custom concatenation of tensors, so that both vertices and faces belonging
            # to each agent (both bbox and direction) are contiguous
            # this allows for subsequent masking of agents
            av = 4  # verts per actor
            dv = 3  # verts per direction
            verts = torch.cat([
                actor_verts.reshape(batch_size, n_actors, av, 2),
                direction_mesh.verts.reshape(batch_size, n_actors, dv, 2)
            ], dim=-2).reshape(batch_size, n_actors * (av + dv), 2)
            actor_faces = actor_faces + actor_faces.div(av, rounding_mode='trunc') * dv
            direction_faces = direction_mesh.faces + av * (direction_mesh.faces.div(dv, rounding_mode='trunc') + 1)
            faces = torch.cat([
                actor_faces.reshape(batch_size, n_actors, 2, 3),
                direction_faces.reshape(batch_size, n_actors, 1, 3)
            ], dim=-2).reshape(batch_size, n_actors * 3, 3)
            categories = agent_type_names + ['direction']
            vert_category=torch.cat([
                agent_types.unsqueeze(-1).expand(agent_types.shape + (av,)),
                torch.ones((batch_size, n_actors, dv), dtype=torch.int64, device=verts.device) * len(agent_type_names)
            ], dim=-1).reshape(batch_size, n_actors * (av + dv))
        else:
            av = 4
            dv = 0
            verts = actor_verts
            faces = actor_faces
            categories = agent_type_names
            vert_category = agent_types.unsqueeze(-1).expand(agent_types.shape + (av,)).reshape(batch_size, n_actors * (av + dv))
        actor_mesh = BirdviewMesh(
            verts=verts, faces=faces, categories=categories,
            vert_category=vert_category,
            colors=dict(), zs=dict(),
            )
        return actor_mesh

    def initialize_actors_mesh(self, agent_attributes: Tensor, agent_types: Tensor, agent_type_names: List[str],
                               render_agent_direction: bool = True):
        self.render_agent_direction = render_agent_direction
        actor_mesh = self._make_actors_mesh(agent_attributes, agent_types, agent_type_names,
                                            render_agent_direction, self.background_mesh.device)
        self.actor_mesh = set_colors_with_defaults(actor_mesh, color_map=self.color_map,
                                                   rendering_levels=self.rendering_levels)

    @classmethod
    def _create_traffic_controls_mesh(cls, traffic_controls: Dict[str, BaseTrafficControl],
                                      selected_traffic_control_types: Optional[List[str]] = None) -> BirdviewMesh:
        """
        Create a mesh showing traffic controls.
        """
        if traffic_controls:
            batch_size = max(element.corners.shape[0] for element in traffic_controls.values())
        else:
            batch_size = 1
        meshes = []
        for control_type, element in traffic_controls.items():
            if selected_traffic_control_types is not None and control_type not in selected_traffic_control_types:
                continue
            if element.corners.shape[-2] == 0:
                continue
            verts, faces = build_verts_faces_from_bounding_box(element.corners)
            if control_type == 'traffic_light':
                categories = [f'{control_type}_{state}' for state in element.allowed_states]
                vert_category = element.state.unsqueeze(-1).expand(element.state.shape + (4,)).flatten(-2, -1)
                meshes.append(BirdviewMesh(
                    verts=verts, faces=faces, categories=categories, vert_category=vert_category,
                    zs=dict(), colors=dict()
                ))
            else:
                meshes.append(rendering_mesh(
                    BaseMesh(verts=verts, faces=faces), category=control_type # TODO: add light state
                ))
        if meshes:
            return BirdviewMesh.concat(meshes)
        else:
            return BirdviewMesh.empty(dim=2, batch_size=batch_size)

    def initialize_traffic_controls_mesh(self, traffic_controls: Dict[str, BaseTrafficControl]):
        static_traffic_controls_mesh = self._create_traffic_controls_mesh(traffic_controls, ['stop_sign', 'yield_sign'])
        self.static_traffic_controls_mesh = set_colors_with_defaults(static_traffic_controls_mesh, color_map=self.color_map,
                                                                     rendering_levels=self.rendering_levels)
        traffic_lights_mesh = self._create_traffic_controls_mesh(traffic_controls, ['traffic_light'])
        self.traffic_lights_mesh = set_colors_with_defaults(traffic_lights_mesh, color_map=self.color_map,
                                                            rendering_levels=self.rendering_levels)
        if 'traffic_light' in traffic_controls:
            self.traffic_light_colors = torch.stack([
                tensor_color(self.color_map[f'traffic_light_{tls}'], device=self.traffic_lights_mesh.device)
                    for tls in traffic_controls['traffic_light'].allowed_states
            ], dim=0).reshape(1, 1, -1, 3).expand(self.traffic_lights_mesh.batch_size,
                                                  traffic_controls['traffic_light'].state.shape[1], -1, -1)

    def generate(self, num_cameras: int, agent_state: Optional[Tensor] = None,
                 present_mask: Optional[Tensor] = None,
                 traffic_lights: Optional[TrafficLightControl] = None,
                 waypoints: Optional[Tensor] = None, waypoints_rendering_mask: Optional[Tensor] = None,
                 custom_agent_colors: Optional[Tensor] = None) -> RGBMesh:
        """
        Create an RGB mesh updates given the provided states.

        Args:
            num_cameras: int the number of cameras
            agent_state: BxNcxAx4 tensor specifying
            present_mask: BxNcxA tensor specifying
            traffic_lights: TrafficLightControl object extended for each camera
            waypoints: BxNcxMx2 tensor of `M` waypoints per camera (x,y)
            waypoints_rendering_mask: BxNcxM tensor of `M` waypoint masks per camera,
                indicating which waypoints should be rendered
            custom_agent_colors: a BxNcxAx3 tensor of specifying what color each agent is to what camera
        """
        actor_mesh = None
        if agent_state is not None and self.actor_mesh is not None:
            assert agent_state.shape[1] == num_cameras
            actor_mesh = self.actor_mesh.expand(num_cameras)
            agent_state = agent_state.flatten(0, 1)
            effective_batch_size, n_actors, _ = agent_state.shape
            agent_verts = transform(actor_mesh.verts[..., :2].reshape(effective_batch_size*n_actors, -1, 2),
                                    agent_state[..., :3].reshape(effective_batch_size*n_actors, 3))
            agent_verts = F.pad(agent_verts.reshape(effective_batch_size, -1, 2), (0,1), value=0.0)
            agent_verts[..., 2:3] = actor_mesh.verts[..., 2:3]

            actor_faces = actor_mesh.faces
            if present_mask is not None:
                assert present_mask.shape[1] == num_cameras
                present_mask = present_mask.flatten(0, 1)
                faces_per_agent = actor_faces.shape[1] // present_mask.shape[1]
                faces_mask = present_mask[..., None].broadcast_to(present_mask.shape + \
                             (faces_per_agent,)).flatten(1, 2)[..., None]
                actor_faces = actor_faces * faces_mask

            actor_attrs = actor_mesh.attrs
            if custom_agent_colors is not None:
                actor_attrs = actor_attrs.clone()
                av = 4
                dv = 3 if self.render_agent_direction else 0
                step = av + dv
                custom_agent_colors = custom_agent_colors.flatten(0, 1)
                for i in range(av):
                    actor_attrs[:,i::step] = custom_agent_colors

            actor_mesh = dataclasses.replace(
                actor_mesh, verts=agent_verts, faces=actor_faces, attrs=actor_attrs
            )

        traffic_lights_mesh = None
        if traffic_lights is not None and \
           (self.traffic_lights_mesh is not None and self.traffic_lights_mesh.faces_count > 0):
            assert traffic_lights.state.shape[0] == self.traffic_lights_mesh.batch_size*num_cameras
            traffic_lights_mesh = self.traffic_lights_mesh.expand(num_cameras)
            # set traffic light state
            traffic_light_colors = self.traffic_light_colors[:, None].repeat_interleave(num_cameras, dim=1).flatten(0, 1)
            current_colors = torch.gather(traffic_light_colors, dim=2,
                index=traffic_lights.state[..., None, None].expand(-1, -1, -1, 3))
            # A traffic light contains 4 vertices
            current_colors = current_colors.expand(-1, -1, 4, -1).reshape(traffic_lights_mesh.batch_size, -1, 3)
            traffic_lights_mesh = dataclasses.replace(
                traffic_lights_mesh, attrs=current_colors
            )

        waypoints_mesh = None
        if waypoints is not None:
            waypoints_mesh = self.waypoint_mesh.clone()
            assert waypoints.shape[1] == num_cameras
            b_size, n_waypoints = waypoints_mesh.batch_size, waypoints.shape[2]
            expanded_waypoints_mesh = waypoints_mesh.expand(num_cameras*n_waypoints)
            waypoints_mesh = waypoints_mesh.expand(num_cameras)
            waypoints_verts = expanded_waypoints_mesh.verts[..., :2]
            n_verts = waypoints_verts.shape[-2]
            waypoints_verts = transform(waypoints_verts, F.pad(waypoints, (0,1), value=0).reshape(-1, 3))
            waypoints_verts = waypoints_verts.reshape(b_size*num_cameras, n_waypoints*waypoints_verts.shape[1], 2)
            waypoints_verts = F.pad(waypoints_verts, (0,1), value=0.0)
            waypoints_verts[..., 2:3] = expanded_waypoints_mesh.verts[..., 2:3].reshape(b_size*num_cameras, -1, 1)

            waypoints_faces = expanded_waypoints_mesh.faces.reshape(b_size*num_cameras, n_waypoints, -1, 3)
            waypoints_faces = waypoints_faces + n_verts*torch.arange(n_waypoints,
                                                                     device=waypoints_faces.device)[None, :, None, None]
            waypoints_faces = waypoints_faces.flatten(1, 2)

            waypoints_attrs = expanded_waypoints_mesh.attrs.reshape(b_size*num_cameras, -1, 3)
            if waypoints_rendering_mask is not None:
                waypoints_mask = waypoints_rendering_mask.reshape(-1, n_waypoints, 1, 1)\
                                 .expand(-1, -1, self.waypoint_num_triangles, 3)
                waypoints_faces = waypoints_faces * waypoints_mask.reshape(-1, n_waypoints*self.waypoint_num_triangles, 3)
            waypoints_mesh = dataclasses.replace(
                waypoints_mesh, verts=waypoints_verts, faces=waypoints_faces, attrs=waypoints_attrs
            )

        meshes = [self.background_mesh.expand(num_cameras)]
        if actor_mesh is not None:
            meshes.append(actor_mesh)
        if self.static_traffic_controls_mesh is not None:
            meshes.append(self.static_traffic_controls_mesh.expand(num_cameras))
        if traffic_lights_mesh is not None:
            meshes.append(traffic_lights_mesh)
        if waypoints_mesh is not None:
            meshes.append(waypoints_mesh)
        rgb_mesh = RGBMesh.concat(meshes)
        return rgb_mesh


def rendering_mesh(mesh: BaseMesh, category: str) -> BirdviewMesh:
    """
    Assigns a category to a given mesh.
    """
    return BirdviewMesh.set_properties(
        BaseMesh(verts=mesh.verts, faces=mesh.faces),
        category=category
    )


def set_colors_with_defaults(mesh: BirdviewMesh, color_map: Dict[str, Tensor], rendering_levels: Dict[str, float]) -> RGBMesh:
    for k in mesh.categories:
        if k not in mesh.colors:
            mesh.colors[k] = tensor_color(color_map[k])
        if k not in mesh.zs:
            mesh.zs[k] = rendering_levels[k]
    # if self.cfg.highlight_ego_vehicle:
    #     mesh.colors["ego"] = tensor_color((color_map["ego"]))
    return mesh.fill_attr()


def generate_trajectory_mesh(points: Tensor, category: Optional[str] = None, edge_length: float = 1) -> BaseMesh:
    """
    Create a triangle mesh used to visualize a given trajectory.
    Each point is converted to a triangle matching its position and orientation.

    Args:
        points: Bx3 tensor of x-y coordinates and orientations in radians
        category: if specified, produces BirdviewMesh
        edge_length: specifies the size of the resulting triangle
    """
    verts = torch.stack([
        torch.stack([points[..., 0] + edge_length * 0.5 * torch.cos(points[..., 2]),
                     points[..., 1] + edge_length * 0.5 * torch.sin(points[..., 2])], dim=-1),
        torch.stack([points[..., 0] + edge_length * 0.5 * torch.cos(points[..., 2] + 2 * math.pi / 3),
                     points[..., 1] + edge_length * 0.5 * torch.sin(points[..., 2] + 2 * math.pi / 3)], dim=-1),
        torch.stack([points[..., 0] + edge_length * 0.5 * torch.cos(points[..., 2] + 4 * math.pi / 3),
                     points[..., 1] + edge_length * 0.5 * torch.sin(points[..., 2] + 4 * math.pi / 3)], dim=-1)
    ], dim=-2)
    verts = torch.flatten(verts, start_dim=-4, end_dim=-2)
    faces = (torch.arange(start=0, end=verts.shape[-2])).reshape(
        (1, int(verts.shape[-2] / 3), 3)
    ).expand(verts.shape[0], -1, -1).int().to(verts.device)
    mesh = BaseMesh(verts=verts, faces=faces)
    if category is not None:
        mesh = rendering_mesh(mesh, category=category)
    return mesh


def generate_annulus_polygon_mesh(polygon: Tensor, scaling_factor: float, origin: Tensor,
                                  category: Optional[str] = None) -> BaseMesh:
    """
    For a given polygon, generates a mesh covering the space between the polygon
    and its scaled version.

    Args:
        polygon: tensor of size Nx2 defining subsequent points of the polygon hull
        scaling_factor: determines the side of the annulus, should be larger than 1
        origin: tensor of size (2,) defining the point around which scaling is performed
        category: if specified, BirdviewMesh will be returned
    """
    center_2d = origin[0:2][None, ...]
    outer_points = (polygon - center_2d) * scaling_factor + center_2d
    polygon = torch.stack([polygon, outer_points], dim=1).flatten(start_dim=1, end_dim=1).reshape(-1, 2).squeeze()

    device = polygon.device
    verts = polygon.reshape(-1, 2)
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device)
    num_verts = int(verts.shape[0])
    num_faces = num_verts
    faces = faces[None, ...].expand(num_faces - 2, -1, -1).reshape(-1, 3).clone()
    offsets = torch.arange(0, num_faces - 2, device=device).repeat_interleave(3, dim=-1).reshape(-1, 3)
    faces += offsets
    faces = torch.cat([faces,
                       torch.tensor([[num_verts - 1, 0, 1]], dtype=torch.int32, device=device),
                       torch.tensor([[num_verts - 2, num_verts - 1, 0]], dtype=torch.int32, device=device)],
                      dim=0)
    mesh = BaseMesh(verts=verts.unsqueeze(0), faces=faces.unsqueeze(0))
    if category is not None:
        mesh = rendering_mesh(mesh, category=category)
    return mesh


def generate_disc_mesh(radius: float = 2, num_triangles: int = 10, device: str = 'cpu') -> Tuple[Tensor, Tensor]:
    """
    For a given radius, it will create a disc mesh using `num_triangles` triangles.

    Args:
        radius: float defining the radius of the disc
        num_triangles: int defining the number of triangles to be used for creating the disc
        device: the device to be used for the generated PyTorch tensors
    """
    angleStep = torch.deg2rad(torch.tensor([[360 / num_triangles]], dtype=torch.float32, device=device))

    vertices = [
        torch.zeros(1, 2, dtype=torch.float32, device=device),
        torch.tensor([[radius, 0]], dtype=torch.float32, device=device),
        rotate(torch.tensor([[radius, 0]], dtype=torch.float32, device=device), angleStep)
    ]
    faces = [
        torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)
    ]

    for i in range(num_triangles - 1):
        if i == num_triangles - 2:
            faces.append(torch.tensor([[0, len(vertices)-1, 1]], dtype=torch.long, device=device))
        else:
            faces.append(torch.tensor([[0, len(vertices)-1, len(vertices)]], dtype=torch.long, device=device))
            vertices.append(rotate(vertices[-1], angleStep))
    vertices = torch.cat(vertices, dim=0)
    faces = torch.cat(faces, dim=0)
    return vertices, faces


def build_verts_faces_from_bounding_box(bbs: Tensor, z: float = 2) -> Tuple[Tensor, Tensor]:
    """
    Triangulates bounding boxes for rendering. Input is a tensor of bounding boxes of shape ...xAx4x2,
    where A is the number of actors. Outputs are shaped ...x4*Ax2 and ...x2*Ax3 respectively.
    """
    batch_dims = bbs.size()[:-3]
    n_actors = bbs.size()[-3]
    verts = bbs.reshape(*batch_dims, -1, 2)

    faces = torch.tensor([[0, 1, 3], [1, 3, 2]], dtype=torch.long, device=bbs.device)
    faces = faces.unsqueeze(0).expand(*batch_dims, n_actors, 2, 3)
    offsets = 4 * torch.arange(start=0, end=n_actors, dtype=torch.long,
                               device=bbs.device).reshape(n_actors, 1, 1).expand_as(faces)
    faces = faces + offsets
    faces = faces.reshape(*batch_dims, 2 * n_actors, 3)

    return verts, faces
