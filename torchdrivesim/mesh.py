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

from torchdrivesim import assert_pytorch3d_available
from torchdrivesim.utils import is_inside_polygon, merge_dicts, rotate

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


def rendering_mesh(mesh: BaseMesh, category: str) -> BirdviewMesh:
    """
    Assigns a category to a given mesh.
    """
    return BirdviewMesh.set_properties(
        BaseMesh(verts=mesh.verts, faces=mesh.faces),
        category=category
    )


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
