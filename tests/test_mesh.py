import dataclasses
import os
import shutil
import numpy as np
import torch
from PIL import Image
import pytest

from torchdrive.rendering import BirdviewRenderer, renderer_from_config, RendererConfig
from torchdrive.mesh import BaseMesh, AttributeMesh, BirdviewMesh
from torchdrive.utils import Resolution
from torchdrive.simulator import TorchDriveConfig


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TestBaseMesh:
    save_dir = None

    @classmethod
    def setup_class(cls):
        cls.mesh = BaseMesh(verts=torch.zeros(1, 3, 2), faces=torch.zeros(1, 1, 3))
        cls.save_dir = os.path.join(os.path.dirname(__file__), 'tmp')

    @classmethod
    def setup_method(cls) -> None:
        pass

    @classmethod
    def teardown_class(cls):
        try:
            shutil.rmtree(cls.save_dir)
        except OSError as e:
            print("Error: %s : %s" % (cls.save_dir, e.strerror))

    def test_expand(self):
        expanded_mesh = self.mesh.expand(2)
        assert expanded_mesh.verts.shape[0] == 2 and expanded_mesh.faces.shape[0] == 2
        return expanded_mesh

    def test_select_batch_elements(self):
        selected_mesh = self.mesh.select_batch_elements(torch.tensor([0], dtype=torch.long))
        assert len(selected_mesh.verts.shape) == 3 and len(selected_mesh.faces.shape) == 3
        return selected_mesh

    def test_collate(self):
        mesh1 = self.mesh.clone()
        mesh2 = self.mesh.clone()
        collated_mesh = self.mesh.collate([mesh1, mesh2])
        assert collated_mesh.verts.shape[0] == 2 and collated_mesh.faces.shape[1] == 1
        return collated_mesh

    def test_concat(self):
        mesh1 = self.mesh.clone()
        mesh2 = self.mesh.clone()
        concatenated_mesh = self.mesh.concat([mesh1, mesh2])
        assert concatenated_mesh.verts.shape[0] == 1 and concatenated_mesh.verts.shape[1] == 6
        return concatenated_mesh

    def test_pickle_and_unpickle(self):
        save_path = os.path.join(self.save_dir, 'test_mesh.pickle')
        _ = self.mesh.pickle(save_path)
        _ = self.mesh.unpickle(save_path)

    def test_pytorch3d(self):
        self.mesh.pytorch3d()

    def test_trim(self):
        self.mesh.trim(torch.zeros(1, 1, 2))
        self.mesh.trim(torch.zeros(1, 1, 2), True)


class TestAttributeMesh(TestBaseMesh):
    mesh = None

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.mesh = AttributeMesh.set_attr(cls.mesh, torch.zeros(1))

    def test_expand(self):
        expanded_mesh = super().test_expand()
        assert expanded_mesh.attrs.shape[0] == 2
        return expanded_mesh

    def test_select_batch_elements(self):
        selected_mesh = super().test_select_batch_elements()
        assert len(selected_mesh.attrs.shape) == 3
        return selected_mesh

    def test_collate(self):
        collated_mesh = super().test_collate()
        assert collated_mesh.attrs.shape[0] == 2
        return collated_mesh

    def test_concat(self):
        concatenated_mesh = super().test_concat()
        assert concatenated_mesh.attrs.shape[0] == 1 and concatenated_mesh.verts.shape[1] == 6
        return concatenated_mesh

    def test_pickle_and_unpickle(self):
        pass


class TestBirdviewMesh(TestBaseMesh):
    mesh = None

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.mesh = BirdviewMesh.set_properties(BaseMesh(verts=torch.zeros(1, 3, 2), faces=torch.zeros(1, 1, 3)), "road", (0, 0, 0), 0)

    def test_expand(self):
        expanded_mesh = super().test_expand()
        assert expanded_mesh.vert_category.shape[0] == 2
        return expanded_mesh

    def test_select_batch_elements(self):
        selected_mesh = super().test_select_batch_elements()
        assert len(selected_mesh.vert_category.shape) == 2
        return selected_mesh

    def test_collate(self):
        collated_mesh = super().test_collate()
        assert collated_mesh.vert_category.shape[0] == 2
        return collated_mesh

    def test_concat(self):
        concatenated_mesh = super().test_concat()
        assert concatenated_mesh.vert_category.shape[0] == 1 and concatenated_mesh.vert_category.shape[1] == 6
        return concatenated_mesh


    def test_pickle_and_unpickle(self):
        pass

    def test_fill_attrs(self):
        self.mesh.fill_attr()

    def test_separate_categories(self):
        road_mesh = self.mesh
        lane_mesh = self.mesh.separate_by_category()['road']
        lane_mesh = dataclasses.replace(lane_mesh, verts=lane_mesh.verts+1)
        lane_mesh = BirdviewMesh.set_properties(lane_mesh, category='lane')
        joint_mesh = BirdviewMesh.concat([road_mesh, lane_mesh])
        split_meshes = joint_mesh.separate_by_category()
        recovered_road_mesh = split_meshes['road']
        recovered_lane_mesh = split_meshes['lane']
        assert (recovered_road_mesh.verts == road_mesh.verts).all().item()
        assert (recovered_road_mesh.faces == road_mesh.faces).all().item()
        assert (recovered_lane_mesh.verts == lane_mesh.verts).all().item()
        assert (recovered_lane_mesh.faces == lane_mesh.faces).all().item()
