import unittest
import pytest
import numpy as np
import torch
try:
    import lanelet2
except ImportError:
    pass
from torchdrivesim.infractions import lanelet_orientation_loss
from torchdrivesim.lanelet2 import find_direction


@pytest.mark.depends_on_lanelet2
class TestModelsUtil(unittest.TestCase):
    env = None

    def test_get_direction_on_linestring(self):
        reference = lanelet2.core.BasicPoint3d(0.5, 0.5, 0)
        # Point3d(id, x, yy, z)
        linestring = lanelet2.core.ConstLineString3d(0, [lanelet2.core.Point3d(0, 0, 0, 10),
                                                         lanelet2.core.Point3d(1, 1, 1, 10),
                                                         lanelet2.core.Point3d(2, 2, 1, 10)])
        self.assertTrue(find_direction(linestring, reference) == np.pi / 4)

    def test_get_lanelet_orientation_loss(self):
        left_bound = lanelet2.core.LineString3d(0, [lanelet2.core.Point3d(0, 0, 0, 10),
                                                    lanelet2.core.Point3d(1, 1, 1, 10),
                                                    lanelet2.core.Point3d(2, 2, 1, 10)])
        right_bound = lanelet2.core.LineString3d(1, [lanelet2.core.Point3d(3, 0.05, 0, 10),
                                                     lanelet2.core.Point3d(4, 1, 0.95, 10),
                                                     lanelet2.core.Point3d(5, 2, 0.95, 10)])
        test_map = lanelet2.core.LaneletMap()
        test_lanelet = lanelet2.core.Lanelet(lanelet2.core.getId(), left_bound, right_bound)
        test_map.add(test_lanelet)
        test_agent_state = torch.Tensor([[[0.5, 0.5, np.pi / 4, 1], [0.5, 0.5, 5 * np.pi / 4, 1]],
                                         [[0.5, 0.5, np.pi / 4, 1], [0.5, 0.5, 5 * np.pi / 4, 1]]])
        self.assertTrue(
            torch.all(torch.isclose(lanelet_orientation_loss([test_map, test_map], test_agent_state),
                                    torch.Tensor([[0.0, 1.0], [0.0, 1.0]]))))

        test_lanelet.attributes["parking"] = ''
        self.assertTrue(
            torch.all(torch.isclose(lanelet_orientation_loss([test_map, test_map], test_agent_state),
                                    torch.Tensor([[0.0, 0.0], [0.0, 0.0]]))))
