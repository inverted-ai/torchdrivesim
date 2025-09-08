from torchdrivesim.utils import *
import pytest
import numpy as np

PI = torch.pi
PI_T = torch.tensor([torch.pi]) # 1-dimensional tensor
sqrt = np.sqrt

class TestNormalizeAngle:
    common_cases = [(1               ,               1),
                    (-1              ,              -1),
                    (0               ,               0),
                    (-PI             ,             -PI),
                    (PI              ,             -PI),
                    (PI + 1          ,         -PI + 1),
                    (2*PI            ,               0),
                    (5*PI            ,             -PI)]
    
    input_ndarray_2x4 = np.array([key for (key, _) in common_cases]).reshape(2,4)
    expected_ndarray_2x4 = np.array([val for (_, val) in common_cases]).reshape(2,4)
    input_ndarray_2x2x2 = np.array([key for (key, _) in common_cases]).reshape(2,2,2)
    expected_ndarray_2x2x2 = np.array([val for (_, val) in common_cases]).reshape(2,2,2)
    
    input_tensor_2x4 = torch.tensor([key for (key, _) in common_cases]).reshape(2,4)
    expected_tensor_2x4 = torch.tensor([val for (_, val) in common_cases]).reshape(2,4)
    input_tensor_2x2x2 = torch.tensor([key for (key, _) in common_cases]).reshape(2,2,2)
    expected_tensor_2x2x2 = torch.tensor([val for (_, val) in common_cases]).reshape(2,2,2)
    
    
    
    @pytest.mark.parametrize("angle, expected", common_cases)
    def test_normalize_angle_floats(self, angle, expected):
        """
        Test behavior when given float
        """
        ret = normalize_angle(angle)
        assert ret == expected
        
        

    @pytest.mark.parametrize("angle, expected", [(input_ndarray_2x4, expected_ndarray_2x4),
                                                 (input_ndarray_2x2x2, expected_ndarray_2x2x2)])
    def test_normalize_angle_ndarray(self, angle, expected):
        """
        Test behavior when given np array
        """
        ret = normalize_angle(angle)
        assert np.allclose(ret, expected, atol=1e-6) # allow tolerance in case some slight numerical error
        
        
        
    @pytest.mark.parametrize("angle, expected", [(input_tensor_2x4, expected_tensor_2x4),
                                                 (input_tensor_2x2x2, expected_tensor_2x2x2)])
    def test_normalize_angle_tensor(self, angle, expected):
        """
        Test behavior when given np array
        """
        ret = normalize_angle(angle)
        assert torch.allclose(ret, expected, atol=1e-6) # allow tolerance in case some slight numerical error

class TestRotationMatix:
    rot90 = torch.tensor([[0,-1],
                          [1, 0]]).float()
    rot180 = rot90 @ rot90
    rot270 = rot90 @ rot180
    
    # Common cases in first quadrant defined here
    common_cases = [(torch.tensor([0])                      ,                      torch.tensor([[1,                   0],
                                                                                                 [0,                   1]]).float()),
                    
                    (torch.tensor([PI/3])                   ,                      torch.tensor([[1/2,        -sqrt(3)/2],
                                                                                                 [sqrt(3)/2,         1/2]]).float()),
                    
                    (torch.tensor([PI/4])                   ,                      torch.tensor([[sqrt(2)/2,  -sqrt(2)/2],
                                                                                                 [sqrt(2)/2,   sqrt(2)/2]]).float()),
                    
                    (torch.tensor([PI/6])                   ,                      torch.tensor([[sqrt(3)/2,        -1/2],
                                                                                                 [1/2,         sqrt(3)/2]]).float())]
    
    # Fill out cases in other three quadrants by means of rotation
    new_cases = []
    for (theta, expected) in common_cases.copy():
        new_cases.append((theta + PI_T/2,    rot90 @ expected))
        new_cases.append((theta + PI_T,     rot180 @ expected))
        new_cases.append((theta + 3*PI_T/2, rot270 @ expected))
    common_cases += new_cases
    
    input_batch = torch.stack([key for (key, _) in common_cases])
    output_batch = torch.stack([val for (_, val) in common_cases])
    
    @pytest.mark.parametrize("theta, expected", common_cases)
    def test_rotation_matrix_simple(self, theta, expected):
        """
        Test behavior with batch size of one, with simple angles
        """
        ret = rotation_matrix(theta=theta)
        assert torch.allclose(ret, expected, atol=1e-6) # allow tolerance in case some slight numerical error
        
    @pytest.mark.parametrize("theta, expected", [(input_batch, output_batch)])
    def test_rotation_matrix_batch(self, theta, expected):
        """
        Test behavior with large batch size
        """
        ret = rotation_matrix(theta=theta)
        assert torch.allclose(ret, expected, atol=1e-6) # allow tolerance in case some slight numerical error