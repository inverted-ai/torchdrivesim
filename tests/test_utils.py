import pdb
from torchdrivesim.utils import *
import pytest
import numpy as np

# Test Settings
TOLERANCE = 1e-6  # allow tolerance in case some slight numerical error



# Common Constants
PI = torch.pi
PI_T = torch.tensor([torch.pi]) # 1-dimensional tensor. NOTE: may want to remove this and change to PI constant if we ever allow scalar inputs
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
        assert ret == expected, f"normalize_angle({angle}) returned {ret}, expected: {expected}"
        
        

    @pytest.mark.parametrize("angle, expected", [(input_ndarray_2x4, expected_ndarray_2x4),
                                                 (input_ndarray_2x2x2, expected_ndarray_2x2x2)])
    def test_normalize_angle_ndarray(self, angle, expected):
        """
        Test behavior when given np array
        """
        ret = normalize_angle(angle)
        assert np.allclose(ret, expected, atol=TOLERANCE), f"normalize_angle({angle}) returned {ret}, expected: {expected}"
        
        
        
    @pytest.mark.parametrize("angle, expected", [(input_tensor_2x4, expected_tensor_2x4),
                                                 (input_tensor_2x2x2, expected_tensor_2x2x2)])
    def test_normalize_angle_tensor(self, angle, expected):
        """
        Test behavior when given np array
        """
        ret = normalize_angle(angle)
        assert torch.allclose(ret, expected, atol=TOLERANCE), f"normalize_angle({angle}) returned {ret}, expected: {expected}"

class TestRotationMatix:
    #NOTE: perhaps the function should be changed to convert to float before running, so we don't have to explicitly pass it floats?
    #NOTE: perhaps we should allow for scalars to be passed in for angle, so we don't have to explicitly pass it 1-dimensional tensors when we are not batching?
    
    rot90 = torch.tensor([[0,-1],
                          [1, 0]]).float()          #NOTE: may want to change this if we ever allow non-float inputs
    rot180 = rot90 @ rot90
    rot270 = rot90 @ rot180
    
    # Common cases in first quadrant defined here
    common_cases = [(torch.tensor([0])                      ,                      torch.tensor([[1,                   0],
                                                                                                 [0,                   1]]).float()),   #NOTE: may want to change this if we ever allow non-float inputs
                    
                    (torch.tensor([PI/3])                   ,                      torch.tensor([[1/2,        -sqrt(3)/2],
                                                                                                 [sqrt(3)/2,         1/2]]).float()),
                    
                    (torch.tensor([PI/4])                   ,                      torch.tensor([[sqrt(2)/2,  -sqrt(2)/2],
                                                                                                 [sqrt(2)/2,   sqrt(2)/2]]).float()),
                    
                    (torch.tensor([PI/6])                   ,                      torch.tensor([[sqrt(3)/2,        -1/2],
                                                                                                 [1/2,         sqrt(3)/2]]).float())]
    
    # Fill out cases in other three quadrants by means of rotation
    new_cases = []
    for (theta, expected) in common_cases:
        new_cases.append((theta + PI_T/2,    rot90 @ expected))
        new_cases.append((theta + PI_T,     rot180 @ expected))
        new_cases.append((theta + 3*PI_T/2, rot270 @ expected))
        new_cases.append((theta - PI_T/2,   rot270 @ expected))
        new_cases.append((theta - PI_T,     rot180 @ expected))
        new_cases.append((theta - 3*PI_T/2,  rot90 @ expected))
    common_cases += new_cases
    
    input_batch = torch.stack([key for (key, _) in common_cases])
    output_batch = torch.stack([val for (_, val) in common_cases])
    
    @pytest.mark.parametrize("theta, expected", common_cases)
    def test_rotation_matrix_simple(self, theta, expected):
        """
        Test behavior with batch size of one, with simple angles
        """
        ret = rotation_matrix(theta=theta)
        assert torch.allclose(ret, expected, atol=TOLERANCE), f"rotation_matrix({theta}) returned {ret}, expected: {expected}"
        
    @pytest.mark.parametrize("theta, expected", [(input_batch, output_batch)])
    def test_rotation_matrix_batch(self, theta, expected):
        """
        Test behavior with large batch size
        """
        ret = rotation_matrix(theta=theta)
        assert torch.allclose(ret, expected, atol=TOLERANCE), f"rotation_matrix({theta}) returned {ret}, expected: {expected}"
        
class TestRotate:
    #NOTE: perhaps the function should be changed to convert to float before running, so we don't have to explicitly pass it floats?
    #NOTE: perhaps we should allow for scalars to be passed in for angle, so we don't have to explicitly pass it 1-dimensional tensors when we are not batching?
    
    rot90 = torch.tensor([[0,-1],
                          [1, 0]]).float()          #NOTE: may want to change this if we ever allow non-float inputs
    rot180 = rot90 @ rot90
    rot270 = rot90 @ rot180
    
    # Vectors whose angles are multiples of 30 degrees, in the first quadrant
    vecs = {
        0: torch.tensor([1, 0]).float(),            #NOTE: may want to change this if we ever allow non-float inputs
        30: torch.tensor([sqrt(3)/2, 1/2]).float(),
        60: torch.tensor([1/2,sqrt(3)/2]).float()
    }
    
    # Fill out vectors in the other three quadrants
    new_vecs = {}
    for (deg, v) in vecs.items():
        new_vecs.update({deg + 90:   rot90 @ v})
        new_vecs.update({deg + 180: rot180 @ v})
        new_vecs.update({deg + 270: rot270 @ v})
    vecs.update(new_vecs)
    
    @pytest.mark.parametrize("angle", list(range(-390, 391, 30))) # list of angles from -390 to 390 inclusive with step size 30
    @pytest.mark.parametrize("vector_angle", vecs.keys())
    def test_rotate_simple(self, angle, vector_angle):
        """
        Test behavior with batch size one.
        """
        ret = rotate(v=self.vecs[vector_angle], angle=torch.deg2rad(torch.tensor([angle]))) #NOTE: may want to change this if we ever allow scalar angle inputs
        expected = self.vecs[(vector_angle + angle) % 360]
        
        assert torch.allclose(ret, expected, atol=TOLERANCE), f"rotate(=) returned {ret}, expected: {expected}"
        
    @pytest.mark.parametrize("angles, vector_angles", [(torch.tensor(range(0, 30*len(vecs), 30)).unsqueeze(1), list(vecs.keys()))]) # angles start from 0, incrementing 30 for each vector.
    def test_rotate_batch_no_broadcast(self, angles, vector_angles):
        """
        Test behavior with large batch size where length of angles and vectors match
        """
        vectors = torch.stack([self.vecs[a] for a in vector_angles])
        ret = rotate(v=vectors, angle=torch.deg2rad(angles))
        expected = torch.stack([rotate(self.vecs[a], torch.deg2rad(angles)[i]) for i, a in enumerate(vector_angles)])
        
        assert torch.allclose(ret, expected, atol=TOLERANCE), f"rotate() returned {ret}, expected: {expected}"
    
    @pytest.mark.parametrize("angle", list(range(-390, 391, 30))) # list of angles from -390 to 390 inclusive with step size 30
    @pytest.mark.parametrize("vectors", [torch.stack(list(vecs.values()))])
    def test_rotate_batch_broadcast_angle(self, angle, vectors):
        """
        Test behavior with large vector batch size where size of angles is one.
        """
        angle = torch.tensor(angle).reshape(1,1)
        ret = rotate(v=vectors, angle=angle)
        expected = torch.cat([rotate(v=v.unsqueeze(0), angle=angle) for v in vectors])
        assert torch.allclose(ret, expected, atol=TOLERANCE), f"rotate() returned {ret}, expected: {expected}"
    
    @pytest.mark.parametrize("angles", [torch.tensor(list(range(-390, 391, 30))).unsqueeze(1)])
    @pytest.mark.parametrize("vector", list(vecs.values()))
    def test_rotate_batch_broadcast_vector(self, angles, vector):
        """
        Test behavior with large angle batch size where size of vectors is one.
        """
        ret = rotate(v=vector, angle=angles)
        expected = torch.cat([rotate(v=vector, angle=a.unsqueeze(0)) for a in angles])
        assert torch.allclose(ret, expected, atol=TOLERANCE), f"rotate() returned {ret}, expected: {expected}"