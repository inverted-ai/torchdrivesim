import math
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
COMMON_ANGLES = [
    0,
    PI/6,
    PI/4,
    PI/2,

    3*PI/4,
    5*PI/6,
    PI,

    7*PI/6,
    5*PI/4,
    3*PI/2,

    7*PI/4,
    11*PI/6,
    2*PI  # Duplicate but good to have
]


class TestIsIn:
    max_int32 = torch.iinfo(torch.int32).max
    min_int32 = torch.iinfo(torch.int32).min
    y_tensor = torch.tensor([min_int32, -100, -3, -1, 0, 1, 3, 100, max_int32], dtype=torch.int32)

    # Test different dtype scalar
    # Test scalar x
    # Test (1,) x
    # Test large (say (2,3)) x
    
    @pytest.mark.parametrize(
        "x_value, expected",
        [
            (-3, torch.tensor(True)),
            (42, torch.tensor(False)),
            (0, torch.tensor(True)),
            (min_int32, torch.tensor(True)),
            (max_int32, torch.tensor(True)),
            (-50, torch.tensor(False)),
        ]
    )
    @pytest.mark.parametrize(
        "dtype", [torch.int32, torch.int64]
    ) # Test if dtype is converted automatically e.g. int64 to int32
    def test_scalar_values(self, x_value, expected, dtype):
        x = torch.tensor(x_value, dtype=dtype)
        result = isin(x, self.y_tensor)
        assert result.shape == x.shape
        assert torch.equal(result, expected)

    @pytest.mark.parametrize(
        "x_tensor, expected",
        [
            (torch.tensor([-1], dtype=torch.int32), torch.tensor([True])),
            (torch.tensor([42], dtype=torch.int32), torch.tensor([False])),
            (torch.tensor([min_int32, 0, 2], dtype=torch.int32),
             torch.tensor([True, True, False])),
            (torch.tensor([[3, 50], [0, -100]], dtype=torch.int32),
             torch.tensor([[True, False], [True, True]])),
        ]
    )
    def test_tensor_values(self, x_tensor, expected):
        result = isin(x_tensor, self.y_tensor)
        assert result.shape == x_tensor.shape
        assert torch.equal(result, expected)

    def test_empty_tensor(self):
        x = torch.tensor([], dtype=torch.int32)
        expected = torch.tensor([], dtype=torch.bool)
        assert torch.equal(isin(x, self.y_tensor), expected)

    def test_y_tensor_empty(self):
        x = torch.tensor([1, 2, 3], dtype=torch.int32)
        y = torch.tensor([], dtype=torch.int32)
        expected = torch.tensor([False, False, False])
        assert torch.equal(isin(x, y), expected)
        
    def test_equal_tensor(self):
        result = isin(self.y_tensor, self.y_tensor)
        expected = torch.ones_like(self.y_tensor, dtype=torch.bool)
        assert result.shape == self.y_tensor.shape
        assert torch.equal(result, expected)
        
    def test_invalid_y_tensor(self):
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([[1, 2], [3, 4]])  # 2D, should fail

        with pytest.raises(Exception):
            isin(x, y)
        
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

class TestRelative:
    @pytest.mark.parametrize("target_xy", [[0,0], [1,1], [-1,-1],
                                           [1,-1], [-1,1], [100,100],
                                           [-100,-100], [100,-100], 
                                           [-100,100]])
    @pytest.mark.parametrize("target_psi", [0.0, PI/4, PI/2,
                                              PI, -PI/2, -PI/4])
    def test_zero_origin(self, target_xy, target_psi):
        """
        Check trivial case where origin_xy and origin_psi are that of the world origin ([0,0] and [0]) 
        """
        origin_xy = torch.tensor([[0]], dtype=torch.float64)
        origin_psi = torch.tensor([[0]], dtype=torch.float64)
        target_xy = torch.tensor([target_xy], dtype=torch.float64)
        target_psi = torch.tensor([target_psi], dtype=torch.float64)  

        rel_xy, rel_psi = relative(origin_xy, origin_psi, target_xy, target_psi)

        # With zero origin, relative should just be target minus origin
        expected_xy = target_xy - origin_xy
        expected_psi = normalize_angle(target_psi - origin_psi)

        assert torch.allclose(rel_xy, expected_xy, atol=TOLERANCE)
        assert torch.allclose(rel_psi, expected_psi, atol=TOLERANCE)

    @pytest.mark.parametrize("origin_xy", [[0,0], [1,1], [-1,-1],
                                           [1,-1], [-1,1], [100,100],
                                           [-100,-100], [100,-100], 
                                           [-100,100]])
    @pytest.mark.parametrize("target_xy", [[0,0], [1,1], [-1,-1],
                                           [1,-1], [-1,1], [100,100],
                                           [-100,-100], [100,-100], 
                                           [-100,100]])
    @pytest.mark.parametrize("target_psi", [0.0, PI/4, PI/2,
                                              PI, -PI/2, -PI/4])
    def test_nontrivial_origin_xy_zero_psi(self, origin_xy, target_xy, target_psi):
        """
        Check case where origin_psi is zero but origin_xy is non-trivial
        """
        origin_xy = torch.tensor([origin_xy], dtype=torch.float64)
        origin_psi = torch.tensor([[0]], dtype=torch.float64)
        target_xy = torch.tensor([target_xy], dtype=torch.float64)
        target_psi = torch.tensor([target_psi], dtype=torch.float64)  

        rel_xy, rel_psi = relative(origin_xy, origin_psi, target_xy, target_psi)

        expected_xy = target_xy - origin_xy
        expected_psi = normalize_angle(target_psi - origin_psi)

        assert torch.allclose(rel_xy, expected_xy, atol=TOLERANCE)
        assert torch.allclose(rel_psi, expected_psi, atol=TOLERANCE)


    @pytest.mark.parametrize("origin_psi", [0.0, PI/4, PI/2,
                                              PI, -PI/2, -PI/4])
    @pytest.mark.parametrize("target_xy", [[0,0], [1,1], [-1,-1],
                                           [1,-1], [-1,1], [100,100],
                                           [-100,-100], [100,-100], 
                                           [-100,100]])
    @pytest.mark.parametrize("target_psi", [0.0, PI/4, PI/2,
                                              PI, -PI/2, -PI/4])
    def test_zero_origin_xy_nontrivial_psi(self, origin_psi, target_xy, target_psi):
        """
        Check case where origin_xy is zero but origin_psi is non-trivial.
        Test multiple angles using pytest parametrize.
        """
        origin_xy = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
        origin_psi = torch.tensor([[origin_psi]], dtype=torch.float64)
        target_xy = torch.tensor([target_xy], dtype=torch.float64)
        target_psi = torch.tensor([target_psi], dtype=torch.float64)

        rel_xy, rel_psi = relative(origin_xy, origin_psi, target_xy, target_psi)

        # Relative position should rotate the vector by -origin_psi
        expected_xy = rotate(target_xy - origin_xy, -origin_psi)
        expected_psi = normalize_angle(target_psi - origin_psi)

        assert torch.allclose(rel_xy, expected_xy, atol=TOLERANCE)
        assert torch.allclose(rel_psi, expected_psi, atol=TOLERANCE)

    @pytest.mark.parametrize("origin_xy", [[0,0], [1,1], [-1,-1],
                                           [1,-1], [-1,1], [100,100],
                                           [-100,-100], [100,-100], 
                                           [-100,100]])
    @pytest.mark.parametrize("origin_psi", [0.0, PI/4, PI/2,
                                              PI, -PI/2, -PI/4])
    @pytest.mark.parametrize("target_xy", [[0,0], [1,1], [-1,-1],
                                           [1,-1], [-1,1], [100,100],
                                           [-100,-100], [100,-100], 
                                           [-100,100]])
    @pytest.mark.parametrize("target_psi", [0.0, PI/4, PI/2,
                                              PI, -PI/2, -PI/4])
    def test_nontrivial_origin_xy_psi_round_trip(self, origin_xy, origin_psi, target_xy, target_psi):
        """
        Shift to relative, then shift back to world origin coordinates, both using relative(). Then check if same as original target
        """
        origin_xy = torch.tensor([origin_xy], dtype=torch.float64)
        origin_psi = torch.tensor([origin_psi], dtype=torch.float64)
        target_xy = torch.tensor([target_xy], dtype=torch.float64)
        target_psi = torch.tensor([target_psi], dtype=torch.float64)  

        # Compute relative coordinates
        rel_xy, rel_psi = relative(origin_xy, origin_psi, target_xy, target_psi)

        # Compute world origin in relative coordinates
        worigin_xy, worigin_psi = relative(origin_xy, origin_psi, torch.tensor([0,0]), torch.tensor([0]))

        # Now compute relative again from recovered world origin
        rel_xy2, rel_psi2 = relative(worigin_xy, worigin_psi, rel_xy, rel_psi)

        assert torch.allclose(rel_xy2, target_xy, atol=TOLERANCE)
        assert torch.allclose(rel_psi2, target_psi, atol=TOLERANCE)

class TestTransform:
    point_list = [[0.0, 0.0],
                  [1.0, 0.0],
                  [0.0, 1.0],
                  [-1.0, 0.0],
                  [0.0, -1.0],
                  [1.0, 1.0],
                  [-1.0, -1.0],
                  [1e-9, -1e-9],
                  [1e9, -1e9],
                  [2.3, -4.7]]

    translation_only_poses = [[*p, 0.0] for p in point_list] # same as points list but each element gets a 0 attached at the end
    rotation_only_poses = [[0.0, 0.0, p] for p in COMMON_ANGLES]
    rotation_only_poses.extend([[0.0, 0.0, 1e-9],
                                [0.0, 0.0, -1e-9],
                                [0.0, 0.0, 2*PI + 1e-9],
                                [0.0, 0.0, 2*PI - 1e-9]])

    poses_list = [[1e-9, -1e-9, PI/6],
                  [1e9, -1e9, -PI/6],
                  [10.0, 10.0, torch.pi/6],
                  [-10.0, 10.0, -torch.pi/6]]
                
    @pytest.mark.parametrize("point", point_list)
    def test_identity_pose(self, point):
        """
        Transform should not change point when given identity pose.
        """
        point = torch.tensor([point]).unsqueeze(0) # Batch size 1
        pose   = torch.tensor([0.0, 0.0, 0.0]).unsqueeze(0) # Batch size 1

        out = transform(point, pose)

        assert torch.allclose(out, point)

    @pytest.mark.parametrize("pose", translation_only_poses)
    def test_translation_only(self, pose):
        """
        psi = 0 tests.
        """
        point = torch.tensor([[1.0, 1.0]]).unsqueeze(0) # Batch size 1
        pose  = torch.tensor(pose).unsqueeze(0) # Batch size 1
        
        assert torch.all(pose[:, -1] == 0.0) # sanity check to make sure we are using psi = 0 poses

        out = transform(point, pose)
        expected = point + pose[:, :-1] # discard phi of pose

        assert torch.allclose(out, expected)

    @pytest.mark.parametrize("pose", rotation_only_poses)
    def test_rotation_only(self, pose):
        """
        translation = 0 tests.
        """
        point = torch.tensor([[1.0, 1.0]]).unsqueeze(0) # Batch size 1
        pose  = torch.tensor(pose).unsqueeze(0) # Batch size 1

        assert torch.all(pose[:, 0:2] == 0.0) # sanity check to make sure we are using translation = 0 poses

        out = transform(point, pose)

        expected = rotate(point, pose[:, -1]) # discard translation of pose
        assert torch.allclose(out, expected, atol=TOLERANCE)


    @pytest.mark.parametrize("batch_shape", [
        (1,),        # simple batch
        (2, 1),      # 2x1 batch
        (2, 2, 1),   # 2x2x1 batch
    ]) # Note we treat N as a batch dimension here
    @pytest.mark.parametrize("pose", poses_list)
    def test_batched(self, batch_shape, pose):
        """
        Test transform with arbitrary batch shapes, using different points inside the batch.
        """
        total_points = 1
        for dim in batch_shape:
            total_points *= dim

        # ============================================= AI SLOP ==============================================
        # Repeat or slice point_list to fill batch
        points_flat = torch.tensor(TestTransform.point_list[:total_points], dtype=torch.float32)
        # If point_list is too short, tile it
        if points_flat.shape[0] < total_points:
            repeats = (total_points + len(TestTransform.point_list) - 1) // len(TestTransform.point_list)
            points_flat = torch.tensor(TestTransform.point_list * repeats, dtype=torch.float32)[:total_points]

        points_batch = points_flat.view(*batch_shape, 1, 2)
        # ====================================================================================================

        assert list(points_batch.shape) == [*batch_shape, 1, 2] # Sanity check

        # Compute as batch size 1 with large N, use the result to check if batched transform is correct
        points_flat_list = points_batch.view(-1, 1, 2)
        pose_ref = torch.tensor([pose] * points_flat_list.shape[0])
        expected_flat = transform(points_flat_list, pose_ref)
        expected_batch = expected_flat.view(*points_batch.shape)

        out_batch = transform(points_batch, torch.tensor([pose]*total_points).view(*batch_shape, 3))

        assert torch.allclose(out_batch, expected_batch, atol=TOLERANCE)

    @pytest.mark.parametrize("origin_xy", point_list)
    @pytest.mark.parametrize("origin_phi", rotation_only_poses)
    @pytest.mark.parametrize("target", point_list)
    def test_relative_consistency(self, origin_xy, origin_phi, target):
        """
        Ensure transform and relative are inverses using all combinations from point_list and poses_list.
        """
        origin_xy = torch.tensor([origin_xy]) # 1x2
        origin_phi = torch.tensor([[origin_phi[2]]]) #1x1
        target_xy = torch.tensor([target]) # 1x2
        target_psi = torch.tensor([[0]]) #1x1

        # Compute relative target pose in origin frame
        rel_xy, rel_psi = relative(origin_xy, origin_phi, target_xy, target_psi)

        # Transform relative coords back to global frame
        rel_points = rel_xy.unsqueeze(0)  # 1x1x2
        origin_pose_tensor = torch.cat([origin_xy, origin_phi], dim=1)  # 1x3
        out = transform(rel_points, origin_pose_tensor)[0, 0]

        assert torch.allclose(out, target_xy[0], atol=TOLERANCE)


    def test_empty_points(self):
        """
        Handle zero points.
        """
        points = torch.zeros((2, 0, 2))  # B=2, N=0
        pose = torch.zeros((2, 3))

        out = transform(points, pose)

        assert out.shape == (2, 0, 2)  # unchanged shape
    
class TestMergeDicts:
    dict_ab = {"a": 1, "b": 2}
    dict_ab_2 = {"a": 3, "b": 4}
    dict_bc = {"b": 5, "c": 6}
    dict_de = {"d": 7, "e": 8}

    dicts = [dict_ab, dict_ab_2, dict_bc, dict_de]
    results = [ #results of merging in the order of dicts list, up to a certain point
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
        {"a": 3, "b": 5, "c": 6},
        {"a": 3, "b": 5, "c": 6, "d": 7, "e": 8}      
    ] 
    
    def test_merge_dicts(self):
        for i in range(len(self.dicts)):
            assert merge_dicts(self.dicts[:i+1]) == self.results[i]