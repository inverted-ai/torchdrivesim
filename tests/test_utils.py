from torchdrivesim.utils import *
import pytest
import numpy as np


class TestNormalizeAngle:
    common_cases = [(1               ,               1),
                    (-1              ,              -1),
                    (0               ,               0),
                    (-np.pi          ,          -np.pi),
                    (np.pi           ,          -np.pi),
                    (np.pi + 1       ,      -np.pi + 1),
                    (2*np.pi         ,               0),
                    (5*np.pi         ,          -np.pi)]
    
    input_ndarray_2x4 = np.array([key for (key, val) in common_cases]).reshape(2,4)
    expected_ndarray_2x4 = np.array([val for (key, val) in common_cases]).reshape(2,4)
    input_ndarray_2x2x2 = np.array([key for (key, val) in common_cases]).reshape(2,2,2)
    expected_ndarray_2x2x2 = np.array([val for (key, val) in common_cases]).reshape(2,2,2)
    
    input_tensor_2x4 = torch.tensor([key for (key, val) in common_cases]).reshape(2,4)
    expected_tensor_2x4 = torch.tensor([val for (key, val) in common_cases]).reshape(2,4)
    input_tensor_2x2x2 = torch.tensor([key for (key, val) in common_cases]).reshape(2,2,2)
    expected_tensor_2x2x2 = torch.tensor([val for (key, val) in common_cases]).reshape(2,2,2)
    
    
    
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
