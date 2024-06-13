import unittest
import pytest
import AndroStats.performance as prf


class TestCanoeAnalysis(unittest.TestCase):
    def test_init(self):
        prf.CanoeAnalysis()
    
    def test_allowable_variance(self):
        VALUE = 50
        TRUE_AV = 10

        ca = prf.CanoeAnalysis()
        allowable_variance = ca.allowable_variance(VALUE)
        self.assertEqual(allowable_variance, TRUE_AV)

    def test_allowable_variance_value_out_of_range(self):
        VALUE = 999
        ca = prf.CanoeAnalysis()
        with pytest.raises(ValueError):
            ca.allowable_variance(VALUE)
    
    def test_prediction_within_canoe(self):
        T_VALUE = 50
        P_VALUE = 55
        ca = prf.CanoeAnalysis()
        self.assertTrue(ca.prediction_within_canoe(T_VALUE, P_VALUE))

    def test_prediction_outside_canoe(self):
        T_VALUE = 50
        P_VALUE = 999
        ca = prf.CanoeAnalysis()
        self.assertFalse(ca.prediction_within_canoe(T_VALUE, P_VALUE))
