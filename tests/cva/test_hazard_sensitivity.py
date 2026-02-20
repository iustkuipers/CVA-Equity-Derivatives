"""
tests/cva/test_hazard_sensitivity.py

Comprehensive test suite for cva.hazard_sensitivity module.

Tests cover:
- Helper functions for data dict access/manipulation
- Forward hazard rate bumping functionality
- CVA sensitivity computation
- Error handling and edge cases
- Different data structure formats
"""

import copy
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from cva.hazard_sensitivity import (
    _get_forward_hazard_rates,
    _set_forward_hazard_rates,
    _get_lgd,
    _get_risk_free_rate,
    bump_forward_hazard,
    compute_cva_sensitivities,
)


# =====================================================================
# Fixtures: Sample test data
# =====================================================================

@pytest.fixture
def base_data_format_1():
    """Data structure with top-level forward_hazard_rates."""
    return {
        "forward_hazard_rates": {
            "0_1": 0.005,
            "1_3": 0.007,
            "3_5": 0.009,
            "5_10": 0.011,
        },
        "lgd": 0.60,
        "risk_free_rate": 0.03,
    }


@pytest.fixture
def base_data_format_2():
    """Data structure with nested credit.forward_hazard_rates."""
    return {
        "credit": {
            "forward_hazard_rates": {
                "0_1": 0.005,
                "1_3": 0.007,
                "3_5": 0.009,
            },
            "lgd": 0.50,
        },
        "rates": {
            "risk_free_rate": 0.02,
        },
    }


@pytest.fixture
def base_data_format_3():
    """Data structure with alternative key names."""
    return {
        "forward_hazard_rates": {
            "0_1": 0.004,
            "1_5": 0.006,
        },
        "LGD": 0.45,
        "r": 0.025,
    }


@pytest.fixture
def base_data_single_bucket():
    """Data with single hazard bucket."""
    return {
        "forward_hazard_rates": {"0_5": 0.008},
        "lgd": 0.55,
        "risk_free_rate": 0.03,
    }


@pytest.fixture
def epe_profile():
    """Sample EPE profile (Expected Positive Exposure)."""
    return np.array([100.0, 150.0, 200.0, 250.0, 280.0])


@pytest.fixture
def times_array():
    """Sample time points."""
    return np.array([0.25, 0.5, 1.0, 2.0, 5.0])


@pytest.fixture
def times_array_extended():
    """Extended time points."""
    return np.array([0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0])


@pytest.fixture
def epe_profile_extended():
    """Extended EPE profile."""
    return np.array([50.0, 100.0, 130.0, 170.0, 220.0, 250.0, 280.0, 300.0])


# =====================================================================
# Tests: _get_forward_hazard_rates
# =====================================================================

class TestGetForwardHazardRates:
    """Test _get_forward_hazard_rates function."""

    def test_top_level_location(self, base_data_format_1):
        """Should retrieve rates from top-level key."""
        rates = _get_forward_hazard_rates(base_data_format_1)
        assert isinstance(rates, dict)
        assert "0_1" in rates
        assert rates["0_1"] == 0.005
        assert len(rates) == 4

    def test_nested_credit_location(self, base_data_format_2):
        """Should retrieve rates from data['credit']['forward_hazard_rates']."""
        rates = _get_forward_hazard_rates(base_data_format_2)
        assert isinstance(rates, dict)
        assert "0_1" in rates
        assert rates["0_1"] == 0.005
        assert len(rates) == 3

    def test_single_bucket(self, base_data_single_bucket):
        """Should handle single bucket case."""
        rates = _get_forward_hazard_rates(base_data_single_bucket)
        assert len(rates) == 1
        assert "0_5" in rates
        assert rates["0_5"] == 0.008

    def test_missing_forward_hazard_rates(self):
        """Should raise KeyError when forward_hazard_rates not found."""
        data = {
            "lgd": 0.60,
            "risk_free_rate": 0.03,
        }
        with pytest.raises(KeyError, match="Could not find forward hazard rates"):
            _get_forward_hazard_rates(data)

    def test_missing_in_nested_structure(self):
        """Should raise KeyError when nested structure incomplete."""
        data = {
            "credit": {
                "lgd": 0.60,
            },
            "rates": {
                "risk_free_rate": 0.03,
            },
        }
        with pytest.raises(KeyError, match="Could not find forward hazard rates"):
            _get_forward_hazard_rates(data)

    def test_empty_forward_hazard_dict(self):
        """Should handle empty forward_hazard_rates dict."""
        data = {
            "forward_hazard_rates": {},
            "lgd": 0.60,
            "risk_free_rate": 0.03,
        }
        rates = _get_forward_hazard_rates(data)
        assert rates == {}

    def test_non_dict_forward_hazard_ignored(self):
        """Should ignore non-dict forward_hazard_rates and check nested."""
        data = {
            "forward_hazard_rates": "not a dict",
            "credit": {
                "forward_hazard_rates": {
                    "0_1": 0.005,
                },
            },
        }
        rates = _get_forward_hazard_rates(data)
        assert "0_1" in rates

    def test_returns_reference_to_dict(self, base_data_format_1):
        """Should return reference to actual dict (not a copy by default)."""
        rates = _get_forward_hazard_rates(base_data_format_1)
        assert isinstance(rates, dict)
        # Verify it's the actual dict
        assert rates is base_data_format_1["forward_hazard_rates"]


# =====================================================================
# Tests: _set_forward_hazard_rates
# =====================================================================

class TestSetForwardHazardRates:
    """Test _set_forward_hazard_rates function."""

    def test_set_to_top_level_location(self, base_data_format_1):
        """Should set rates at top-level location."""
        new_rates = {"0_1": 0.010, "1_3": 0.015}
        _set_forward_hazard_rates(base_data_format_1, new_rates)
        assert base_data_format_1["forward_hazard_rates"] == new_rates

    def test_set_to_nested_location(self, base_data_format_2):
        """Should set rates at nested credit location."""
        new_rates = {"0_1": 0.010, "1_3": 0.015}
        _set_forward_hazard_rates(base_data_format_2, new_rates)
        assert base_data_format_2["credit"]["forward_hazard_rates"] == new_rates

    def test_set_empty_dict(self, base_data_format_1):
        """Should handle setting empty dict."""
        _set_forward_hazard_rates(base_data_format_1, {})
        assert base_data_format_1["forward_hazard_rates"] == {}

    def test_set_overwrites_safely(self, base_data_format_1):
        """Should overwrite existing rates without side effects."""
        original_lgd = base_data_format_1["lgd"]
        new_rates = {"new_bucket": 0.012}
        _set_forward_hazard_rates(base_data_format_1, new_rates)
        assert base_data_format_1["forward_hazard_rates"] == new_rates
        assert base_data_format_1["lgd"] == original_lgd

    def test_set_with_float_values(self, base_data_format_1):
        """Should handle various float formats."""
        new_rates = {
            "0_1": 0.001,
            "1_3": 1e-3,
            "3_5": 0.005,
        }
        _set_forward_hazard_rates(base_data_format_1, new_rates)
        assert all(isinstance(v, float) or isinstance(v, (int, np.number)) for v in base_data_format_1["forward_hazard_rates"].values())

    def test_set_missing_location_raises(self):
        """Should raise KeyError when location cannot be determined."""
        data = {"some": "data"}
        new_rates = {"0_1": 0.01}
        with pytest.raises(KeyError, match="cannot set forward hazard rates"):
            _set_forward_hazard_rates(data, new_rates)


# =====================================================================
# Tests: _get_lgd
# =====================================================================

class TestGetLgd:
    """Test _get_lgd function."""

    def test_top_level_lgd(self, base_data_format_1):
        """Should retrieve LGD from top level."""
        lgd = _get_lgd(base_data_format_1)
        assert lgd == 0.60
        assert isinstance(lgd, float)

    def test_top_level_LGD_uppercase(self, base_data_format_3):
        """Should handle uppercase LGD key."""
        lgd = _get_lgd(base_data_format_3)
        assert lgd == 0.45

    def test_nested_lgd_in_credit(self, base_data_format_2):
        """Should retrieve LGD from nested credit dict."""
        lgd = _get_lgd(base_data_format_2)
        assert lgd == 0.50

    def test_lgd_as_string_converts_to_float(self):
        """Should convert string LGD to float."""
        data = {"lgd": "0.60"}
        lgd = _get_lgd(data)
        assert lgd == 0.60
        assert isinstance(lgd, float)

    def test_lgd_as_int_converts_to_float(self):
        """Should convert int LGD to float."""
        data = {"lgd": 1}
        lgd = _get_lgd(data)
        assert lgd == 1.0
        assert isinstance(lgd, float)

    def test_missing_lgd_raises(self):
        """Should raise KeyError when LGD not found."""
        data = {
            "forward_hazard_rates": {"0_1": 0.01},
            "risk_free_rate": 0.03,
        }
        with pytest.raises(KeyError, match="Could not find LGD"):
            _get_lgd(data)

    def test_lgd_priority_top_over_nested(self):
        """Should prefer top-level lgd over credit.lgd."""
        data = {
            "lgd": 0.70,
            "credit": {
                "lgd": 0.50,
            },
        }
        lgd = _get_lgd(data)
        assert lgd == 0.70

    def test_various_lgd_values(self):
        """Should handle realistic LGD values (0 to 1)."""
        for lgd_val in [0.0, 0.25, 0.50, 0.75, 1.0]:
            data = {"lgd": lgd_val}
            lgd = _get_lgd(data)
            assert lgd == lgd_val


# =====================================================================
# Tests: _get_risk_free_rate
# =====================================================================

class TestGetRiskFreeRate:
    """Test _get_risk_free_rate function."""

    def test_top_level_risk_free_rate(self, base_data_format_1):
        """Should retrieve rate from top level."""
        r = _get_risk_free_rate(base_data_format_1)
        assert r == 0.03
        assert isinstance(r, float)

    def test_alternative_key_r(self, base_data_format_3):
        """Should handle 'r' key."""
        r = _get_risk_free_rate(base_data_format_3)
        assert r == 0.025

    def test_nested_in_rates_dict(self, base_data_format_2):
        """Should retrieve rate from nested rates dict."""
        r = _get_risk_free_rate(base_data_format_2)
        assert r == 0.02

    def test_rate_as_string_converts(self):
        """Should convert string rate to float."""
        data = {"risk_free_rate": "0.03"}
        r = _get_risk_free_rate(data)
        assert r == 0.03
        assert isinstance(r, float)

    def test_rate_as_percentage_when_numeric(self):
        """Should handle numeric rates."""
        data = {"risk_free_rate": 3}  # Could be interpreted as 3% or 3
        r = _get_risk_free_rate(data)
        assert r == 3.0

    def test_missing_rate_raises(self):
        """Should raise KeyError when rate not found."""
        data = {
            "forward_hazard_rates": {"0_1": 0.01},
            "lgd": 0.60,
        }
        with pytest.raises(KeyError, match="Could not find risk-free rate"):
            _get_risk_free_rate(data)

    def test_key_priority(self):
        """Should find rate in priority order."""
        data = {
            "RISK_FREE_RATE": 0.02,
            "risk_free_rate": 0.03,
        }
        r = _get_risk_free_rate(data)
        # Should find risk_free_rate first based on loop order
        assert r == 0.03

    def test_realistic_rate_values(self):
        """Should handle realistic rate values."""
        for rate_val in [0.0, 0.01, 0.02, 0.05, 0.10]:
            data = {"risk_free_rate": rate_val}
            r = _get_risk_free_rate(data)
            assert r == rate_val


# =====================================================================
# Tests: bump_forward_hazard
# =====================================================================

class TestBumpForwardHazard:
    """Test bump_forward_hazard function."""

    def test_basic_bump_10bps(self, base_data_format_1):
        """Should bump a bucket by 10bps (0.001) by default."""
        bucket = "0_1"
        original_rate = base_data_format_1["forward_hazard_rates"][bucket]
        
        bumped_data = bump_forward_hazard(base_data_format_1, bucket)
        
        bumped_rate = bumped_data["forward_hazard_rates"][bucket]
        assert bumped_rate == pytest.approx(original_rate + 0.001)

    def test_custom_bump_size(self, base_data_format_1):
        """Should respect custom bump size."""
        bucket = "1_3"
        original_rate = base_data_format_1["forward_hazard_rates"][bucket]
        custom_bump = 0.0050  # 50bps
        
        bumped_data = bump_forward_hazard(base_data_format_1, bucket, bump=custom_bump)
        
        bumped_rate = bumped_data["forward_hazard_rates"][bucket]
        assert bumped_rate == pytest.approx(original_rate + custom_bump)

    def test_negative_bump(self, base_data_format_1):
        """Should handle negative bumps (down bumps)."""
        bucket = "3_5"
        original_rate = base_data_format_1["forward_hazard_rates"][bucket]
        down_bump = -0.002  # -20bps
        
        bumped_data = bump_forward_hazard(base_data_format_1, bucket, bump=down_bump)
        
        bumped_rate = bumped_data["forward_hazard_rates"][bucket]
        assert bumped_rate == pytest.approx(original_rate + down_bump)

    def test_deep_copy_independence(self, base_data_format_1):
        """Bumped data should be independent from original."""
        bucket = "0_1"
        bumped_data = bump_forward_hazard(base_data_format_1, bucket)
        
        # Modify bumped data
        bumped_data["forward_hazard_rates"]["0_1"] = 0.999
        bumped_data["lgd"] = 0.99
        
        # Original should be unchanged
        assert base_data_format_1["forward_hazard_rates"][bucket] == 0.005
        assert base_data_format_1["lgd"] == 0.60

    def test_only_one_bucket_bumped(self, base_data_format_1):
        """Only specified bucket should be bumped, others unchanged."""
        bucket = "1_3"
        original_rates = base_data_format_1["forward_hazard_rates"].copy()
        
        bumped_data = bump_forward_hazard(base_data_format_1, bucket)
        bumped_rates = bumped_data["forward_hazard_rates"]
        
        for b, rate in original_rates.items():
            if b == bucket:
                assert bumped_rates[b] != rate
            else:
                assert bumped_rates[b] == rate

    def test_bump_preserves_other_fields(self, base_data_format_1):
        """Bumping should preserve other fields in data dict."""
        bucket = "0_1"
        bumped_data = bump_forward_hazard(base_data_format_1, bucket)
        
        assert bumped_data["lgd"] == base_data_format_1["lgd"]
        assert bumped_data["risk_free_rate"] == base_data_format_1["risk_free_rate"]

    def test_nonexistent_bucket_raises(self, base_data_format_1):
        """Should raise KeyError for non-existent bucket."""
        with pytest.raises(KeyError, match="Bucket 'nonexistent' not found"):
            bump_forward_hazard(base_data_format_1, "nonexistent")

    def test_bump_nested_structure(self, base_data_format_2):
        """Should work with nested credit structure."""
        bucket = "0_1"
        original_rate = base_data_format_2["credit"]["forward_hazard_rates"][bucket]
        
        bumped_data = bump_forward_hazard(base_data_format_2, bucket)
        bumped_rate = bumped_data["credit"]["forward_hazard_rates"][bucket]
        
        assert bumped_rate == pytest.approx(original_rate + 0.001)

    def test_bump_single_bucket_data(self, base_data_single_bucket):
        """Should work with single bucket data."""
        bucket = "0_5"
        original_rate = base_data_single_bucket["forward_hazard_rates"][bucket]
        
        bumped_data = bump_forward_hazard(base_data_single_bucket, bucket)
        bumped_rate = bumped_data["forward_hazard_rates"][bucket]
        
        assert bumped_rate == pytest.approx(original_rate + 0.001)

    def test_zero_bump(self, base_data_format_1):
        """Should allow zero bump (identity operation)."""
        bucket = "0_1"
        original_rate = base_data_format_1["forward_hazard_rates"][bucket]
        
        bumped_data = bump_forward_hazard(base_data_format_1, bucket, bump=0.0)
        bumped_rate = bumped_data["forward_hazard_rates"][bucket]
        
        assert bumped_rate == pytest.approx(original_rate)

    def test_bump_very_small_value(self, base_data_format_1):
        """Should handle very small bumps."""
        bucket = "0_1"
        small_bump = 1e-6
        original_rate = base_data_format_1["forward_hazard_rates"][bucket]
        
        bumped_data = bump_forward_hazard(base_data_format_1, bucket, bump=small_bump)
        bumped_rate = bumped_data["forward_hazard_rates"][bucket]
        
        assert bumped_rate == pytest.approx(original_rate + small_bump)

    def test_bump_large_value(self, base_data_format_1):
        """Should handle large bumps."""
        bucket = "0_1"
        large_bump = 0.10
        original_rate = base_data_format_1["forward_hazard_rates"][bucket]
        
        bumped_data = bump_forward_hazard(base_data_format_1, bucket, bump=large_bump)
        bumped_rate = bumped_data["forward_hazard_rates"][bucket]
        
        assert bumped_rate == pytest.approx(original_rate + large_bump)


# =====================================================================
# Tests: compute_cva_sensitivities
# =====================================================================

class TestComputeCvaSensitivities:
    """Test compute_cva_sensitivities function."""

    @patch('cva.hazard_sensitivity.compute_survival_and_default')
    @patch('cva.hazard_sensitivity.compute_cva_from_epe')
    def test_basic_sensitivity_computation(
        self,
        mock_cva_from_epe,
        mock_survival_default,
        base_data_format_1,
        epe_profile,
        times_array,
    ):
        """Should compute sensitivities for all buckets."""
        # Mock survival/default computation
        mock_default_probs_base = np.array([0.01, 0.02, 0.05, 0.10, 0.20])
        mock_default_probs_bumped = np.array([0.015, 0.025, 0.055, 0.105, 0.205])
        
        mock_survival_default.side_effect = [
            (None, mock_default_probs_base),  # Base
            (None, mock_default_probs_bumped),  # Bump 1
            (None, mock_default_probs_bumped),  # Bump 2
            (None, mock_default_probs_bumped),  # Bump 3
            (None, mock_default_probs_bumped),  # Bump 4
        ]
        
        # Mock CVA computation
        cva_base = 10.0
        cva_bumped = 12.0
        mock_cva_from_epe.side_effect = [cva_base] + [cva_bumped] * 4
        
        sensitivities = compute_cva_sensitivities(
            base_data_format_1,
            epe_profile,
            times_array,
        )
        
        # Should have entry for each bucket
        assert len(sensitivities) == 4
        assert "0_1" in sensitivities
        assert "1_3" in sensitivities
        assert "3_5" in sensitivities
        assert "5_10" in sensitivities
        
        # All sensitivities should be 2.0 (bump effect)
        for bucket, sens in sensitivities.items():
            assert sens == pytest.approx(2.0)

    @patch('cva.hazard_sensitivity.compute_survival_and_default')
    @patch('cva.hazard_sensitivity.compute_cva_from_epe')
    def test_sensitivities_are_positive_for_up_bump(
        self,
        mock_cva_from_epe,
        mock_survival_default,
        base_data_format_1,
        epe_profile,
        times_array,
    ):
        """CVA sensitivity should typically be positive when hazard rate increases."""
        mock_default_probs_base = np.array([0.01, 0.02, 0.05, 0.10, 0.20])
        mock_default_probs_bumped = np.array([0.015, 0.025, 0.055, 0.105, 0.205])
        
        mock_survival_default.side_effect = [
            (None, mock_default_probs_base),  # Base
            (None, mock_default_probs_bumped),
            (None, mock_default_probs_bumped),
            (None, mock_default_probs_bumped),
            (None, mock_default_probs_bumped),
        ]
        
        cva_base = 10.0
        cva_bumped = 10.5  # Higher due to higher default probability
        mock_cva_from_epe.side_effect = [cva_base] + [cva_bumped] * 4
        
        sensitivities = compute_cva_sensitivities(
            base_data_format_1,
            epe_profile,
            times_array,
        )
        
        for sens in sensitivities.values():
            assert sens >= 0.0

    @patch('cva.hazard_sensitivity.compute_survival_and_default')
    @patch('cva.hazard_sensitivity.compute_cva_from_epe')
    def test_sensitivities_different_per_bucket(
        self,
        mock_cva_from_epe,
        mock_survival_default,
        base_data_format_1,
        epe_profile,
        times_array,
    ):
        """Different buckets may have different sensitivities."""
        mock_default_probs_base = np.array([0.01, 0.02, 0.05, 0.10, 0.20])
        
        # Simulate different sensitivity to different buckets
        mock_survival_default.side_effect = [
            (None, mock_default_probs_base),  # Base
            (None, np.array([0.012, 0.021, 0.051, 0.101, 0.201])),  # Small bump
            (None, np.array([0.015, 0.025, 0.055, 0.105, 0.205])),  # Medium bump
            (None, np.array([0.020, 0.030, 0.060, 0.110, 0.210])),  # Large bump
            (None, np.array([0.011, 0.021, 0.051, 0.101, 0.201])),  # Tiny bump
        ]
        
        mock_cva_from_epe.side_effect = [10.0, 10.5, 11.0, 12.0, 10.2]
        
        sensitivities = compute_cva_sensitivities(
            base_data_format_1,
            epe_profile,
            times_array,
        )
        
        # Check sensitivities are different
        sens_values = list(sensitivities.values())
        assert len(set(sens_values)) > 1  # Not all the same

    def test_mismatched_epe_times_raises(self, base_data_format_1):
        """Should raise ValueError if epe and times have different lengths."""
        epe = np.array([100.0, 150.0, 200.0])
        times = np.array([0.25, 0.5])
        
        with pytest.raises(ValueError, match="epe and times length mismatch"):
            compute_cva_sensitivities(base_data_format_1, epe, times)

    @patch('cva.hazard_sensitivity.compute_survival_and_default')
    @patch('cva.hazard_sensitivity.compute_cva_from_epe')
    def test_custom_bump_size_used(
        self,
        mock_cva_from_epe,
        mock_survival_default,
        base_data_format_1,
        epe_profile,
        times_array,
    ):
        """Should use custom bump size parameter."""
        mock_default_probs = np.array([0.01, 0.02, 0.05, 0.10, 0.20])
        mock_survival_default.side_effect = [
            (None, mock_default_probs),  # Base
        ] + [(None, mock_default_probs)] * 4  # All bumped
        
        mock_cva_from_epe.side_effect = [10.0] * 5
        
        custom_bump = 0.0050  # 50bps instead of default 10bps
        sensitivities = compute_cva_sensitivities(
            base_data_format_1,
            epe_profile,
            times_array,
            bump=custom_bump,
        )
        
        # Verify bump_forward_hazard was called (mocked through side effects)
        assert len(sensitivities) == 4

    @patch('cva.hazard_sensitivity.compute_survival_and_default')
    @patch('cva.hazard_sensitivity.compute_cva_from_epe')
    def test_numpy_array_conversion(
        self,
        mock_cva_from_epe,
        mock_survival_default,
        base_data_format_1,
    ):
        """Should handle list inputs and convert to numpy arrays."""
        epe = [100.0, 150.0, 200.0, 250.0, 280.0]  # List instead of array
        times = [0.25, 0.5, 1.0, 2.0, 5.0]  # List instead of array
        
        mock_default_probs = np.array([0.01, 0.02, 0.05, 0.10, 0.20])
        mock_survival_default.side_effect = [
            (None, mock_default_probs),
        ] + [(None, mock_default_probs)] * 4
        
        mock_cva_from_epe.side_effect = [10.0] * 5
        
        sensitivities = compute_cva_sensitivities(
            base_data_format_1,
            epe,
            times,
        )
        
        assert isinstance(sensitivities, dict)

    @patch('cva.hazard_sensitivity.compute_survival_and_default')
    @patch('cva.hazard_sensitivity.compute_cva_from_epe')
    def test_extended_time_profile(
        self,
        mock_cva_from_epe,
        mock_survival_default,
        base_data_format_1,
        epe_profile_extended,
        times_array_extended,
    ):
        """Should handle extended time profiles."""
        mock_default_probs = np.linspace(0.01, 0.50, len(times_array_extended))
        mock_survival_default.side_effect = [
            (None, mock_default_probs),
        ] + [(None, mock_default_probs)] * 4
        
        mock_cva_from_epe.side_effect = [10.0] * 5
        
        sensitivities = compute_cva_sensitivities(
            base_data_format_1,
            epe_profile_extended,
            times_array_extended,
        )
        
        assert len(sensitivities) == 4

    @patch('cva.hazard_sensitivity.compute_survival_and_default')
    @patch('cva.hazard_sensitivity.compute_cva_from_epe')
    def test_single_bucket_sensitivities(
        self,
        mock_cva_from_epe,
        mock_survival_default,
        base_data_single_bucket,
        epe_profile,
        times_array,
    ):
        """Should work with single bucket data."""
        mock_default_probs = np.array([0.01, 0.02, 0.05, 0.10, 0.20])
        mock_survival_default.side_effect = [
            (None, mock_default_probs),  # Base
            (None, np.array([0.015, 0.025, 0.055, 0.105, 0.205])),  # Bumped
        ]
        
        mock_cva_from_epe.side_effect = [10.0, 10.5]
        
        sensitivities = compute_cva_sensitivities(
            base_data_single_bucket,
            epe_profile,
            times_array,
        )
        
        assert len(sensitivities) == 1
        assert "0_5" in sensitivities
        assert sensitivities["0_5"] == pytest.approx(0.5)

    @patch('cva.hazard_sensitivity.compute_survival_and_default')
    @patch('cva.hazard_sensitivity.compute_cva_from_epe')
    def test_sensitivities_dict_structure(
        self,
        mock_cva_from_epe,
        mock_survival_default,
        base_data_format_1,
        epe_profile,
        times_array,
    ):
        """Returned sensitivities should be dict with correct structure."""
        mock_default_probs = np.array([0.01, 0.02, 0.05, 0.10, 0.20])
        mock_survival_default.side_effect = [
            (None, mock_default_probs),
        ] + [(None, mock_default_probs)] * 4
        
        mock_cva_from_epe.side_effect = [10.0] * 5
        
        sensitivities = compute_cva_sensitivities(
            base_data_format_1,
            epe_profile,
            times_array,
        )
        
        assert isinstance(sensitivities, dict)
        for key, value in sensitivities.items():
            assert isinstance(key, str)
            assert isinstance(value, (float, int, np.number))

    @patch('cva.hazard_sensitivity.compute_survival_and_default')
    @patch('cva.hazard_sensitivity.compute_cva_from_epe')
    def test_negative_sensitivities_possible(
        self,
        mock_cva_from_epe,
        mock_survival_default,
        base_data_format_1,
        epe_profile,
        times_array,
    ):
        """Sensitivities can be negative if CVA_bumped < CVA_base."""
        mock_default_probs_base = np.array([0.02, 0.04, 0.08, 0.12, 0.25])
        mock_default_probs_bumped = np.array([0.015, 0.035, 0.075, 0.115, 0.245])
        
        mock_survival_default.side_effect = [
            (None, mock_default_probs_base),  # Base (higher defaults)
        ] + [(None, mock_default_probs_bumped)] * 4  # Lower defaults
        
        # Base CVA higher due to higher default probs
        mock_cva_from_epe.side_effect = [15.0, 12.0, 12.0, 12.0, 12.0]
        
        sensitivities = compute_cva_sensitivities(
            base_data_format_1,
            epe_profile,
            times_array,
        )
        
        # All sensitivities should be negative (CVA decreased from bump)
        for sens in sensitivities.values():
            assert sens < 0.0

    @patch('cva.hazard_sensitivity.compute_survival_and_default')
    @patch('cva.hazard_sensitivity.compute_cva_from_epe')
    def test_zero_epe_profile(
        self,
        mock_cva_from_epe,
        mock_survival_default,
        base_data_format_1,
        times_array,
    ):
        """Should handle zero EPE profile."""
        epe = np.zeros(5)
        mock_default_probs = np.array([0.01, 0.02, 0.05, 0.10, 0.20])
        mock_survival_default.side_effect = [
            (None, mock_default_probs),
        ] + [(None, mock_default_probs)] * 4
        
        mock_cva_from_epe.side_effect = [0.0] * 5
        
        sensitivities = compute_cva_sensitivities(
            base_data_format_1,
            epe,
            times_array,
        )
        
        # All sensitivities should be zero
        for sens in sensitivities.values():
            assert sens == 0.0

    @patch('cva.hazard_sensitivity.compute_survival_and_default')
    @patch('cva.hazard_sensitivity.compute_cva_from_epe')
    def test_very_long_epe_profile(
        self,
        mock_cva_from_epe,
        mock_survival_default,
        base_data_format_1,
    ):
        """Should handle very long EPE profiles."""
        epe = np.linspace(100, 300, 100)  # 100 time steps
        times = np.linspace(0.1, 10.0, 100)
        
        mock_default_probs = np.linspace(0.01, 0.50, 100)
        mock_survival_default.side_effect = [
            (None, mock_default_probs),
        ] + [(None, mock_default_probs)] * 4
        
        mock_cva_from_epe.side_effect = [10.0] * 5
        
        sensitivities = compute_cva_sensitivities(
            base_data_format_1,
            epe,
            times,
        )
        
        assert len(sensitivities) == 4


# =====================================================================
# Integration Tests
# =====================================================================

class TestIntegration:
    """Integration tests combining multiple functions."""

    @patch('cva.hazard_sensitivity.compute_survival_and_default')
    @patch('cva.hazard_sensitivity.compute_cva_from_epe')
    def test_full_workflow(
        self,
        mock_cva_from_epe,
        mock_survival_default,
        base_data_format_1,
        epe_profile,
        times_array,
    ):
        """Test complete workflow: initialize -> bump -> compute sensitivities."""
        # Verify initial setup
        rates = _get_forward_hazard_rates(base_data_format_1)
        assert "0_1" in rates
        
        lgd = _get_lgd(base_data_format_1)
        assert 0 <= lgd <= 1
        
        r = _get_risk_free_rate(base_data_format_1)
        assert r > 0
        
        # Mock for sensitivity computation
        mock_default_probs = np.array([0.01, 0.02, 0.05, 0.10, 0.20])
        mock_survival_default.side_effect = [
            (None, mock_default_probs),
        ] + [(None, mock_default_probs)] * 4
        
        mock_cva_from_epe.side_effect = [10.0] * 5
        
        sensitivities = compute_cva_sensitivities(
            base_data_format_1,
            epe_profile,
            times_array,
        )
        
        assert len(sensitivities) > 0

    def test_data_independence_across_bumps(self, base_data_format_1):
        """Multiple bumps should not interfere with each other."""
        bucket1 = "0_1"
        bucket2 = "1_3"
        
        bumped1 = bump_forward_hazard(base_data_format_1, bucket1)
        bumped2 = bump_forward_hazard(base_data_format_1, bucket2)
        
        # First bump should not affect bucket2
        assert bumped1["forward_hazard_rates"][bucket2] == base_data_format_1["forward_hazard_rates"][bucket2]
        
        # Second bump should not affect bucket1
        assert bumped2["forward_hazard_rates"][bucket1] == base_data_format_1["forward_hazard_rates"][bucket1]

    def test_multiple_sequential_bumps(self, base_data_format_1):
        """Should handle multiple sequential bumps."""
        bucket = "0_1"
        original = base_data_format_1["forward_hazard_rates"][bucket]
        
        bumped1 = bump_forward_hazard(base_data_format_1, bucket, bump=0.001)
        bumped2 = bump_forward_hazard(bumped1, bucket, bump=0.001)
        
        final_rate = bumped2["forward_hazard_rates"][bucket]
        assert final_rate == pytest.approx(original + 0.002)


# =====================================================================
# Edge Case Tests
# =====================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_large_hazard_rates(self):
        """Should handle unrealistically large hazard rates."""
        data = {
            "forward_hazard_rates": {"0_1": 10.0},
            "lgd": 0.60,
            "risk_free_rate": 0.03,
        }
        bumped = bump_forward_hazard(data, "0_1", bump=0.001)
        assert bumped["forward_hazard_rates"]["0_1"] > data["forward_hazard_rates"]["0_1"]

    def test_very_small_hazard_rates(self):
        """Should handle very small hazard rates."""
        data = {
            "forward_hazard_rates": {"0_1": 1e-6},
            "lgd": 0.60,
            "risk_free_rate": 0.03,
        }
        bumped = bump_forward_hazard(data, "0_1")
        assert bumped["forward_hazard_rates"]["0_1"] > data["forward_hazard_rates"]["0_1"]

    def test_bucket_with_special_characters(self):
        """Should handle bucket keys with underscores."""
        data = {
            "forward_hazard_rates": {"0_1": 0.005, "10_inf": 0.015},
            "lgd": 0.60,
            "risk_free_rate": 0.03,
        }
        bumped = bump_forward_hazard(data, "10_inf")
        assert bumped["forward_hazard_rates"]["10_inf"] > data["forward_hazard_rates"]["10_inf"]

    def test_lgd_at_boundaries(self):
        """Should handle LGD at boundary values."""
        for lgd_val in [0.0, 1.0]:
            data = {"lgd": lgd_val}
            lgd = _get_lgd(data)
            assert lgd == lgd_val

    def test_zero_risk_free_rate(self):
        """Should handle zero risk-free rate."""
        data = {"risk_free_rate": 0.0}
        r = _get_risk_free_rate(data)
        assert r == 0.0


# =====================================================================
# Property-Based / Parametrized Tests
# =====================================================================

class TestParametrized:
    """Parametrized tests for multiple scenarios."""

    @pytest.mark.parametrize("bump_size", [0.0001, 0.001, 0.005, 0.01, 0.05])
    def test_various_bump_sizes(self, base_data_format_1, bump_size):
        """Test bumping with various sizes."""
        bucket = "0_1"
        original = base_data_format_1["forward_hazard_rates"][bucket]
        
        bumped = bump_forward_hazard(base_data_format_1, bucket, bump=bump_size)
        bumped_rate = bumped["forward_hazard_rates"][bucket]
        
        assert bumped_rate == pytest.approx(original + bump_size)

    @pytest.mark.parametrize("bucket", ["0_1", "1_3", "3_5", "5_10"])
    def test_bump_each_bucket(self, base_data_format_1, bucket):
        """Test bumping each bucket individually."""
        original = base_data_format_1["forward_hazard_rates"][bucket]
        bumped = bump_forward_hazard(base_data_format_1, bucket)
        bumped_rate = bumped["forward_hazard_rates"][bucket]
        
        assert bumped_rate > original

    @pytest.mark.parametrize("lgd", [0.0, 0.25, 0.50, 0.75, 1.0])
    def test_various_lgd_values_extraction(self, lgd):
        """Test LGD extraction with various values."""
        data = {"lgd": lgd}
        extracted = _get_lgd(data)
        assert extracted == lgd


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
