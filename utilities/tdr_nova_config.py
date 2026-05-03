"""TDR Nova VST3 plugin configuration and parameter presets.

TDR Nova is a free dynamic EQ by Tokyo Dawn Records, loaded via pedalboard.
All parameters use normalized raw_value (0.0-1.0) — this module converts
human-readable values to raw_value for each parameter type.

Parameter mapping (discovered via pedalboard introspection):
  - Band type:    0.0=Low Shelf, 0.5=Bell, 1.0=High Shelf
  - Dynamics:     0.0=Off, 0.5=On, 1.0=Sticky
  - Gain/Threshold: linear mapping
  - Frequency/Q/Attack/Release: logarithmic mapping
"""

import logging
import math
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Band type raw_value encoding
# ---------------------------------------------------------------------------
BAND_TYPE_LOW_SHELF = 0.0
BAND_TYPE_BELL = 0.5
BAND_TYPE_HIGH_SHELF = 1.0

# ---------------------------------------------------------------------------
# Dynamics mode raw_value encoding
# ---------------------------------------------------------------------------
DYN_OFF = 0.0
DYN_ON = 0.5
DYN_STICKY = 1.0

# ---------------------------------------------------------------------------
# Quality raw_value encoding
# ---------------------------------------------------------------------------
QUALITY_ECONOMY = 0.0
QUALITY_STANDARD = 0.5
QUALITY_PRECISE = 1.0

# ---------------------------------------------------------------------------
# Filter slope raw_value encoding (HP/LP type)
# ---------------------------------------------------------------------------
SLOPE_6DB = 0.0
SLOPE_12DB = 0.333333
SLOPE_24DB = 0.666667
SLOPE_48DB = 1.0


# ---------------------------------------------------------------------------
# Value conversion utilities
# ---------------------------------------------------------------------------
def _linear_to_raw(value: float, min_val: float, max_val: float) -> float:
    """Convert linear value to normalized raw_value (0-1)."""
    if max_val == min_val:
        return 0.5
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


def _log_to_raw(value: float, min_val: float, max_val: float) -> float:
    """Convert logarithmic value to normalized raw_value (0-1)."""
    if value <= 0 or min_val <= 0 or max_val <= min_val:
        return 0.0
    return max(0.0, min(1.0, math.log(value / min_val) / math.log(max_val / min_val)))


def _raw_to_linear(raw: float, min_val: float, max_val: float) -> float:
    """Convert normalized raw_value to linear value."""
    return min_val + raw * (max_val - min_val)


def _raw_to_log(raw: float, min_val: float, max_val: float) -> float:
    """Convert normalized raw_value to logarithmic value."""
    return min_val * (max_val / min_val) ** raw


# ---------------------------------------------------------------------------
# TDR Nova parameter ranges (from discovery)
# ---------------------------------------------------------------------------
GAIN_RANGE = (-18.0, 18.0)
THRESHOLD_RANGE = (-50.0, 0.0)
FREQ_RANGE = (10.0, 40000.0)
Q_RANGE = (0.1, 6.0)
ATTACK_RANGE = (0.1, 500.0)
RELEASE_RANGE = (10.0, 3000.0)
OUTPUT_GAIN_RANGE = (-20.0, 20.0)
RATIO_RANGE = (0.5, 100.0)  # Practical upper limit


def _gain_raw(db: float) -> float:
    return _linear_to_raw(db, *GAIN_RANGE)


def _threshold_raw(db: float) -> float:
    return _linear_to_raw(db, *THRESHOLD_RANGE)


def _freq_raw(hz: float) -> float:
    return _log_to_raw(hz, *FREQ_RANGE)


def _q_raw(q: float) -> float:
    return _log_to_raw(q, *Q_RANGE)


def _attack_raw(ms: float) -> float:
    return _log_to_raw(ms, *ATTACK_RANGE)


def _release_raw(ms: float) -> float:
    return _log_to_raw(ms, *RELEASE_RANGE)


def _output_gain_raw(db: float) -> float:
    return _linear_to_raw(db, *OUTPUT_GAIN_RANGE)


def _ratio_raw(ratio: float) -> float:
    """Ratio uses a special mapping — approximate with log scaling."""
    return _log_to_raw(min(ratio, 100.0), *RATIO_RANGE)


# ---------------------------------------------------------------------------
# Preset builder for TTS speech processing
# ---------------------------------------------------------------------------
def build_tts_preset(
    de_ess_intensity: float = 0.5,
    eq_intensity: float = 1.0,
) -> Dict[str, float]:
    """
    Build TDR Nova parameter dict optimized for TTS speech output.

    Band allocation:
        - Band 1 (Bell): Warmth boost at ~200Hz (+1.5dB scaled by eq_intensity)
        - Band 2 (Bell): Boxiness cut at ~400Hz (-2dB scaled by eq_intensity)
        - Band 3 (Bell): Presence boost at ~3kHz (+2dB scaled by eq_intensity)
        - Band 4 (High Shelf): De-esser at ~6.5kHz with dynamic compression
          (static gain cut + dynamic threshold, scaled by de_ess_intensity)
        - HP: 80Hz high-pass to remove rumble

    Args:
        de_ess_intensity: De-essing strength (0.0-1.0).
        eq_intensity: EQ processing strength (0.0-1.0).

    Returns:
        Dict mapping pedalboard parameter names to raw_value floats.
    """
    params = {}

    # ---- Master settings ----
    params['master_bypass'] = 0.0  # Off = processing active
    params['quality'] = QUALITY_PRECISE
    params['eq_auto_gain'] = 1.0  # On — auto level compensation
    params['output_gain_db'] = _output_gain_raw(0.0)
    params['dry_mix'] = 0.0  # 0% dry

    # ---- HP filter: 80Hz to remove rumble ----
    params['hp_active'] = 1.0  # On
    params['hp_frequency_hz'] = _freq_raw(80.0)
    params['hp_type'] = SLOPE_24DB

    # ---- LP filter: off (let high frequencies through for de-esser) ----
    params['lp_active'] = 0.0

    # ---- Band 1: Warmth boost at 200Hz ----
    warmth_gain = 1.5 * eq_intensity  # dB
    params['band_1_active'] = 1.0 if abs(warmth_gain) > 0.05 else 0.0
    params['band_1_type'] = BAND_TYPE_BELL
    params['band_1_frequency_hz'] = _freq_raw(200.0)
    params['band_1_gain_db'] = _gain_raw(warmth_gain)
    params['band_1_q'] = _q_raw(0.7)
    params['band_1_dyn'] = DYN_OFF  # Static EQ only

    # ---- Band 2: Boxiness cut at 400Hz ----
    boxy_gain = -2.0 * eq_intensity  # dB (cut)
    params['band_2_active'] = 1.0 if abs(boxy_gain) > 0.05 else 0.0
    params['band_2_type'] = BAND_TYPE_BELL
    params['band_2_frequency_hz'] = _freq_raw(400.0)
    params['band_2_gain_db'] = _gain_raw(boxy_gain)
    params['band_2_q'] = _q_raw(1.0)
    params['band_2_dyn'] = DYN_OFF

    # ---- Band 3: Presence boost at 3kHz ----
    presence_gain = 2.0 * eq_intensity  # dB
    params['band_3_active'] = 1.0 if abs(presence_gain) > 0.05 else 0.0
    params['band_3_type'] = BAND_TYPE_BELL
    params['band_3_frequency_hz'] = _freq_raw(3000.0)
    params['band_3_gain_db'] = _gain_raw(presence_gain)
    params['band_3_q'] = _q_raw(1.0)
    params['band_3_dyn'] = DYN_OFF

    # ---- Band 4: De-esser at 6.5kHz (High Shelf + Dynamic) ----
    if de_ess_intensity > 0.01:
        de_ess_gain = -3.0 * de_ess_intensity  # Static cut
        # Dynamic threshold: lower = more aggressive
        # At intensity=0.0 → threshold=0dB (no dynamic), at 1.0 → -20dB (aggressive)
        threshold_db = -20.0 * de_ess_intensity
        ratio = 2.0 + 3.0 * de_ess_intensity  # 2:1 to 5:1

        params['band_4_active'] = 1.0
        params['band_4_type'] = BAND_TYPE_HIGH_SHELF
        params['band_4_frequency_hz'] = _freq_raw(6500.0)
        params['band_4_gain_db'] = _gain_raw(de_ess_gain)
        params['band_4_q'] = _q_raw(1.0)
        params['band_4_dyn'] = DYN_ON
        params['band_4_threshold_db'] = _threshold_raw(threshold_db)
        params['band_4_ratio'] = _ratio_raw(ratio)
        params['band_4_attack_ms'] = _attack_raw(1.0)  # Fast for sibilance
        params['band_4_release_ms'] = _release_raw(50.0)  # Quick release
    else:
        params['band_4_active'] = 0.0

    # ---- Wide band: off ----
    params['wide_band_dyn'] = DYN_OFF

    return params


def apply_preset(plugin, params: Dict[str, float]) -> Dict[str, str]:
    """
    Apply a parameter preset to a loaded TDR Nova plugin.

    Args:
        plugin: A pedalboard VST3 plugin instance.
        params: Dict of parameter_name -> raw_value.

    Returns:
        Dict of {param_name: error_message} for any params that failed to set.
    """
    errors = {}
    for name, value in params.items():
        if name not in plugin.parameters:
            errors[name] = "parameter not found"
            continue
        try:
            plugin.parameters[name].raw_value = value
        except Exception as e:
            errors[name] = str(e)

    if errors:
        logger.warning(f"TDR Nova preset errors: {errors}")
    return errors


def discover_parameters(plugin) -> Dict[str, dict]:
    """
    Discover all TDR Nova parameters via pedalboard introspection.

    Development/debugging utility — prints parameter details.

    Args:
        plugin: A loaded pedalboard VST3 plugin instance.

    Returns:
        Dict of parameter_name -> {value, str_repr}.
    """
    result = {}
    for name, param in plugin.parameters.items():
        result[name] = {
            'raw_value': param.raw_value,
            'str': str(param),
        }
    return result
