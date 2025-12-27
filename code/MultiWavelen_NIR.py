"""
Multi-Wavelength NIR Spectroscopy Pipeline 
=====================================================
VITARON NIR module - LOADS FROM EXISTING CSV .
Processes 4-wavelength reflectance → 7 features for fusion.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple
from scipy.stats import linregress

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class NIRSignal:
    """Raw 4-wavelength photodiode voltages"""
    voltages: np.ndarray        # [V_750, V_810, V_940, V_1600]
    wavelengths_nm: np.ndarray  # [750, 810, 940, 1600]
    metadata: Dict = None

@dataclass
class NIRFeatures:
    """7 features for fusion layer"""
    absorbance_750: float
    absorbance_810: float
    absorbance_940: float
    absorbance_1600: float
    ratio_1600_to_940: float
    absorption_slope: float
    water_band_index: float
    debug: Dict = None

# =============================================================================
# UTILITY FUNCTIONS 
# =============================================================================

def load_nir_csv(path: str) -> NIRSignal:
    #Load NIR voltages from CSV: voltage_750, voltage_810, voltage_940, voltage_1600
    df = pd.read_csv(path)
    voltages = df[["voltage_750", "voltage_810", "voltage_940", "voltage_1600"]].to_numpy()[0]
    return NIRSignal(voltages, np.array([750, 810, 940, 1600]), metadata={"source": "csv"})

def dark_subtract_nir(signal: NIRSignal, dark_current: np.ndarray) -> NIRSignal:
    #Subtract dark current (ambient + electronics)
    clean_voltages = np.maximum(signal.voltages - dark_current, 0.01)
    return NIRSignal(clean_voltages, signal.wavelengths_nm, signal.metadata)

def convert_to_absorbance(signal: NIRSignal, reference_signal: NIRSignal) -> np.ndarray:
    #A(λ) = -log(I(λ)/I0(λ)) - reference-normalized absorbance
    ratio = signal.voltages / reference_signal.voltages
    absorbance = -np.log(np.clip(ratio, 0.01, 2.0))
    return absorbance

# =============================================================================
# MAIN PROCESSING 
# =============================================================================

def preprocess_single_nir_frame(signal: NIRSignal, dark_current: np.ndarray,  water_reference: np.ndarray) -> Tuple[NIRSignal, Dict]:
    #Single frame NIR preprocessing
    # 1. Dark subtraction
    clean_signal = dark_subtract_nir(signal, dark_current)
    
    # 2. Convert to absorbance (like baseline correction)
    absorbance = convert_to_absorbance(clean_signal, NIRSignal(water_reference, signal.wavelengths_nm))
    processed_signal = NIRSignal(absorbance, signal.wavelengths_nm, signal.metadata)
    
    debug = {
        "raw_voltages": signal.voltages.copy(),
        "clean_voltages": clean_signal.voltages.copy(),
        "absorbance_raw": absorbance.copy()
    }
    
    return processed_signal, debug

def accumulate_nir_frames(frames: List[NIRSignal]) -> Tuple[NIRSignal, float]:
    if len(frames) == 0:
        raise ValueError("No frames provided for accumulation.")

    voltages = np.stack([f.voltages for f in frames], axis=0)
    mean_voltages = np.mean(voltages, axis=0)
    frame_var = float(np.mean(np.var(voltages, axis=0)))

    acc_signal = NIRSignal(
        voltages=mean_voltages,
        wavelengths_nm=frames[0].wavelengths_nm.copy(),
        metadata={"n_frames": len(frames)},
    )
    return acc_signal, frame_var

def extract_nir_features(signal: NIRSignal, debug_info: Dict | None = None) -> NIRFeatures:
    #Extract NIR features 
    abs_vals = signal.voltages
    
    # Individual absorbances
    abs_750 = float(abs_vals[0])
    abs_810 = float(abs_vals[1])
    abs_940 = float(abs_vals[2])
    abs_1600 = float(abs_vals[3])
    
    # Ratios
    ratio_1600_to_940 = float(abs_1600 / (abs_940 + 1e-6))
    
    # Slope
    slope, _, _, _, _ = linregress(signal.wavelengths_nm, abs_vals)
    absorption_slope = float(slope)
    
    # Water band
    water_band_index = float(abs_940 - 0.5*(abs_750 + abs_810))
    
    frame_var = float(debug_info.get("frame_variance", 0.0)) if debug_info is not None else 0.0

    features = NIRFeatures(
        absorbance_750=abs_750, absorbance_810=abs_810,
        absorbance_940=abs_940, absorbance_1600=abs_1600,
        ratio_1600_to_940=ratio_1600_to_940,
        absorption_slope=absorption_slope,
        water_band_index=water_band_index,
        debug={"frame_variance": frame_var}
    )
    return features

# =============================================================================
# SIMULATION & MAIN PIPELINE
# =============================================================================

def simulate_multi_frame_from_single(spec: NIRSignal, n_frames: int = 10, noise_scale: float = 0.015) -> List[NIRSignal]:
    frames = []
    for i in range(n_frames):
        noisy_voltages = spec.voltages + np.random.normal(0.0, noise_scale, size=spec.voltages.shape)
        frames.append(NIRSignal(np.clip(noisy_voltages, 0.1, 4.0), spec.wavelengths_nm))
    return frames

def nir_pipeline_from_csv(path: str) -> NIRFeatures:
    # Load base spectrum
    base_spec = load_nir_csv(path)

    # Fixed calibration values
    dark_current = np.array([0.02, 0.02, 0.01, 0.03])
    water_reference = np.array([0.15, 0.22, 0.45, 0.88])

    # Simulate multiple frames from this base spectrum
    raw_frames = simulate_multi_frame_from_single(base_spec, n_frames=10, noise_scale=0.015)

    # Preprocess each frame
    preprocessed_frames = []
    for f in raw_frames:
        cleaned, dbg = preprocess_single_nir_frame(f, dark_current, water_reference)
        preprocessed_frames.append(cleaned)

    # Accumulate frames
    acc_spec, frame_var = accumulate_nir_frames(preprocessed_frames)

    # Build debug info
    acc_cleaned, acc_dbg = preprocess_single_nir_frame(acc_spec, dark_current, water_reference)
    acc_dbg["frame_variance"] = frame_var

    # Extract features
    features = extract_nir_features(acc_cleaned, debug_info=acc_dbg)
    return features

# =============================================================================
# MAIN EXECUTION 
# =============================================================================

if __name__ == "__main__":
    csv_path = "synthetic_nir_voltages.csv"
    nfeats = nir_pipeline_from_csv(csv_path)
    
    nfeature_vector = [
        nfeats.absorbance_750,
        nfeats.absorbance_810,
        nfeats.absorbance_940,
        nfeats.absorbance_1600,
        nfeats.ratio_1600_to_940,
        nfeats.absorption_slope,
        nfeats.water_band_index,
    ]
    print(f"NIR Feature Vector: {feature_vector}")
    
    print("Multi-Wavelength NIR Feature Vector:")
    print(f"Abs 750nm: {nfeats.absorbance_750:.3f}")
    print(f"Abs 810nm: {nfeats.absorbance_810:.3f}")
    print(f"Abs 940nm: {nfeats.absorbance_940:.3f}")
    print(f"Abs 1600nm: {nfeats.absorbance_1600:.3f}")
    print(f"Ratio 1600/940: {nfeats.ratio_1600_to_940:.3f}")
    print(f"Slope: {nfeats.absorption_slope:.4f}")
    print(f"Water index: {nfeats.water_band_index:.3f}")