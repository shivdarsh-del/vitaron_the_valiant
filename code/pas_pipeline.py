"""
PAS (Photoacoustic Spectroscopy) Processing Pipeline
==================================================
Stages:
1. Raw time-domain waveform → feature extraction (4 features)
2. Baseline subtraction + early-window isolation
3. Melanin index computation
4. Features ready for fusion with Raman/NIR/PPG/RF

"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from scipy.signal import find_peaks, savgol_filter

# =============================================================================
# DATA STRUCTURES 
# =============================================================================

@dataclass
class PASWaveform:
    #Raw time-domain photoacoustic signal
    time_us: np.ndarray      # Time axis (microseconds)
    voltage: np.ndarray      # Acoustic voltage (V)
    metadata: Dict = None    # Pulse energy, transducer gain, etc.

@dataclass
class PASFeatures:
    early_peak_amplitude: float     # Max amplitude in epidermal window (15-50 µs)
    early_window_energy: float      # Integrated energy in early window
    absorption_consistency: float   # Frame-to-frame consistency (0-1)
    melanin_index: float            # Normalized melanin proxy (0-1)
    debug: Dict = None              # Processing intermediates

# =============================================================================
# UTILITY FUNCTIONS 
# =============================================================================

def load_pas_csv(path: str) -> PASWaveform:
    #Load PAS waveform from CSV
    df = pd.read_csv(path)
    time_us = df["time_us"].to_numpy()
    voltage = df["voltage_raw"].to_numpy()
    return PASWaveform(time_us=time_us, voltage=voltage, metadata={"source": "csv"})

def baseline_subtract_pas(waveform: PASWaveform, pre_trigger_us: float = 15.0) -> Tuple[PASWaveform, float]:
    #Subtract pre-trigger baseline (0-15 µs) from entire waveform
    pre_trigger_samples = waveform.time_us < pre_trigger_us
    baseline_level = float(np.mean(waveform.voltage[pre_trigger_samples]))
    corrected_voltage = waveform.voltage - baseline_level
    corrected_waveform = PASWaveform(
        time_us=waveform.time_us.copy(),
        voltage=corrected_voltage,
        metadata=waveform.metadata
    )
    return corrected_waveform, baseline_level

def isolate_early_window(waveform: PASWaveform,  early_start_us: float = 15.0, 
                         early_end_us: float = 50.0) -> Tuple[np.ndarray, np.ndarray]:
    #Extract epidermal/melanin-sensitive early-time window (15-50 µs)
    early_mask = (waveform.time_us >= early_start_us) & (waveform.time_us <= early_end_us)
    early_time = waveform.time_us[early_mask]
    early_voltage = waveform.voltage[early_mask]
    return early_time, early_voltage

# =============================================================================
# MAIN PROCESSING FUNCTIONS 
# =============================================================================

def preprocess_single_pas_frame(waveform: PASWaveform) -> Tuple[PASWaveform, Dict]:
    # Baseline correction
    corrected_waveform, baseline_level = baseline_subtract_pas(waveform)
    
    # Optional denoising
    if len(corrected_waveform.voltage) > 21:
        denoised_voltage = savgol_filter(corrected_waveform.voltage, window_length=21, polyorder=3)
        corrected_waveform.voltage = denoised_voltage
    
    debug = {
        "baseline_level": baseline_level,
        "corrected_waveform": corrected_waveform.voltage.copy()
    }
    
    return corrected_waveform, debug

def accumulate_pas_frames(frames: List[PASWaveform]) -> Tuple[PASWaveform, float]:
    if len(frames) == 0:
        raise ValueError("No frames provided for accumulation.")

    times = np.array([f.time_us for f in frames])
    if not np.allclose(times, times[0], rtol=1e-3):
        raise ValueError("All frames must share identical time axis.")

    voltages = np.stack([f.voltage for f in frames], axis=0)
    mean_voltage = np.mean(voltages, axis=0)
    frame_var = float(np.mean(np.var(voltages, axis=0)))

    acc_waveform = PASWaveform(
        time_us=frames[0].time_us.copy(),
        voltage=mean_voltage,
        metadata={"n_frames": len(frames)},
    )
    return acc_waveform, frame_var

def extract_pas_features(waveform: PASWaveform, debug_info: Dict | None = None) -> PASFeatures:
    # Isolate early epidermal window
    early_time, early_voltage = isolate_early_window(waveform)
    
    # Early peak amplitude
    early_peak_amplitude = float(np.max(np.abs(early_voltage)))
    
    # Early window energy
    early_window_energy = float(np.sum(early_voltage**2))
    
    # Melanin index
    melanin_index = np.clip(early_peak_amplitude / 0.8, 0.0, 1.0)
    
    # Consistency from frame variance
    frame_var = float(debug_info.get("frame_variance", 0.0)) if debug_info is not None else 0.0
    consistency = float(np.clip(1.0 - frame_var / 0.1, 0.0, 1.0))

    features = PASFeatures(
        early_peak_amplitude=early_peak_amplitude,
        early_window_energy=early_window_energy,
        absorption_consistency=consistency,
        melanin_index=float(melanin_index),
        debug={
            "early_peak_time": float(waveform.time_us[np.argmax(np.abs(early_voltage))]) 
                               if len(early_voltage) > 0 else 0.0,
            "frame_variance": frame_var
        }
    )
    return features

# =============================================================================
# SIMULATION & MAIN PIPELINE (MATCHING RAMAN EXACTLY)
# =============================================================================

def simulate_multi_frame_from_single(spec: PASWaveform, n_frames: int = 8, noise_scale: float = 0.01) -> List[PASWaveform]:
    frames = []
    for i in range(n_frames):
        noisy_voltage = spec.voltage + np.random.normal(0.0, noise_scale, size=spec.voltage.shape)
        frames.append(PASWaveform(time_us=spec.time_us, voltage=noisy_voltage))
    return frames

def pas_pipeline_from_csv(path: str) -> PASFeatures:
    # Load base waveform
    base_spec = load_pas_csv(path)

    # Simulate multiple frames from this base waveform
    raw_frames = simulate_multi_frame_from_single(base_spec, n_frames=8, noise_scale=0.01)

    # Preprocess each frame
    preprocessed_frames = []
    for f in raw_frames:
        cleaned, dbg = preprocess_single_pas_frame(f)
        preprocessed_frames.append(cleaned)

    # Accumulate frames
    acc_spec, frame_var = accumulate_pas_frames(preprocessed_frames)

    # Build debug info
    acc_cleaned, acc_dbg = preprocess_single_pas_frame(acc_spec)
    acc_dbg["frame_variance"] = frame_var

    # Extract features
    features = extract_pas_features(acc_cleaned, debug_info=acc_dbg)
    return features

# =============================================================================
# MAIN EXECUTION 
# =============================================================================

if __name__ == "__main__":
    # Example usage with the CSV (EXACTLY like Raman)
    csv_path = "synthetic_pas_waveform.csv"
    pas_feats = pas_pipeline_from_csv(csv_path)
    
    pas_feature_vector = [
        pas_feats.early_peak_amplitude,
        pas_feats.early_window_energy,
        pas_feats.absorption_consistency,
        pas_feats.melanin_index,
    ]
    print(f"PAS Vector: {pas_feature_vector}")
    
    print("PAS (Photoacoustic) Feature Vector:")
    print(f"Early peak: {pas_feats.early_peak_amplitude:.3f}V")
    print(f"Early energy: {pas_feats.early_window_energy:.3f}")
    print(f"Consistency: {pas_feats.absorption_consistency:.3f}")
    print(f"Melanin index: {pas_feats.melanin_index:.3f}")