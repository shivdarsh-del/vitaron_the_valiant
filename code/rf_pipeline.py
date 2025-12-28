"""
RF Dielectric Sensing Pipeline 
=========================================
Stages:
1. Load S11 sweep (from CSV or VNA hardware)
2. Resonance peak detection
3. Smoothing + phase unwrapping
4. Frame accumulation (multi-sweep averaging)
5. Feature extraction (permittivity, phase, hydration, thickness)
6. Quality metrics computation

"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple
from scipy.signal import find_peaks

# =============================================================================
# DATA STRUCTURES 
# =============================================================================

@dataclass
class RFSweep:
    #Raw S11 frequency sweep data
    frequency_mhz: np.ndarray    # 100-500 MHz sweep
    s11_magnitude_db: np.ndarray # Reflection coefficient (dB)
    s11_phase_deg: np.ndarray    # Phase (degrees)
    metadata: Dict = None

@dataclass
class RFFeatures:
    effective_permittivity: float
    phase_delay: float
    hydration_index: float
    skin_thickness_estimate: float
    debug: Dict = None

# =============================================================================
# UTILITY FUNCTIONS 
# =============================================================================

def load_rf_csv(path: str) -> RFSweep:
    #Load S11 sweep from CSV: frequency_mhz, s11_magnitude_db, s11_phase_deg
    df = pd.read_csv(path)
    return RFSweep(
        frequency_mhz=df["frequency_mhz"].to_numpy(),
        s11_magnitude_db=df["s11_magnitude_db"].to_numpy(),
        s11_phase_deg=df["s11_phase_deg"].to_numpy(),
        metadata={"source": "csv"}
    )

def find_resonance_peak(sweep: RFSweep) -> Tuple[float, float]:
    #Finding resonance frequency (S11 minimum) and Q-factor
    peaks, _ = find_peaks(-sweep.s11_magnitude_db, height=-25, distance=10)
    if len(peaks) == 0:
        return 300.0, 10.0
    
    f_res = sweep.frequency_mhz[peaks[0]]
    half_power_idx = np.where(sweep.s11_magnitude_db > -3)[0]
    q_factor = f_res / (sweep.frequency_mhz[half_power_idx[-1]] - sweep.frequency_mhz[half_power_idx[0]]) if len(half_power_idx) > 1 else 10
    return float(f_res), float(q_factor)

def preprocess_rf_sweep(sweep: RFSweep) -> Tuple[RFSweep, Dict]:
    #Denoise + unwrap phase 
    # Smooth magnitude
    window = min(21, len(sweep.s11_magnitude_db)//10 * 2 + 1)
    if window % 2 == 0: window += 1
    s11_smooth = np.convolve(sweep.s11_magnitude_db, np.ones(window)/window, mode='same')
    
    # Unwrap phase
    phase_unwrapped = np.unwrap(sweep.s11_phase_deg * np.pi/180) * 180/np.pi
    
    cleaned = RFSweep(
        sweep.frequency_mhz.copy(),
        s11_smooth,
        phase_unwrapped,
        sweep.metadata
    )
    
    return cleaned, {"raw_resonance": find_resonance_peak(sweep)}

# =============================================================================
# MAIN PROCESSING FUNCTIONS 
# =============================================================================

def extract_rf_features(sweep: RFSweep, debug_info: Dict | None = None) -> RFFeatures:
    # Resonance analysis
    f_res, q_factor = find_resonance_peak(sweep)
    
    # Effective permittivity
    effective_permittivity = float(2500 / (f_res**2) * 1e6 + 30)
    
    # Phase delay at resonance
    res_idx = np.argmin(np.abs(sweep.frequency_mhz - f_res))
    phase_delay = float(np.abs(sweep.s11_phase_deg[res_idx]))
    
    # Hydration index
    hydration_index = float(np.clip((effective_permittivity - 40) / 40, 0.0, 1.0))
    
    # Skin thickness
    skin_thickness = float(phase_delay / 12.5)
    
    frame_var = float(debug_info.get("frame_variance", 0.0)) if debug_info is not None else 0.0

    features = RFFeatures(
        effective_permittivity=effective_permittivity,
        phase_delay=phase_delay,
        hydration_index=hydration_index,
        skin_thickness_estimate=skin_thickness,
        debug={
            "resonance_mhz": f_res,
            "q_factor": q_factor,
            "frame_variance": frame_var
        }
    )
    return features

def accumulate_rf_frames(frames: List[RFSweep]) -> Tuple[RFSweep, float]:
    if len(frames) == 0:
        raise ValueError("No frames provided for accumulation.")

    freqs = np.array([f.frequency_mhz for f in frames])
    if not np.allclose(freqs, freqs[0], rtol=1e-3):
        raise ValueError("All frames must share identical frequency axis.")

    mags = np.stack([f.s11_magnitude_db for f in frames], axis=0)
    phases = np.stack([f.s11_phase_deg for f in frames], axis=0)
    
    mean_mag = np.mean(mags, axis=0)
    mean_phase = np.mean(phases, axis=0)
    frame_var = float(np.mean(np.var(mags, axis=0)))
    
    acc_sweep = RFSweep(
        frequency_mhz=frames[0].frequency_mhz.copy(),
        s11_magnitude_db=mean_mag,
        s11_phase_deg=mean_phase,
        metadata={"n_frames": len(frames)}
    )
    return acc_sweep, frame_var

# =============================================================================
# SIMULATION & MAIN PIPELINE
# =============================================================================

def simulate_multi_frame_from_single(spec: RFSweep, n_frames: int = 8, noise_scale: float = 0.5) -> List[RFSweep]:
    frames = []
    for i in range(n_frames):
        noisy_mag = spec.s11_magnitude_db + np.random.normal(0, noise_scale, spec.s11_magnitude_db.shape)
        noisy_phase = spec.s11_phase_deg + np.random.normal(0, 2, spec.s11_phase_deg.shape)
        frames.append(RFSweep(spec.frequency_mhz, noisy_mag, noisy_phase))
    return frames

def rf_pipeline_from_csv(path: str) -> RFFeatures:
    # Load base sweep
    base_spec = load_rf_csv(path)

    # Simulate multiple frames from this base sweep
    raw_frames = simulate_multi_frame_from_single(base_spec, n_frames=8, noise_scale=0.5)

    # Preprocess each frame
    preprocessed_frames = []
    for f in raw_frames:
        cleaned, dbg = preprocess_rf_sweep(f)
        preprocessed_frames.append(cleaned)

    # Accumulate frames
    acc_spec, frame_var = accumulate_rf_frames(preprocessed_frames)

    # Build debug info
    acc_cleaned, acc_dbg = preprocess_rf_sweep(acc_spec)
    acc_dbg["frame_variance"] = frame_var

    # Extract features
    features = extract_rf_features(acc_cleaned, debug_info=acc_dbg)
    return features

# =============================================================================
# MAIN EXECUTION 
# =============================================================================

if __name__ == "__main__":
    # Example usage with the CSV (EXACTLY like Raman)
    csv_path = "synthetic_rf_sweep.csv"
    rf_feats = rf_pipeline_from_csv(csv_path)
    
    rf_feature_vector = [
        rf_feats.effective_permittivity,
        rf_feats.phase_delay,
        rf_feats.hydration_index,
        rf_feats.skin_thickness_estimate,
    ]
    print(f"RF Vector: {rf_feature_vector}")
    
    print("RF Dielectric Sensing Feature Vector:")
    print(f"Permittivity: {rf_feats.effective_permittivity:.1f}")
    print(f"Phase delay: {rf_feats.phase_delay:.1f}°")
    print(f"Hydration index: {rf_feats.hydration_index:.3f}")
    print(f"Skin thickness: {rf_feats.skin_thickness_estimate:.1f}mm")