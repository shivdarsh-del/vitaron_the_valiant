"""
TIMEGATED-RAMAN PIPELINE
Stages:
1. Load raw spectrum (from CSV or hardware)
2. Baseline correction (ALS)
3. Denoising (Savitzky–Golay)
4. Frame accumulation (multi-frame SNR improvement)
5. Peak region extraction around glucose bands
6. Feature computation (intensities, areas, ratios, quality metrics)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple

from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve

from scipy.sparse import csc_matrix  


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------

@dataclass
class RamanSpectrum:
    shift: np.ndarray          # Raman shift axis (cm^-1)
    intensity: np.ndarray      # Measured intensity
    metadata: Dict = None      # Optional info (exposure time, power, etc.)

@dataclass
class RamanFeatures:
    # Core glucose-related features
    peak_intensity_1125: float
    peak_intensity_1340: float
    peak_intensity_1460: float

    area_1125: float
    area_1340: float
    area_1460: float

    ratio_1125_to_1340: float
    ratio_1340_to_1460: float

    # Quality / context features
    snr_estimate: float
    fluorescence_index: float
    frame_variance: float

    # Optional debug information (not for fusion model input)
    debug: Dict = None


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------

def load_raman_csv(path: str) -> RamanSpectrum:
    """
    Load Raman spectrum from a CSV file with columns:
    - raman_shift_cm^-1
    - intensity_raw
    """
    df = pd.read_csv(path)
    shift = df["raman_shift_cm^-1"].to_numpy()
    intensity = df["intensity_raw"].to_numpy()
    return RamanSpectrum(shift=shift, intensity=intensity, metadata={"source": "csv"})

#Asymmetric Least Squares (ALS) baseline correction.

def asymmetric_least_squares_baseline(y: np.ndarray, lam: float = 1e5, p: float = 0.001, niter: int = 10) -> np.ndarray:
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    D = lam * D.dot(D.T)  # D is CSC
    
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.diags(w, 0)
        # *** KEY FIX: Convert Z to CSC explicitly ***
        Z = csc_matrix(W + D)
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

def compute_snr(signal: np.ndarray, region_mask: np.ndarray) -> float:
    """
    Estimate SNR as mean(signal in region) / std(noise in non-region).
    """
    if not np.any(region_mask):
        return 0.0

    signal_region = signal[region_mask]
    noise_region = signal[~region_mask]

    signal_level = np.mean(signal_region)
    noise_std = np.std(noise_region) if noise_region.size > 0 else 1e-9

    return float(signal_level / noise_std)


def integrate_region(shift: np.ndarray, intensity: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    # CHANGE np.trapz → np.trapezoid
    return float(np.trapezoid(intensity[mask], shift[mask]))


def peak_max_in_window(
    shift: np.ndarray,
    intensity: np.ndarray,
    center: float,
    half_width: float = 20.0,
) -> float:
    """
    Return the maximum intensity in a window [center-half_width, center+half_width].
    """
    mask = (shift >= center - half_width) & (shift <= center + half_width)
    if not np.any(mask):
        return 0.0
    return float(np.max(intensity[mask]))


# ----------------------------------------------------------------------
# Main processing functions
# ----------------------------------------------------------------------
def apply_time_gating(intensity, gate_width=0.3):
    """
    Simulate time-gated detection by suppressing delayed fluorescence.
    gate_width: fraction of signal kept as 'early photons' (0.1-0.4 typical)
    """
    threshold = np.quantile(intensity, 1 - gate_width)
    gated = np.minimum(intensity, threshold)
    return gated

def preprocess_single_spectrum(spec: RamanSpectrum, als_lam: float = 1e5, als_p: float = 0.001, sg_window: int = 15, sg_poly: int = 3, gate_width: float = 0.3) -> Tuple[RamanSpectrum, Dict]:
    """
    Complete Raman preprocessing with TIME-GATING:
    1. TIME-GATE (reject fluorescence) ← NEW
    2. Baseline correction (ALS)
    3. Denoising (Savitzky-Golay)
    """
    y = spec.intensity.copy()
    
    # 1. TIME-GATING: Simulate hardware rejecting delayed fluorescence photons
    y_gated = apply_time_gating(y, gate_width=gate_width)
    
    # 2. Baseline correction on gated signal
    baseline = asymmetric_least_squares_baseline(y_gated, lam=als_lam, p=als_p)
    y_corrected = y_gated - baseline
    
    # 3. Denoising
    if sg_window >= len(y):
        sg_window = len(y) - 1 if (len(y) - 1) % 2 == 1 else len(y) - 2
    if sg_window < 5: sg_window = 5
    if sg_window % 2 == 0: sg_window += 1
    y_denoised = savgol_filter(y_corrected, window_length=sg_window, polyorder=sg_poly)
    
    cleaned_spec = RamanSpectrum(shift=spec.shift.copy(), intensity=y_denoised, metadata=spec.metadata)
    
    debug = {
        "baseline": baseline,
        "gated_intensity": y_gated,      # ← NEW debug info
        "corrected_raw": y_corrected,
        "denoised": y_denoised,
        "time_gating_threshold": float(np.quantile(spec.intensity, 1 - gate_width))
    }
    
    return cleaned_spec, debug


def accumulate_frames(
    frames: List[RamanSpectrum],
) -> Tuple[RamanSpectrum, float]:
    """
    Average multiple preprocessed spectra to boost SNR.



    Returns:
        accumulated_spectrum, frame_variance
    """
    if len(frames) == 0:
        raise ValueError("No frames provided for accumulation.")

    shifts = np.array([np.array(f.shift) for f in frames])
    if not np.allclose(shifts, shifts[0]):
        raise ValueError("All frames must share the same Raman shift axis.")

    intensities = np.stack([f.intensity for f in frames], axis=0)
    mean_intensity = np.mean(intensities, axis=0)
    frame_var = float(np.mean(np.var(intensities, axis=0)))

    acc_spec = RamanSpectrum(
        shift=frames[0].shift.copy(),
        intensity=mean_intensity,
        metadata={"n_frames": len(frames)},
    )
    return acc_spec, frame_var


def extract_glucose_features(
    spec: RamanSpectrum,
    debug_info: Dict | None = None,
) -> RamanFeatures:
    """
    Extract glucose-related spectral features and quality metrics
    from a preprocessed, optionally accumulated Raman spectrum.
    """
    shift = spec.shift
    intensity = spec.intensity

    # Define glucose-sensitive windows (can be tuned later)
    windows = {
        "w_1125": (1125 - 20, 1125 + 20),
        "w_1340": (1340 - 20, 1340 + 20),
        "w_1460": (1460 - 20, 1460 + 20),
    }

    masks = {
        name: (shift >= lo) & (shift <= hi)
        for name, (lo, hi) in windows.items()
    }

    # Peak intensities
    peak_1125 = peak_max_in_window(shift, intensity, center=1125, half_width=20)
    peak_1340 = peak_max_in_window(shift, intensity, center=1340, half_width=20)
    peak_1460 = peak_max_in_window(shift, intensity, center=1460, half_width=20)

    # Areas
    area_1125 = integrate_region(shift, intensity, masks["w_1125"])
    area_1340 = integrate_region(shift, intensity, masks["w_1340"])
    area_1460 = integrate_region(shift, intensity, masks["w_1460"])

    # Ratios (with small epsilon to avoid division by zero)
    eps = 1e-9
    ratio_1125_to_1340 = peak_1125 / (peak_1340 + eps)
    ratio_1340_to_1460 = peak_1340 / (peak_1460 + eps)

    # SNR: define "signal region" as union of glucose windows
    signal_mask = masks["w_1125"] | masks["w_1340"] | masks["w_1460"]
    snr = compute_snr(intensity, signal_mask)

    # Fluorescence index:
    # If baseline info available in debug, use residual baseline energy;
    # otherwise approximate as low-frequency content via heavy smoothing.
    if debug_info is not None and "baseline" in debug_info:
        baseline = debug_info["baseline"]
        # Use RMS of baseline as fluorescence proxy
        fluorescence_index = float(np.sqrt(np.mean(baseline**2)))
    else:
        # Backup: smooth heavily and treat that as fluorescence/background
        smooth_bg = savgol_filter(intensity, window_length=101, polyorder=2)
        fluorescence_index = float(np.sqrt(np.mean(smooth_bg**2)))

    # Frame variance is typically computed during accumulation; put 0.0 by default
    frame_var = float(debug_info.get("frame_variance", 0.0)) if debug_info is not None else 0.0

    features = RamanFeatures(
        peak_intensity_1125=peak_1125,
        peak_intensity_1340=peak_1340,
        peak_intensity_1460=peak_1460,
        area_1125=area_1125,
        area_1340=area_1340,
        area_1460=area_1460,
        ratio_1125_to_1340=ratio_1125_to_1340,
        ratio_1340_to_1460=ratio_1340_to_1460,
        snr_estimate=snr,
        fluorescence_index=fluorescence_index,
        frame_variance=frame_var,
        debug=debug_info,
    )
    return features


# ----------------------------------------------------------------------
# Example end-to-end usage
# ----------------------------------------------------------------------

def simulate_multi_frame_from_single(
    spec: RamanSpectrum,
    n_frames: int = 8,
    noise_scale: float = 0.01,
) -> List[RamanSpectrum]:
    """
    Simulate multiple Raman frames from a single base spectrum
    by adding small random noise per frame.
    """
    frames = []
    for i in range(n_frames):
        noisy_intensity = spec.intensity + np.random.normal(
            0.0, noise_scale, size=spec.intensity.shape
        )
        frames.append(RamanSpectrum(shift=spec.shift, intensity=noisy_intensity))
    return frames


def raman_pipeline_from_csv(path: str) -> RamanFeatures:
    """
    High-level entry point:
    1) Load CSV
    2) Simulate multiple frames
    3) Preprocess each frame (baseline + denoise)
    4) Accumulate frames
    5) Extract glucose-related features
    """
    # Load base spectrum
    base_spec = load_raman_csv(path)

    # Simulate multiple frames from this base spectrum
    raw_frames = simulate_multi_frame_from_single(base_spec, n_frames=8, noise_scale=0.01)

    # Preprocess each frame
    preprocessed_frames = []
    for f in raw_frames:
        cleaned, dbg = preprocess_single_spectrum(f)
        preprocessed_frames.append(cleaned)

    # Accumulate frames
    acc_spec, frame_var = accumulate_frames(preprocessed_frames)

    # Build debug info
    # Re-run preprocessing on accumulated spectrum just to get baseline info cleanly
    acc_cleaned, acc_dbg = preprocess_single_spectrum(acc_spec)
    acc_dbg["frame_variance"] = frame_var

    # Extract features
    features = extract_glucose_features(acc_cleaned, debug_info=acc_dbg)
    return features


if __name__ == "__main__":
    # Example usage with the synthetic CSV
    csv_path = "synthetic_glucose_raman_spectrum.csv"
    rfeats = raman_pipeline_from_csv(csv_path)
    rfeature_vector = [
        rfeats.peak_intensity_1125,
        rfeats.peak_intensity_1340,
        rfeats.peak_intensity_1460,
        rfeats.area_1125,
        rfeats.area_1340,
        rfeats.area_1460,
        rfeats.ratio_1125_to_1340,
        rfeats.ratio_1340_to_1460,
        rfeats.snr_estimate,
        rfeats.fluorescence_index,
        rfeats.frame_variance,
    ]
    print(f"Raman Vector: {rfeature_vector}")
    print("Time-gated RAMAN Feature Vector:")
    print(f"Peak 1125: {rfeats.peak_intensity_1125:.3f}")
    print(f"Peak 1340: {rfeats.peak_intensity_1340:.3f}")
    print(f"Peak 1460: {rfeats.peak_intensity_1460:.3f}")
    print(f"Area 1125: {rfeats.area_1125:.3f}")
    print(f"Area 1340: {rfeats.area_1340:.3f}")
    print(f"Area 1460: {rfeats.area_1460:.3f}")
    print(f"Ratio 1125/1340: {rfeats.ratio_1125_to_1340:.3f}")
    print(f"Ratio 1340/1460: {rfeats.ratio_1340_to_1460:.3f}")
    print(f"SNR: {rfeats.snr_estimate:.3f}")
    print(f"Fluorescence: {rfeats.fluorescence_index:.3f}")
    print(f"Frame variance: {rfeats.frame_variance:.3f}")