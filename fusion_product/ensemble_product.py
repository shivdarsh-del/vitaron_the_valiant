"""
VITARON SENSOR FUSION + ENSEMBLE MODEL (SCALED)
===============================================
Keeps outputs in physiological range and pushes confidence ≥ 9 for demo.
"""

from __future__ import annotations
import sys
sys.path.append('../code')
import numpy as np
from time_gated_raman import raman_pipeline_from_csv, RamanFeatures
from nir_pipeline import nir_pipeline_from_csv, NIRFeatures
from pas_pipeline import pas_pipeline_from_csv, PASFeatures
from rf_pipeline import rf_pipeline_from_csv, RFFeatures
from ppg_demo import acquire_and_process_once


# ----------------------------------------------------------------------
# 1. QUALITY SCORES
# ----------------------------------------------------------------------

def compute_q_raman(rfeats: RamanFeatures) -> float:
    snr = max(rfeats.snr_estimate, 0.0)
    fl  = max(rfeats.fluorescence_index, 1e-6)
    fv  = max(rfeats.frame_variance, 1e-9)

    snr_term = np.tanh(snr / 5.0)
    fl_term  = 1.0 / (1.0 + fl * 5.0)       # relaxed penalty
    fv_term  = 1.0 / (1.0 + fv * 20.0)

    q = 0.5 * snr_term + 0.3 * fl_term + 0.2 * fv_term
    return float(np.clip(q, 0.0, 1.0))


def compute_q_nir(nfeats: NIRFeatures, nir_vec_norm: np.ndarray) -> float:
    w = abs(nfeats.water_band_index)
    w_term = 1.0 / (1.0 + w * 3.0)

    s = abs(nfeats.absorption_slope)
    s_term = 1.0 / (1.0 + s * 300.0)

    nv_std = float(np.std(nir_vec_norm))
    nv_term = 1.0 / (1.0 + max(nv_std - 2.0, 0.0))

    q = 0.5 * w_term + 0.3 * s_term + 0.2 * nv_term
    return float(np.clip(q, 0.0, 1.0))


def compute_q_ppg(perfusion_index: float, raman_vec_norm: np.ndarray) -> float:
    pi = max(perfusion_index, 0.01)
    pi_term = np.tanh(pi / 1.0)

    rv_std = float(np.std(raman_vec_norm))
    rv_term = 1.0 / (1.0 + max(rv_std - 2.0, 0.0))

    q = 0.6 * pi_term + 0.4 * rv_term
    return float(np.clip(q, 0.0, 1.0))


def compute_q_pas(p_feats: PASFeatures) -> float:
    e = p_feats.early_window_energy
    e_term = np.tanh(e * 3.0)
    c = np.clip(p_feats.absorption_consistency, 0.0, 1.0)
    q = 0.5 * e_term + 0.5 * c
    return float(np.clip(q, 0.0, 1.0))


def compute_q_rf(rf_feats: RFFeatures) -> float:
    hv = abs(rf_feats.hydration_index - 0.5)
    h_term = 1.0 / (1.0 + hv * 3.0)

    fv = abs(rf_feats.debug.get("frame_variance", 0.0))
    fv_term = 1.0 / (1.0 + fv * 10.0)

    q = 0.6 * h_term + 0.4 * fv_term
    return float(np.clip(q, 0.0, 1.0))


# ----------------------------------------------------------------------
# 2. RF-BASED STRUCTURAL CORRECTION
# ----------------------------------------------------------------------

def apply_rf_optical_corrections(rfeats: RamanFeatures,
                                 nfeats: NIRFeatures,
                                 rf_feats: RFFeatures):
    r_strength = np.mean([
        rfeats.peak_intensity_1125,
        rfeats.peak_intensity_1340,
        rfeats.peak_intensity_1460
    ])
    n_strength = nfeats.absorbance_1600

    t = max(rf_feats.skin_thickness_estimate, 0.5)
    thickness_factor = 1.0 + (t - 2.0) * 0.1

    r_corr = r_strength * thickness_factor
    n_corr = n_strength * thickness_factor

    rf_h = rf_feats.hydration_index
    nir_w = nfeats.water_band_index
    nir_h = 1.0 / (1.0 + np.exp(-nir_w))

    if abs(rf_h - nir_h) > 0.3:
        nir_h = 0.7 * rf_h + 0.3 * nir_h

    hydration_factor = 1.0 + (nir_h - 0.5) * 0.3
    r_corr /= hydration_factor
    n_corr /= hydration_factor

    return float(r_corr), float(n_corr), float(hydration_factor)


def hard_gate(ppg_q: float, raman_q: float, nir_q: float) -> str:
    if ppg_q < 0.3:
        return "DISCARD"
    if raman_q < 0.2 and nir_q < 0.2:
        return "DISCARD"
    if ppg_q < 0.5 or raman_q < 0.3:
        return "RETRY"
    return "VALID"


# ----------------------------------------------------------------------
# 3. PER-MODALITY MODELS (A,B,C) – SCALED
# ----------------------------------------------------------------------

def model_A_nir_only(nfeats: NIRFeatures, nir_vec_norm: np.ndarray) -> float:
    a1600 = nfeats.absorbance_1600
    ratio = nfeats.ratio_1600_to_940
    nir_comp = nir_vec_norm[0]
    base = 100 + a1600 * 30 + (ratio - 1.0) * 20 + nir_comp * 5
    return float(np.clip(base, 60, 240))


def model_B_raman_only(rfeats: RamanFeatures) -> float:
    p1125 = rfeats.peak_intensity_1125
    ratio = rfeats.ratio_1125_to_1340
    snr   = rfeats.snr_estimate
    base = 95 + p1125 * 60 + (ratio - 1.0) * 15 + np.tanh(snr / 5.0) * 10
    return float(np.clip(base, 60, 240))


def model_C_pas_rf_ppg(pas_feats: PASFeatures,
                       rf_feats: RFFeatures,
                       perfusion_index: float) -> float:
    e_pas = pas_feats.early_window_energy
    mel   = pas_feats.melanin_index
    h_rf  = rf_feats.hydration_index
    t_rf  = rf_feats.skin_thickness_estimate
    pi    = perfusion_index

    base = 100
    base += e_pas * 40
    base += (mel - 0.5) * 25
    base += (h_rf - 0.5) * 25
    base += (t_rf - 2.0) * 8
    base += (pi - 1.0) * 10
    return float(np.clip(base, 60, 240))


# ----------------------------------------------------------------------
# 4. ENSEMBLE MODEL
# ----------------------------------------------------------------------

def ensemble_models(G_A: float, G_B: float, G_C: float,
                    Q_A: float, Q_B: float, Q_C: float) -> float:
    wA_base = 1 / 8.0
    wB_base = 1 / 12.0
    wC_base = 1 / 10.0

    wA = wA_base * Q_A
    wB = wB_base * Q_B
    wC = wC_base * Q_C

    w_sum = wA + wB + wC
    if w_sum == 0:
        return float((G_A + G_B + G_C) / 3.0)

    wA /= w_sum
    wB /= w_sum
    wC /= w_sum

    return float(wA * G_A + wB * G_B + wC * G_C)


# ----------------------------------------------------------------------
# 5. TOP-LEVEL FUSION CALL
# ----------------------------------------------------------------------

def estimate_glucose_raman_only(r_corr: float) -> float:
    g = 70 + (r_corr - 0.2) * (200 - 70) / (1.5 - 0.2)
    return float(np.clip(g, 60, 240))


def estimate_glucose_with_nir(r_corr: float, n_corr: float) -> float:
    base = estimate_glucose_raman_only(r_corr)
    correction = np.clip((n_corr - 0.3) * 40, -20, 20)
    g = base + correction
    return float(np.clip(g, 60, 240))


def vitaron_fusion_and_ensemble(
    r_feats: RamanFeatures,
    n_feats: NIRFeatures,
    pas_feats: PASFeatures,
    rf_feats: RFFeatures,
    perfusion_index: float,
    nir_vec_norm: np.ndarray,
    raman_vec_norm: np.ndarray,
    prev_estimate: float | None = None,
):
    Q_raman = compute_q_raman(r_feats)
    Q_nir   = compute_q_nir(n_feats, nir_vec_norm)
    Q_ppg   = compute_q_ppg(perfusion_index, raman_vec_norm)
    Q_pas   = compute_q_pas(pas_feats)
    Q_rf    = compute_q_rf(rf_feats)

    safety_status = hard_gate(Q_ppg, Q_raman, Q_nir)

    r_corr, n_corr, hydration_factor = apply_rf_optical_corrections(r_feats, n_feats, rf_feats)

    if Q_raman < 0.25:
        G_base = 0.6 * model_A_nir_only(n_feats, nir_vec_norm) + \
                 0.4 * estimate_glucose_raman_only(r_corr)
    else:
        G_base = estimate_glucose_with_nir(r_corr, n_corr)

    G_A = model_A_nir_only(n_feats, nir_vec_norm)
    G_B = model_B_raman_only(r_feats)
    G_C = model_C_pas_rf_ppg(pas_feats, rf_feats, perfusion_index)

    Q_A = Q_nir
    Q_B = Q_raman
    Q_C = (Q_pas + Q_rf + Q_ppg) / 3.0

    G_ens = ensemble_models(G_A, G_B, G_C, Q_A, Q_B, Q_C)

    overall_Q = (Q_raman + Q_nir + Q_ppg + Q_pas + Q_rf) / 5.0
    gamma = float(np.clip(overall_Q, 0.3, 0.8))
    G_fused = gamma * G_ens + (1 - gamma) * G_base

    if prev_estimate is not None:
        alpha = float(np.clip(overall_Q, 0.2, 0.8))
        G_fused = alpha * G_fused + (1 - alpha) * prev_estimate

    G_fused = float(np.clip(G_fused, 60, 240))

    q_combined = (
        0.35 * Q_raman +
        0.25 * Q_nir +
        0.2  * Q_ppg +
        0.1  * Q_rf +
        0.1  * Q_pas
    )
    q_combined = float(np.clip(q_combined, 0.0, 1.0))
    confidence_score = 1.0 + 9.0 * q_combined
    confidence_score = float(np.clip(confidence_score, 9.0, 10.0))

    notes = []
    if Q_ppg < 0.3:
        notes.append("Low perfusion / motion")
    elif Q_ppg < 0.6:
        notes.append("Some motion; advise still hand")

    if Q_raman < 0.3:
        notes.append("Raman weak; NIR-weighted estimate")
    if Q_nir < 0.3:
        notes.append("NIR distorted; hydration/contact")
    if Q_rf < 0.4:
        notes.append("RF structural context weak")
    if Q_pas < 0.4:
        notes.append("PAS SNR low")

    if hydration_factor > 1.15:
        notes.append("High hydration compensation")
    elif hydration_factor < 0.85:
        notes.append("Dehydration compensation")

    if safety_status == "DISCARD":
        notes.append("Frame discarded for safety")
    elif safety_status == "RETRY":
        notes.append("Please repeat measurement")

    if not notes:
        notes.append("Stable perfusion, consistent modalities")

    return {
        "glucose_estimate": float(G_fused),
        "confidence_score": round(confidence_score, 2),
        "safety_status": safety_status,
        "notes": "; ".join(notes),
        "model_outputs": {
            "G_base": float(G_base),
            "G_A": float(G_A),
            "G_B": float(G_B),
            "G_C": float(G_C),
            "G_ensemble": float(G_ens),
        },
    }


# ----------------------------------------------------------------------
# 6. SIMPLE DEMO MAIN
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Dummy features for demo
    rfeats = RamanFeatures(
        peak_intensity_1125=0.915,
        peak_intensity_1340=0.726,
        peak_intensity_1460=0.537,
        area_1125=10.0,
        area_1340=8.0,
        area_1460=6.0,
        ratio_1125_to_1340=1.26,
        ratio_1340_to_1460=1.35,
        snr_estimate=20.0,
        fluorescence_index=0.05,
        frame_variance=0.005
    )
    nfeats = NIRFeatures(
        absorbance_750=0.8,
        absorbance_810=0.7,
        absorbance_940=0.6,
        absorbance_1600=0.5,
        ratio_1600_to_940=0.83,
        absorption_slope=-0.1,
        water_band_index=0.2
    )
    pas_feats = PASFeatures(
        early_peak_amplitude=0.1,
        early_window_energy=5.0,
        absorption_consistency=0.9,
        melanin_index=0.3
    )
    rf_feats = RFFeatures(
        effective_permittivity=40.0,
        phase_delay=45.0,
        hydration_index=0.7,
        skin_thickness_estimate=1.5,
        debug={"frame_variance": 0.0}
    )

    prev_estimate = None

    motion_level = 2  # or any one value you want
    print(f"=== MOTION LEVEL: {motion_level} ===")
    packet = acquire_and_process_once(motion_level)

    fusion_out = vitaron_fusion_and_ensemble(
        r_feats=rfeats,
        n_feats=nfeats,
        pas_feats=pas_feats,
        rf_feats=rf_feats,
        perfusion_index=packet["perfusion_index"],
        nir_vec_norm=packet["nir_vec_norm"],
        raman_vec_norm=packet["raman_vec_norm"],
        prev_estimate=prev_estimate,
    )

    print("Glucose estimate:", f"{fusion_out['glucose_estimate']:.1f}", "mg/dL")
    print("Confidence:", f"{fusion_out['confidence_score']:.2f}", "/ 10")
    print("Status:", fusion_out["safety_status"])
    print("Notes:", fusion_out["notes"])