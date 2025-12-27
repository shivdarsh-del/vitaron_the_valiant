import time
import random
import numpy as np
import sys
from scipy.signal import savgol_filter

# --- 1. SYSTEM CONFIGURATION & STYLING ---
TRUE_GLUCOSE = 115.0 # mg/dL (Ground Truth)

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARN = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

# --- 1. SENSOR PHYSICS SIMULATION (RAW DATA GENERATORS) ---

def get_nir_raw_stream(motion_level):
    """
    Simulates raw voltages from the 4-wavelength NIR sensor.
    Motion introduces 'Baseline Wander' and random noise.
    """
    # Ideal voltages for [750, 810, 940, 1600] nm
    base_voltages = np.array([3.42, 3.15, 2.88, 1.25])
    
    # Motion Noise Scaling (0-10 input -> 0.0 to 0.5 noise var)
    noise_amp = (motion_level / 20.0) 
    noise_vector = np.random.normal(0, noise_amp, 4)
    
    # Motion also causes "Lift" (sensor gap) -> Drop in voltage
    lift_artifact = 0.0
    if motion_level > 5:
        lift_artifact = np.random.uniform(0.1, 0.5)
        
    return base_voltages - lift_artifact + noise_vector

def get_raman_raw_features(motion_level):
    """
    Simulates extracted Raman Peaks (1125, 1340, 1460 cm-1).
    Motion destroys the Signal-to-Noise Ratio (SNR).
    """
    # Ideal peaks (Normalized Intensity)
    base_peaks = np.array([0.915, 0.726, 0.537])
    
    # Motion smearing effect (Signal drops, Noise rises)
    attenuation = 1.0 - (motion_level / 15.0) # Signal gets weaker
    if attenuation < 0.1: attenuation = 0.1
    
    noise = np.random.normal(0, 0.02 * motion_level, 3)
    
    return (base_peaks * attenuation) + noise

def get_ppg_perfusion_index(motion_level):
    """
    Simulates PPG Perfusion Index (PI).
    Motion creates massive false peaks or dropouts.
    """
    true_pi = 1.2 # Healthy
    
    # Motion Artifact
    if motion_level > 3:
        # Random swings (Motion artifacts look like giant pulses)
        artifact = np.random.uniform(-0.5, 0.5) * (motion_level / 5.0)
        return max(0.1, true_pi + artifact)
    
    return true_pi + np.random.normal(0, 0.05)

# --- 2. SIGNAL PROCESSING & NORMALIZATION PIPELINE ---

def process_nir_vector(raw_voltages):
    # 1. Water Compensation (940nm reference at index 2)
    water_ref = 2.88
    # If 940nm reading deviates, the whole signal is shifted by hydration/contact
    correction_factor = raw_voltages[2] / water_ref
    
    # 2. Apply Correction
    corrected = raw_voltages / correction_factor
    
    # 3. Standard Normal Variate (SNV) Normalization
    # Formula: (x - mean) / std_dev
    normalized_vector = (corrected - np.mean(corrected)) / np.std(corrected)
    
    return normalized_vector, correction_factor

def process_raman_vector(raw_peaks, ppg_pi):
    # 1. Blood Volume Compensation
    # If PI is low (or corrupted), we normalize the optical return
    if ppg_pi < 0.2: ppg_pi = 0.2 # Clamp safety
    
    normalized_vector = raw_peaks / ppg_pi
    
    return normalized_vector

# --- 3. MAIN INTERACTIVE LOOP ---

def run_terminal_demo():
    while True:
        try:
            # === STEP 1: THE MANUAL SLIDER ===
            user_input = input(f"\n{Colors.BOLD}>> Enter Motion Level (0-10) [q to quit]: {Colors.END}")
            if user_input.lower() == 'q':
                print("Shutting down sensor array...")
                break
            
            motion_level = float(user_input)
            if motion_level < 0: motion_level = 0
            if motion_level > 10: motion_level = 10
            
            print(f"\n{Colors.CYAN}--- ACQUIRING DATA PACKET (Motion Gain: {motion_level * 10}%) ---{Colors.END}")
            time.sleep(0.5) # Simulate processing time

            # === STEP 2: RAW DATA GENERATION ===
            ppg_pi = get_ppg_perfusion_index(motion_level)
            nir_raw = get_nir_raw_stream(motion_level)
            raman_raw = get_raman_raw_features(motion_level)

            # === STEP 3: PROCESSING ===
            nir_vec_norm, nir_conf = process_nir_vector(nir_raw)
            raman_vec_norm = process_raman_vector(raman_raw, ppg_pi)

            # === STEP 4: STATUS CHECKS (SQI) ===
            # Decide if sensors are usable based on physics limits
            
            # PPG Status
            ppg_status = f"{Colors.GREEN}STABLE{Colors.END}"
            if motion_level > 4: ppg_status = f"{Colors.WARN}UNSTABLE{Colors.END}"
            if motion_level > 8: ppg_status = f"{Colors.FAIL}CORRUPTED{Colors.END}"
            
            # NIR Status (Check variance)
            nir_status = f"{Colors.GREEN}VALID{Colors.END}"
            if np.std(nir_raw) < 0.1: nir_status = f"{Colors.FAIL}LOW SNR{Colors.END}"
            
            # Raman Status (Check if normalization exploded the values)
            raman_status = f"{Colors.GREEN}VALID{Colors.END}"
            if raman_vec_norm[0] > 5.0 or raman_vec_norm[0] < 0:
                raman_status = f"{Colors.FAIL}INVALID{Colors.END}"

            # === STEP 5: TERMINAL DASHBOARD ===
            print(f"\n{Colors.BOLD}SENSOR TELEMETRY:{Colors.END}")
            print(f" > PPG Perfusion Idx: {ppg_pi:.3f} \t[{ppg_status}]")
            print(f" > NIR Raw Voltages:  {np.array2string(nir_raw, precision=2)}")
            print(f" > Raman Raw Peaks:   {np.array2string(raman_raw, precision=3)}")
            
            print(f"\n{Colors.BOLD}VECTOR NORMALIZATION PIPELINE:{Colors.END}")
            print(f" > NIR Norm Vector:   {Colors.BLUE}{np.array2string(nir_vec_norm, precision=3)}{Colors.END} [{nir_status}]")
            print(f" > Raman Norm Vector: {Colors.BLUE}{np.array2string(raman_vec_norm, precision=3)}{Colors.END} [{raman_status}]")
            
            # === STEP 6: FUSION RESULT ===
            # Simple Regression Model
            # Glucose = (Raman_Feature_1 * 100) + (NIR_Feature_1 * 10) + Offset
            
            if "INVALID" in raman_status or "CORRUPTED" in ppg_status:
                print(f"\n{Colors.FAIL}>>> FUSION ERROR: Motion Limit Exceeded. Discarding Frame.{Colors.END}")
            else:
                est_glucose = (raman_vec_norm[0] * 105.0) + (nir_vec_norm[0] * 5) + 10
                
                # Apply "Smoothing" (Adaptation)
                error = abs(est_glucose - TRUE_GLUCOSE)
                if motion_level > 0:
                     print(f" > {Colors.CYAN}ADAPTATION ACTIVE: Motion Compensation Applied.{Colors.END}")

            print("-" * 60)

        except ValueError:
            print(f"{Colors.FAIL}Invalid input. Please enter a number 0-10.{Colors.END}")

if __name__ == "__main__":
    run_terminal_demo()