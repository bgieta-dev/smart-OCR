import numpy as np
import os
import re
import requests
import json
import cv2
import ocr_config as config
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def debug_print(message):
    if config.DEBUG_MODE:
        print(message)

def prompt_reader(label="Unknown", prompt_type="main"):
    filename = "prompt.txt" if prompt_type == "main" else "label_prompt.txt"
    prompt_path = os.path.join(SCRIPT_DIR, filename)
    if not os.path.exists(prompt_path):
        return f"Extract numbers from grid {label}." if prompt_type == "main" else "Identify label."
    with open(prompt_path, "r") as f:
        p = f.read().strip()
        return p.replace("{label}", label)

def call_remote_worker(img_array, prompt, host_type='remote'):
    worker_url = config.worker_remote if host_type == 'remote' else config.worker_backup
    worker_url = f"{worker_url.rstrip('/')}/process"
    
    _, img_encoded = cv2.imencode('.jpg', img_array)
    img_bytes = img_encoded.tobytes()
    
    try:
        files = {'image': ('image.jpg', img_bytes, 'image/jpeg')}
        request_id = str(int(time.time() * 1000))
        data = {'prompt': prompt, 'request_id': request_id}
        
        response = requests.post(worker_url, files=files, data=data, timeout=300)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        debug_print(f"DEBUG: Remote worker exception: {e}")
        return None

def check_worker_health(url):
    try:
        base_url = url.rstrip('/')
        response = requests.get(f"{base_url}/", timeout=2)
        return response.status_code == 200 or response.status_code == 404
    except:
        return False

def find_grid_boundaries(img):
    """Dynamically find the top and bottom horizontal lines of the grid using morphological filtering."""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Threshold to get black features
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # USE MORPHOLOGY TO TARGET HORIZONTAL LINES ONLY
        # A 50x1 rectangle kernel will preserve lines but eliminate squares (markers)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        detect_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        h_sum = np.sum(detect_lines, axis=1)
        w = img.shape[1]
        
        # A valid grid line should cover a significant portion of the width
        threshold = w * 255 * 0.3
        line_indices = np.where(h_sum > threshold)[0]
        
        if len(line_indices) > 3:
            top = line_indices[0]
            bottom = line_indices[-1]
            # Sanity check: the grid area should be a reasonable size
            if (bottom - top) > (img.shape[0] * 0.10):
                return top, bottom
    except Exception as e:
        debug_print(f"DEBUG: Grid detection error: {e}")
    return None, None

def ai_check(processed_array, marker_y_levels=None, host=None, host_type='remote'):
    if isinstance(processed_array, tuple):
        processed_array = processed_array[0]
    
    if processed_array is None: return json.dumps({"error": "No image"})
    
    selected_host_type = 'remote'
    if check_worker_health(config.worker_remote):
        selected_host_type = 'remote'
        debug_print("DEBUG: Using Remote Worker (Primary).")
    elif check_worker_health(config.worker_backup):
        selected_host_type = 'backup'
        debug_print("DEBUG: Using Backup Worker (Laptop).")
    else:
        return json.dumps({"error": "No vision workers available"})

    h, w = processed_array.shape[:2]
    
    # 1. Vertical Slicing
    slices = []
    if marker_y_levels and len(marker_y_levels) >= 2:
        slices.append(processed_array[50:int(marker_y_levels[1] - 40), :])
        if len(marker_y_levels) >= 3:
            slices.append(processed_array[int(marker_y_levels[1] - 10):int(marker_y_levels[2] - 60), :])
            slices.append(processed_array[int(marker_y_levels[2] - 50):, :])
        else:
            slices.append(processed_array[int(marker_y_levels[1] - 10):, :])
    else:
        slices = [
            processed_array[50:int(h*0.33), :],
            processed_array[int(h*0.34):int(h*0.64), :],
            processed_array[int(h*0.65):, :]
        ]
    
    internal_results = []
    label_counts = {}
    
    debug_print(f"DEBUG: Grid Isolation extraction for {len(slices)} slices...")
    for i, slice_full in enumerate(slices):
        h_s, w_s = slice_full.shape[:2]
        label_area = slice_full[:, 0:int(w*0.12)]
        res_label = call_remote_worker(label_area, prompt_reader(prompt_type="label"), selected_host_type)
        
        label_raw = res_label['structured'] if res_label else f"SLICE_{i+1}"
        label_match = re.search(r"([Pp]\d+)", label_raw)
        base_label = label_match.group(1).upper() if label_match else f"SLICE_{i+1}"
        len_match = re.search(r"(\d+,\d+\s?dł\.?)", label_raw, re.IGNORECASE)
        length_val = len_match.group(1).lower() if len_match else ""
        full_label = f"{base_label} ({length_val})" if length_val else base_label
        
        grid_only = slice_full[:, int(w*0.10):]
        h_g, w_g = grid_only.shape[:2]
        
        # UNIFIED GEOMETRY CALIBRATION
        master_row_height = int(h * 0.041) 
        
        if i == 0:
            top, _ = find_grid_boundaries(grid_only)
            if top is not None:
                debug_print(f"DEBUG: Dynamic top found for S1: {top}")
                # User says 0.95 was too much (1 row too low), so we use a half-shift
                top_offset = top + int(master_row_height * 0.10)
            else:
                # Fallback sweet-spot for P4
                top_offset = int(h_g * 0.115)
        elif i == 1:
            # P15: Git
            top_offset = int(h_g * 0.120)
        else:
            # P14: Git
            top_offset = int(h_g * 0.005) + 10

        # --- VISUALIZATION DEBUG ---
        if config.DEBUG_MODE:
            vis_map = grid_only.copy()
            colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
            for r in range(6):
                c_y = top_offset + int((r + 0.5) * master_row_height)
                s = int(master_row_height * 0.65) - 5 # Back to user-confirmed symmetric
                y_s = max(0, int(c_y - s))
                y_e = min(h_g, int(c_y + s))
                cv2.rectangle(vis_map, (0, y_s), (w_g, y_e), colors[r % len(colors)], 2)
                cv2.putText(vis_map, f"R{r+1}", (5, y_s + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[r % len(colors)], 1)
            cv2.imwrite(os.path.join(SCRIPT_DIR, f"debug_map_section_{i+1}.jpg"), vis_map)

        section_nums = []
        for row_idx in range(6):
            center_y = top_offset + int((row_idx + 0.5) * master_row_height)
            s_top = int(master_row_height * 0.65) - 5
            s_bottom = int(master_row_height * 0.65) - 15
            y_start = max(0, int(center_y - s_top))
            y_end = min(h_g, int(center_y + s_bottom))
            row_strip = grid_only[y_start:y_end, :]
            
            res_row = call_remote_worker(row_strip, prompt_reader(full_label, prompt_type="main"), selected_host_type)
            if res_row and 'structured' in res_row:
                row_content = res_row['structured']
                debug_print(f"DEBUG {full_label} Row {row_idx+1} RAW: {row_content.strip()}")
                raw_nums = re.findall(r"(?<!\d)(\d{2})(?!\d)", row_content)
                cleaned_row = [int(n) for n in raw_nums if 10 <= int(n) <= 99]
                debug_print(f"DEBUG {full_label} Row {row_idx+1} EXTRACTED: {cleaned_row}")

                if cleaned_row:
                    section_nums.extend(cleaned_row)
                else:
                    if row_idx >= 3: break

        if section_nums:
            internal_results.append((full_label, section_nums))
            label_counts[full_label] = label_counts.get(full_label, 0) + 1

    final_results = {}
    current_counters = {}
    for base_str, nums in internal_results:
        if label_counts[base_str] > 1:
            count = current_counters.get(base_str, 0) + 1
            current_counters[base_str] = count
            final_results[f"{base_str}.{count}"] = nums
        else:
            final_results[base_str] = nums

    return json.dumps(final_results, indent=2)
