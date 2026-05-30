import numpy as np
import os
import re
import requests
import json
import cv2
import ocr_config as config
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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
        print(f"DEBUG: Remote worker exception: {e}")
        return None

def check_worker_health(url):
    try:
        base_url = url.rstrip('/')
        response = requests.get(f"{base_url}/", timeout=2)
        return response.status_code == 200 or response.status_code == 404
    except:
        return False

def ai_check(processed_array, marker_y_levels=None, host=None, host_type='remote'):
    if isinstance(processed_array, tuple):
        processed_array = processed_array[0]
    
    if processed_array is None: return json.dumps({"error": "No image"})
    
    selected_host_type = 'remote'
    if check_worker_health(config.worker_remote):
        selected_host_type = 'remote'
    elif check_worker_health(config.worker_backup):
        selected_host_type = 'backup'
    else:
        return json.dumps({"error": "No vision workers available"})

    h, w = processed_array.shape[:2]
    
    # Surgical Vertical Slicing
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
        
        # B. GRID ISOLATION & MICRO-SLICING
        grid_only = slice_full[:, int(w*0.10):]
        h_g, w_g = grid_only.shape[:2]
        
        vis_map = grid_only.copy()
        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
        
        # GLOBAL GEOMETRY: Calculate row height based on TOTAL height for consistency
        # 4.1% of total image height is the physical "gold standard" for one row
        master_row_height = int(h * 0.041) 
        
        if i == 0:   top_offset_ratio = 0.255 # P4
        elif i == 1: top_offset_ratio = 0.120 # P15
        else:        top_offset_ratio = 0.010 # P14
        
        top_offset = int(h_g * top_offset_ratio)
        if i == 2: top_offset += 10 # Final fine-tune for Slice 3 (push down 10px)
        
        # Draw ALL 6 rows with increased overlap (30% total span)
        for r in range(6):
            c_y = top_offset + int((r + 0.5) * master_row_height)
            # Slim down boxes by ~20px total height as requested
            s = int(master_row_height * 0.65) - 10 
            cv2.rectangle(vis_map, (0, max(0, c_y-s)), (w_g, min(h_g, c_y+s)), colors[r % len(colors)], 2)
            cv2.putText(vis_map, f"R{r+1}", (5, max(15, c_y - s + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[r % len(colors)], 1)
        
        cv2.imwrite(os.path.join(SCRIPT_DIR, f"debug_map_section_{i+1}.jpg"), vis_map)

        section_nums = []
        for row_idx in range(6):
            center_y = top_offset + int((row_idx + 0.5) * master_row_height)
            span = int(master_row_height * 0.65) - 10
            y_start = max(0, center_y - span)
            y_end = min(h_g, center_y + span)
            row_strip = grid_only[y_start:y_end, :]
            
            res_row = call_remote_worker(row_strip, prompt_reader(full_label, prompt_type="main"), selected_host_type)
            if res_row and 'structured' in res_row:
                row_content = res_row['structured']
                print(f"DEBUG {full_label} Row {row_idx+1} RAW: {row_content.strip()}")
                
                raw_nums = re.findall(r"(?<!\d)(\d{2})(?!\d)", row_content)
                cleaned_row = [int(n) for n in raw_nums if 10 <= int(n) <= 99]
                
                print(f"DEBUG {full_label} Row {row_idx+1} EXTRACTED: {cleaned_row}")
                
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
