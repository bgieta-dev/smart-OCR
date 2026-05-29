import numpy as np
import os
import re
import requests
import json
import cv2
import ocr_config as config

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def prompt_reader():
    prompt_path = os.path.join(SCRIPT_DIR, "prompt.txt")
    if not os.path.exists(prompt_path):
        return "Extract all 2-digit numbers from the grid sections (P4, P13) and format as P<ID>: [numbers]."
    with open(prompt_path, "r") as f:
        return f.read().strip()

import time

def call_remote_worker(img_array, prompt, host_type='remote'):
    worker_url = config.worker_remote if host_type == 'remote' else config.worker_backup
    worker_url = f"{worker_url.rstrip('/')}/process"
    
    print(f"\n--- DEBUG VISION SEND ({host_type}) ---")
    _, img_encoded = cv2.imencode('.jpg', img_array)
    img_bytes = img_encoded.tobytes()
    
    try:
        files = {'image': ('image.jpg', img_bytes, 'image/jpeg')}
        # Cache buster: unique ID for every single slice to force context reset
        request_id = str(int(time.time() * 1000))
        data = {'prompt': prompt, 'request_id': request_id}
        
        response = requests.post(worker_url, files=files, data=data, timeout=300)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"DEBUG: Worker error: {response.text}")
            return None
    except Exception as e:
        print(f"DEBUG: Remote worker exception: {e}")
        return None

def ai_check(processed_array, marker_y_levels=None, host=None, host_type='remote'):
    if isinstance(processed_array, tuple):
        processed_array = processed_array[0]
    
    if processed_array is None: return json.dumps({"error": "No image"})
    
    prompt = prompt_reader()
    h, w = processed_array.shape[:2]
    
    # Precise slicing using dynamic markers
    slices = []
    if marker_y_levels and len(marker_y_levels) >= 2:
        print(f"DEBUG: Using {len(marker_y_levels)} markers for precise slicing.")
        # Slice 1: Ideal Top
        slices.append(processed_array[50:int(marker_y_levels[1] - 40), :])
        # Slice 2: Ideal Middle
        if len(marker_y_levels) >= 3:
            slices.append(processed_array[int(marker_y_levels[1] - 10):int(marker_y_levels[2] - 60), :])
            # Slice 3: Ideal Bottom
            slices.append(processed_array[int(marker_y_levels[2] - 50):, :])
        else:
            slices.append(processed_array[int(marker_y_levels[1] - 10):, :])
    else:
        print("DEBUG: Markers failed, using IDEAL fallback ratios.")
        # Slice 1: 0.01 to 0.33
        slices.append(processed_array[50:int(h*0.33), :])
        # Slice 2: 0.34 to 0.64
        slices.append(processed_array[int(h*0.34):int(h*0.64), :])
        # Slice 3: 0.65 to end
        slices.append(processed_array[int(h*0.65):, :])
    
    all_results = {}
    print(f"DEBUG: Processing {len(slices)} slices via VISION worker {host_type}...")
    for i, slice_img in enumerate(slices):
        debug_path = os.path.join(SCRIPT_DIR, f"debug_slice_{i+1}.jpg")
        try: cv2.imwrite(debug_path, slice_img)
        except Exception: pass

        print(f"DEBUG: Processing slice {i+1}/{len(slices)}...")
        res = call_remote_worker(slice_img, prompt, host_type)

        if res and 'structured' in res:
            content = res['structured']
            print(f"DEBUG SLICE {i+1} RAW: {content[:100]}...")
            
            # Dynamic Label Extraction
            label_match = re.search(r"([Pp]\d+)", content)
            if label_match:
                label = label_match.group(1).upper().strip()
                label_num = label.replace("P", "")
                
                # Extract numbers from Markdown table cells
                cells = content.split("|")
                cleaned_nums = []
                
                # Header filter: Skip cells that likely belong to the header (first 11 cells)
                # Tables in Markdown have: | Header1 | Header2 | ... |
                # and then a separator |---|---|...|
                # and then Data. We want to skip headers.
                
                header_detected = False
                for cell in cells:
                    cell = cell.strip()
                    # Skip the "---|---" row
                    if "---" in cell:
                        header_detected = True
                        continue
                    if not header_detected:
                        continue
                        
                    # Now we are in data rows. Look for 2-digit numbers (10-99)
                    nums = re.findall(r"(?<!\d)(\d{2})(?!\d)", cell)
                    if nums:
                        val = int(nums[0])
                        # Extra safeguard: Skip the number if it's the section ID (e.g. 4 for P4)
                        # but only if it's suspicious. 10-99 are almost always measurements.
                        if val >= 10:
                            cleaned_nums.append(val)
                
                # Anti-Hallucination: Variety Check
                if len(cleaned_nums) > 10:
                    unique_count = len(set(cleaned_nums))
                    if unique_count <= 3:
                        print(f"DEBUG: Low variety in table {label}. Discarding.")
                        cleaned_nums = []

                if cleaned_nums:
                    if label not in all_results: all_results[label] = []
                    all_results[label].extend(cleaned_nums)
            else:
                print(f"DEBUG: No label found in slice {i+1}.")

    return json.dumps(all_results, indent=2)
