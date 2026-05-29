import numpy as np
import os
import torch
import re
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from PIL import Image
from openai import OpenAI
import ocr_config as config
import json
import cv2

# Initialize DocTR
# DocTR is highly reliable for complex layouts and handwritten text
# Using 'parseq' recognition which is state-of-the-art for handwriting
print("Loading OCR Model (DocTR with PARSeq)...")
os.environ["USE_TORCH"] = "1"
predictor = ocr_predictor(det_arch='fast_base', reco_arch='parseq', pretrained=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    predictor.to(device)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def prompt_reader():
    prompt_path = os.path.join(SCRIPT_DIR, "prompt.txt")
    if not os.path.exists(prompt_path):
        return "Convert the following raw OCR text into a structured JSON grid."
    with open(prompt_path, "r") as f:
        return f.read().strip()

def call_vllm_text(text_content, prompt):
    base_url = f"{config.host_remote.rstrip('/')}/v1"
    client = OpenAI(api_key="token-is-ignored-by-vllm", base_url=base_url, timeout=120.0)
    full_prompt = f"{prompt}\n\nRAW OCR TEXT:\n{text_content}"
    try:
        response = client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.1,
            max_tokens=2048
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"DEBUG: vLLM error: {e}")
        return None

def get_table_slices(img_array, marker_y_levels=None):
    h, w = img_array.shape[:2]
    slices = []
    if marker_y_levels and len(marker_y_levels) >= 2:
        y_levels = sorted(marker_y_levels)
        if len(y_levels) >= 3:
            split1 = (y_levels[0] + y_levels[1]) // 2 - 50
            split2 = (y_levels[1] + y_levels[2]) // 2 - 50
            slices.append(img_array[0 : split1, :])
            slices.append(img_array[split1 : split2, :])
            slices.append(img_array[split2 : , :])
            return slices
    slice_h = h // 3
    slices.append(img_array[0 : slice_h, :])
    slices.append(img_array[slice_h : 2*slice_h, :])
    slices.append(img_array[2*slice_h : , :])
    return slices

def run_ocr_on_slice(slice_img_array):
    # Use temporary file to ensure compatibility with DocTR
    temp_path = "temp_slice.jpg"
    cv2.imwrite(temp_path, slice_img_array)
    doc = DocumentFile.from_images([temp_path])
    
    print("DEBUG: Running DocTR on slice...")
    result = predictor(doc)
    
    # Clean up
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    # Export to JSON format for easy parsing
    json_output = result.export()
    
    line_texts = []
    for page in json_output['pages']:
        for block in page['blocks']:
            for line in block['lines']:
                # Combine words in line
                line_text = " ".join([word['value'] for word in line['words']])
                if line_text.strip():
                    line_texts.append(line_text.strip())
                    
    return "\n".join(line_texts)

def ai_check(processed_array, marker_y_levels=None, host=None):
    if isinstance(processed_array, tuple):
        if not marker_y_levels: marker_y_levels = processed_array[1]
        processed_array = processed_array[0]
    
    if processed_array is None: return json.dumps({"error": "No image"})
    if host: config.host_remote = host
    
    img_slices = get_table_slices(processed_array, marker_y_levels)
    results = {}
    prompt = prompt_reader()
    
    for i, slice_img in enumerate(img_slices):
        debug_path = f"debug_slice_{i+1}.jpg"
        cv2.imwrite(debug_path, slice_img)
        
        # --- STAGE 1: DocTR OCR ---
        print(f"DEBUG: Processing Slice {i+1} with DocTR...")
        try:
            raw_text = run_ocr_on_slice(slice_img)
        except Exception as e:
            print(f"Error running OCR on slice {i+1}: {e}")
            continue
        
        print(f"DEBUG OCR Output:\n{raw_text[:300]}...")
        
        if not raw_text.strip():
            continue

        # --- STAGE 2: LLM STRUCTURING ---
        structured_content = call_vllm_text(raw_text, prompt)
        if not structured_content:
            continue
            
        print(f"DEBUG LLM Output for Slice {i+1}:\n{structured_content}")
            
        try:
            clean = structured_content.strip()
            if "```json" in clean:
                clean = clean.split("```json")[1].split("```")[0].strip()
            data = json.loads(clean)
            
            for k, v in data.items():
                if isinstance(v, list):
                    # Normalized key (P_UNKNOWN should be merged carefully or uniquely identified)
                    # But for now, we follow the key found by LLM
                    cleaned_list = []
                    for item in v:
                        s_item = str(item).strip().replace(" ", "")
                        # Match only 2-digit integers
                        match = re.search(r"\b\d{2}\b", s_item)
                        if match:
                            num_str = match.group()
                            try:
                                cleaned_list.append(int(num_str))
                            except:
                                pass
                    if cleaned_list:
                        if k in results and isinstance(results[k], list):
                            results[k].extend(cleaned_list)
                        else:
                            results[k] = cleaned_list
        except Exception as e:
            print(f"DEBUG: Parsing error on slice {i+1}: {e}")
            continue
            
    return json.dumps(results, indent=2)

if __name__ == "__main__":
    from image_processing import image_processing
    try:
        res = image_processing('canvas.png')
        print(ai_check(res))
    except Exception as e:
        print(f"Error: {e}")
