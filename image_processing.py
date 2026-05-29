import cv2 as cv
import numpy as np
import os
from pdf2image import convert_from_path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def image_processing(path):
    if not os.path.isabs(path):
        path = os.path.join(SCRIPT_DIR, path)
        
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nie znaleziono pliku wejściowego: {path}")

    if path.lower().endswith(".pdf"):
        try:
            pages = convert_from_path(path)
            if not pages:
                raise ValueError(f"Nie udało się skonwertować PDF (pusta lista stron): {path}")
            img = pages[0]
            img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
        except Exception as e:
            raise RuntimeError(f"Błąd podczas konwersji PDF: {str(e)}")
    else:
        img = cv.imread(path)
        if img is None:
            raise ValueError(f"Nie udało się wczytać obrazu przez OpenCV: {path}")

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    template_path = os.path.join(SCRIPT_DIR, 'template.png')
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Nie znaleziono pliku szablonu: {template_path}")
        
    template = cv.imread(template_path, 0)
    if template is None:
        raise ValueError(f"Błąd podczas wczytywania szablonu: {template_path}")

    h_img, w_img = img_gray.shape[:2]
    top, bottom = 0, 0
    
    if h_img > w_img:
        img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
        h_img, w_img = w_img, h_img
        img_gray = cv.rotate(img_gray, cv.ROTATE_90_CLOCKWISE)

    # Refined marker detection to return all Y-coordinates
    res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
    threshold = 0.45
    loc = np.where(res >= threshold)
    
    marker_points = []
    for pt in zip(*loc[::-1]):
        marker_points.append(pt)
        if pt[1] < h_img / 2:
            top += 1
        else:
            bottom += 1

    if top < bottom:
        img = cv.rotate(img, cv.ROTATE_180)
        # Flip marker points as well for consistency
        marker_points = [(w_img - p[0], h_img - p[1]) for p in marker_points]
    
    # Sort markers by Y coordinate to identify rows
    marker_points.sort(key=lambda p: p[1])
    
    # Group markers that are on the same Y level (within 20px)
    unique_y_levels = []
    if marker_points:
        current_y = marker_points[0][1]
        unique_y_levels.append(current_y)
        for p in marker_points[1:]:
            if abs(p[1] - current_y) > 40: # Significant jump to next row
                current_y = p[1]
                unique_y_levels.append(current_y)

    # Enhance contrast using CLAHE
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv.merge((cl,a,b))
    img_enhanced = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
    
    return img_enhanced, unique_y_levels

if __name__=="__main__":
    try:
        result = image_processing('Skan 1.pdf')
        print(f"Przetworzono obraz. Kształt: {result.shape}")
    except Exception as e:
        print(f"Błąd: {e}")
