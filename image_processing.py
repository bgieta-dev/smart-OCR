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
        img_gray = cv.rotate(img_gray, cv.ROTATE_90_CLOCKWISE)
        h_img, w_img = w_img, h_img

    res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
    threshold = 0.75
    loc = np.where(res >= threshold)

    for pt in zip(*loc[::-1]): 
        if pt[1] < h_img / 2:
            top += 1
        else:
            bottom += 1
    

    if top < bottom:
        img_gray = cv.rotate(img_gray, cv.ROTATE_180)
    
    img_resized = cv.resize(img_gray, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
    return img_resized

if __name__=="__main__":
    try:
        result = image_processing('Skan 1.pdf')
        print(f"Przetworzono obraz. Kształt: {result.shape}")
    except Exception as e:
        print(f"Błąd: {e}")
