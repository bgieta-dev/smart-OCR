import cv2 as cv
import numpy as np
import os
from pdf2image import convert_from_path

# Pobieramy ścieżkę do katalogu, w którym znajduje się skrypt
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def image_processing(path):
    # Jeśli ścieżka nie jest absolutna, szukaj jej względem katalogu skryptu
    if not os.path.isabs(path):
        path = os.path.join(SCRIPT_DIR, path)
        
    if not os.path.exists(path):
        print(f"Błąd: Nie znaleziono pliku wejściowego: {path}")
        return

    if path.lower().endswith(".pdf"):
        # Convert PDF to PNG with highest resolution (600 DPI)
        try:
            pages = convert_from_path(path)
            if not pages:
                print(f"Błąd: Nie udało się skonwertować PDF: {path}")
                return
            img = pages[0]
        except Exception as e:
            print(f"Błąd podczas konwersji PDF: {e}")
            return
    else:
        img = cv.imread(path)
        if img is None:
            print(f"Błąd: Nie udało się wczytać obrazu: {path}")
            return

    img_gray = cv.cvtColor(np.array(img), cv.COLOR_BGR2GRAY)
    
    template_path = os.path.join(SCRIPT_DIR, 'template.png')
    template = cv.imread(template_path, 0)
    if template is None:
        print(f"Błąd: Nie znaleziono szablonu: {template_path}")
        # Tworzymy pusty placeholder jeśli brak szablonu, aby skrypt się nie wywalił
        # lub po prostu przerywamy
        return

    h, w = template.shape[:2]
    h_img, w_img = img_gray.shape[:2]
    top, bottom = 0, 0
    
    if h_img > w_img:
        # Zakładamy orientację poziomą dla przetwarzania
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
    
    print(f"Top matches: {top}, Bottom matches: {bottom}")

    if top < bottom:
        img_gray = cv.rotate(img_gray, cv.ROTATE_180)
    
    img_resized = cv.resize(img_gray, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
    
    output_path = os.path.join(SCRIPT_DIR, "ready_image.png")
    cv.imwrite(output_path, img_resized)
    print(f"Zapisano wynik do: {output_path}")

if __name__=="__main__":
    image_processing('Skan 1.pdf')
