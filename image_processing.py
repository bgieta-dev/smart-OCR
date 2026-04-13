import cv2 as cv
import numpy as np
from pdf2image import convert_from_path


def image_processing(path):
    if path.lower().endswith(".pdf"):
        # Convert PDF to PNG with highest resolution (600 DPI)
        img = convert_from_path(path)[0]
        
    else:
        img = cv.imread(path)
    img = cv.cvtColor(np.array(img),cv.COLOR_BGR2GRAY)
    template = cv.imread('template.png', 0)
    h, w = template.shape[:2]
    h_img, w_img = img.shape[:2]
    top, bottom = 0, 0
    if h_img > w_img:
        h_img, w_img = w_img, h_img
        img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)

    res = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    threshold = 0.75
    loc = np.where(res >= threshold)

    for pt in zip(*loc[::-1]): 
        bottom_right = (pt[0] + w, pt[1] + h)
        #cv.rectangle(img, pt, bottom_right, (0, 255, 0), 2)
        if(pt[1]<h_img/2):
            top+=1
        else:
            bottom+=1
    print(top, bottom)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    if top < bottom:
        img = cv.rotate(img, cv.ROTATE_180)
        
    
    img = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
    cv.imwrite("ready_image.png",img)
    """
    cv.imshow("Result", img)
    cv.waitKey(0) 
    cv.destroyAllWindows()
    """

if __name__=="__main__":
    #image_processing("Szablon.png")
    image_processing('Skan 1.pdf')
