import cv2
import numpy as np


def insert_to_middle(back, img):
    if len(img.shape)<3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # if back.shape[0] < img.shape[0] or back.shape[1] < img.shape[1]:s
    #     img = rescale(img, scale_percent=50)


# load resized image as grayscale
    h, w, _= img.shape

# load background image as grayscale
    hh, ww, _ = back.shape

    ratio = hh/h 
    img = cv2.resize(img, (int(w * ratio), int(h * ratio)), interpolation= cv2.INTER_LINEAR)
    h, w, _= img.shape
    
    
# compute xoff and yoff for placement of upper left corner of resized image   
    yoff = round((hh-h)/2)
    xoff = round((ww-w)/2)
    if yoff<0:
        yoff=0
    if xoff<0:
        xoff=0

# use numpy indexing to place the resized image in the center of background image
    result = back.copy()
    result[yoff:yoff+h, xoff:xoff+w] = img[:hh, :ww]

    return result

def rescale(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    return img

def get_mask(webcam, background):
    webcam = webcam.copy()
    background = background.copy()
    
    # resize image
    webcam = rescale(webcam, scale_percent=40)
    background = rescale(background, scale_percent=40)
    
    # webcam = cv2.GaussianBlur(webcam,(7,7),0)
    # background = cv2.GaussianBlur(background,(7,7),0)

    webcam = cv2.medianBlur(src=webcam, ksize=5)
    background = cv2.medianBlur(src=background, ksize=5)


    image3 = cv2.absdiff(webcam, background)

    image3 = cv2.inRange(image3, (5, 5, 5), (255, 255, 255))
    for i in range(100):
        image3 = cv2.GaussianBlur(image3,(3,3),0)

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # (thresh, binRed) = cv2.threshold(image3, 128, 255, cv2.THRESH_BINARY)
    # image3 = cv2.morphologyEx(image3, cv2.MORPH_OPEN, kernel, iterations=3)

    return image3

def apply_mask(webcam, mask, virtual_background):
    result = webcam.copy()
    # result[mask<128] = [0,0,0]


    frame = np.zeros((virtual_background.shape[0], virtual_background.shape[1], 3), np.uint8) # RGBA
    result = insert_to_middle(frame, result)
    mask_resized = insert_to_middle(frame, mask)

    result[mask_resized<128] = virtual_background[mask_resized<128]
    return result

def apply_virtual_background(webcam, background, virtual_background):
    mask = get_mask(webcam, background)
    result = apply_mask(webcam, mask, virtual_background)

    return result


if __name__ == '__main__':
    webcam = cv2.imread('images/webcam3.png')
    background = cv2.imread('images/background1.png')
    virtual_background = cv2.imread('images/nodeflux_background.png')
    mask = get_mask(webcam, background)
    result = apply_mask(webcam, mask, virtual_background)

    

    cv2.imshow('webcam', webcam)
    cv2.imshow('background', background)
    cv2.imshow('segmentation', mask)
    cv2.imshow('result', result)

    while(True):
        #Read each frame and flip it, and convert to grayscale
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

    cv2.destroyAllWindows()