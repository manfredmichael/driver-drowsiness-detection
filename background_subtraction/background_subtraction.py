import cv2
import numpy as np




def resize_image(img1, resize_to):
    # load resized image as grayscale
    h, w, _= img1.shape

    # load background image as grayscale
    hh, ww, _ = resize_to.shape

    ratio = hh/h 
    img1 = cv2.resize(img1, (int(w * ratio), int(h * ratio)), interpolation= cv2.INTER_LINEAR)
    
    h, w, _= img1.shape

    ratio = ww/w
    if ratio < 1:
        img1 = cv2.resize(img1, (int(w * ratio), int(h * ratio)), interpolation= cv2.INTER_LINEAR)
    
    return img1

def insert_to_middle(back, img):
    if len(img.shape)<3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # if back.shape[0] < img.shape[0] or back.shape[1] < img.shape[1]:s
    #     img = rescale(img, scale_percent=50)

    img = resize_image(img, resize_to=back)


    h, w, _= img.shape
    hh, ww, _ = back.shape
    
    
# compute xoff and yoff for placement of upper left corner of resized image   
    yoff = round(hh-h)
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

def get_mask(webcam, background, return_steps=False):
    webcam = webcam.copy()
    background = background.copy()

    step_images = []
    
    # resize image
    webcam = rescale(webcam, scale_percent=40)
    background = rescale(background, scale_percent=40)

    step_images.append(['rescale', webcam])

    webcam = cv2.medianBlur(src=webcam, ksize=5)
    step_images.append(['medianBlur', webcam])
    background = cv2.medianBlur(src=background, ksize=5)

    image3 = cv2.absdiff(webcam, background)
    step_images.append(['absdiff', image3])

    image3 = cv2.inRange(image3, (5, 5, 5), (255, 255, 255))
    step_images.append(['inRange', image3])
    for i in range(100):
        image3 = cv2.GaussianBlur(image3,(5,5),0)
    step_images.append(['GaussianBlur', image3])

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # (thresh, binRed) = cv2.threshold(image3, 128, 255, cv2.THRESH_BINARY)
    # image3 = cv2.morphologyEx(image3, cv2.MORPH_OPEN, kernel, iterations=3)
    image3 = cv2.inRange(image3, (128), (255))
    step_images.append(['inRange', image3])

    if return_steps:
        return step_images, image3
    return image3

def apply_mask(webcam, mask, virtual_background):
    result = webcam.copy()
    result= resize_image(result, resize_to=virtual_background)
    # result[mask<128] = [0,0,0]


    frame = np.zeros((virtual_background.shape[0], virtual_background.shape[1], 3), np.uint8) # RGBA
    result = insert_to_middle(frame, result)
    mask_resized = insert_to_middle(frame, mask)

    result[mask_resized<128] = virtual_background[mask_resized<128]
    return result

def apply_virtual_background(webcam, background, virtual_background, return_steps=False):
    step_images, mask = get_mask(webcam, background, return_steps)
    result = apply_mask(webcam, mask, virtual_background)

    if return_steps:
        return step_images, result
    return result

def main():
    webcam = cv2.imread('images/webcam4.png')
    background = cv2.imread('images/background1.png')
    virtual_background = cv2.imread('images/nodeflux_background.png')
    virtual_background = rescale(virtual_background, scale_percent=40)
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


if __name__ == '__main__':
    main()
