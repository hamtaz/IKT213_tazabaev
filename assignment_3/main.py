import cv2
import numpy as np

def sobel_edge_function(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, ksize=(3,3), sigmaX=0)

    #https://opencv.org/edge-detection-using-opencv/#h-sobel-operator
    sobelx = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=1)
    sobely = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=1)
    gradient_magnitude = cv2.magnitude(sobelx, sobely)
    gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)

    cv2.imwrite("solutions/sobel.png", gradient_magnitude)


def canny_edge_detection(image, threshold_1, threshold_2):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, ksize=(3,3), sigmaX=0)

    edges = cv2.Canny(img_blur, threshold_1, threshold_2)
    cv2.imwrite("solutions/canny.png", edges)

def template_match(image, template):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = template_gray.shape[::-1]
    method = cv2.TM_CCOEFF_NORMED

    res = cv2.matchTemplate(img_gray, template_gray, method)
    threshold_loc = np.where(res>=0.9)

    for i in zip(*threshold_loc[::-1]):
        cv2.rectangle(image, i, (i[0]+w, i[1]+h), (0, 0, 255), 1)

    cv2.imwrite("solutions/template_match.png", image)

def resize(image, scale_factor:int, up_or_down:str):
    up_or_down = up_or_down.lower()
    rows, cols, _channels = map(int, image.shape)

    if up_or_down == 'up':
        image = cv2.pyrUp(image, dstsize=(scale_factor*cols, scale_factor*rows))
        cv2.imwrite("solutions/resize_up.png", image)
    elif up_or_down == "down":
        image = cv2.pyrDown(image, dstsize=(cols//scale_factor, rows//scale_factor))
        cv2.imwrite("solutions/resize_down.png", image)
    else:
        print("String is neither up or down")

if __name__ == '__main__':
    image = cv2.imread('lambo.png')
    temp_img = cv2.imread('shapes.png')
    template = cv2.imread('shapes_template.jpg')
    sobel_edge_function(image)
    canny_edge_detection(image,50,50)
    template_match(temp_img, template)
    resize(image, 2, "up")
    resize(image, 2, "DOWN")
    resize(image, 2, "notupordown")
