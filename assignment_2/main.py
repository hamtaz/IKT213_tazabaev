import cv2
import numpy as np

def padding(image, border_width):
    reflect = cv2.copyMakeBorder(image, border_width, border_width, border_width, border_width, cv2.BORDER_REFLECT)
    cv2.imwrite("solutions/padded.png", reflect)

def cropping(image, x_0, x_1, y_0, y_1):
    height = image.shape[0]
    width = image.shape[1]

    cropped = image[y_0:height - y_1, x_0: width - x_1]
    cv2.imwrite("solutions/cropped.png", cropped)

def resize(image, height, width):
    resized = cv2.resize(image, dsize=(width, height))
    cv2.imwrite("solutions/resized.png", resized)

def copy(image, emptyPictureArray):
    height = image.shape[0]
    width = image.shape[1]

    emptyPictureArray[0:height, 0:width] = image
    cv2.imwrite("solutions/copy.png", emptyPictureArray)

def grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("solutions/grayscale.png", gray)

def hsv(image):
    hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imwrite("solutions/hsv.png", hsvImage)

def hue_shifted(image, emptyPictureArray, hue):
    height = image.shape[0]
    width = image.shape[1]

    emptyPictureArray[0:height, 0:width] = image + hue
    cv2.imwrite("solutions/hue_shifted.png", emptyPictureArray)

def smoothing(image):
    ksize=(15,15)
    smoothed_image = cv2.GaussianBlur(image, ksize, 0)
    cv2.imwrite("solutions/smoothed.png", smoothed_image)

def rotation(image, rotation_angle):
    if rotation_angle == 90:
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite("solutions/90_rotated.png", rotated)
    elif rotation_angle == 180:
        rotated = cv2.rotate(image, cv2.ROTATE_180)
        cv2.imwrite("solutions/180_rotated.png", rotated)
if __name__ == "__main__":
    image = cv2.imread("iris-1.png")

    height = image.shape[0]
    width = image.shape[1]
    emptyPictureArray = np.zeros((height, width, 3), dtype=np.uint8)

    #1
    padding(image,100)

    #2
    cropping(image, 200, 130, 200, 130)

    #3
    resize(image, 200, 200)

    #4
    copy(image, emptyPictureArray)

    #5
    grayscale(image)

    #6
    hsv(image)

    #7
    hue_shifted(image, emptyPictureArray, 50)

    #8
    smoothing(image)

    #9
    #rotation(image,90)
    rotation(image, 180)
    cv2.waitKey(0)