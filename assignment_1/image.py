import cv2

def print_image_information(image):
    height = image.shape[0]
    width = image.shape[1]
    if len(image.shape) > 2:
        channel = image.shape[2]
    else:
        channel = 1

    size = image.size

    data_type = image.dtype

    print("Height: {} , Width: {}, Channel: {}, Size: {}, Data Type: {}".format(height,width,channel,size,data_type))

img = cv2.imread("iris-1.jpg")
print_image_information(img)
