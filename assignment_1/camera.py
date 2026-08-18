#Boilerplate code from https://www.geeksforgeeks.org/python/python-opencv-capture-video-from-camera/
import cv2

def camera_info_to_txt(cam):
    fps = cam.get(cv2.CAP_PROP_FPS)
    height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)

    with open('camera_outputs.txt', 'w') as f:
        f.write("Fps: {}, Height: {}, Width: {}".format(fps, height, width))


# Open the default camera
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()

    # Write camera information to 'camera_outputs.txt'
    camera_info_to_txt(cam)

    # Display the captured frame
    cv2.imshow('Camera', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
cv2.destroyAllWindows()