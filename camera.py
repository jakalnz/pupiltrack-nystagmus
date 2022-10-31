#currenly only opens the camera ever 6-8 time, obviously not quite called right

import cv2
import numpy as np
import matplotlib.pyplot as plt

#
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
#cap = cv2.VideoCapture('Vertical Nystagmus.mp4')
#cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap_args = [0, cv2.CAP_DSHOW]

cap = cv2.VideoCapture(*cap_args)

#Check if camera was opened correctly
if not (cap.isOpened()):
    print("Could not open video device")


#Set the resolution
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 576)
#cap.set(cv2.CAP_PROP_CODEC_PIXEL_FORMAT, 0x32595559)

# Capture frame-by-frame
while(True):
    ret, frame = cap.read()

    # Display the resulting frame
    
    cv2.imshow("preview",frame)
    
    #cv2.imwrite("outputImage.jpg", frame)

    #Waits for a user input to quit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()