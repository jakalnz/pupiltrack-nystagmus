import cv2
from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt
import csv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import _thread
import os
from datetime import datetime
import shutil #rcopy and rename
from graph_csv import graph_csv
import argparse
from gooey import Gooey, GooeyParser



#### ARGUMENTS #####
#source_video_path = 'Bob.MPG'
#output_filename = 'Bob'
#isCamera = False # if True overrides source_video_path
# If the input is a camera - set isStream to True to record (have not tested)
#isStream = False

#### GUI GOES HERE ####
# GUI to simplify argument input


@Gooey(
    program_name="Pupil Tracker",
    program_description="Program takes file or camera input of pupil and outputs a csv file of pupil movement with graph, can also optionally capture camera input.",
)
def get_args():
  parser = GooeyParser()
  parser.add_argument("-p", "--source_video_path", help='input the source file for analysis', default="Bob.MPG",widget="FileChooser",gooey_options=dict(wildcard="Video files (*.mp4, *.mkv, *.avi, *.mpg)|*.mp4;*.mkv;*.avi;*.mpg)"))
  parser.add_argument("-o", "--output_filename", help='input the patient name', default="output")
  parser.add_argument("-c", "--camera", help="applying this flag activates camera as source and overides --source_video_path input",action="store_true")
  parser.add_argument("-s", "--save_processed_video", help="applying this flag will save the processed video file", action="store_true")
  args = parser.parse_args()
  return args

args = get_args() #used Gooey as a shortcut
source_video_path = args.source_video_path
output_filename = args.output_filename
isCamera = args.camera
isStream = args.save_processed_video


#### METHODS ####

# Call the plot.py in a function plot, plot.py will run in a separate thread and graph pupil movement in near realtime
# thanks to  https://pyshine.com/How-to-plot-real-time-frame-rate-in-opencv-and-matplotlib/
def plot():
	os.system('python plot.py')


#Identify pupil position - awesome methods from nphilip1098
def fit_rotated_ellipse_ransac(data,iter=50,sample_num=10,offset=80.0):

    count_max = 0
    effective_sample = None

    for i in range(iter):
        sample = np.random.choice(len(data), sample_num, replace=False)

        xs = data[sample][:,0].reshape(-1,1)
        ys = data[sample][:,1].reshape(-1,1)

        J = np.mat( np.hstack((xs*ys,ys**2,xs, ys, np.ones_like(xs,dtype=np.float))) )
        Y = np.mat(-1*xs**2)
        P= (J.T * J).I * J.T * Y

        # fitter a*x**2 + b*x*y + c*y**2 + d*x + e*y + f = 0
        a = 1.0; b= P[0,0]; c= P[1,0]; d = P[2,0]; e= P[3,0]; f=P[4,0];
        ellipse_model = lambda x,y : a*x**2 + b*x*y + c*y**2 + d*x + e*y + f

        # threshold 
        ran_sample = np.array([[x,y] for (x,y) in data if np.abs(ellipse_model(x,y)) < offset ])

        if(len(ran_sample) > count_max):
            count_max = len(ran_sample) 
            effective_sample = ran_sample

    return fit_rotated_ellipse(effective_sample)


def fit_rotated_ellipse(data):

    xs = data[:,0].reshape(-1,1) 
    ys = data[:,1].reshape(-1,1)

    J = np.mat( np.hstack((xs*ys,ys**2,xs, ys, np.ones_like(xs,dtype=np.float))) )
    Y = np.mat(-1*xs**2)
    P= (J.T * J).I * J.T * Y

    a = 1.0; b= P[0,0]; c= P[1,0]; d = P[2,0]; e= P[3,0]; f=P[4,0];
    theta = 0.5* np.arctan(b/(a-c))  
    
    cx = (2*c*d - b*e)/(b**2-4*a*c)
    cy = (2*a*e - b*d)/(b**2-4*a*c)

    cu = a*cx**2 + b*cx*cy + c*cy**2 -f
    w= np.sqrt(cu/(a*np.cos(theta)**2 + b* np.cos(theta)*np.sin(theta) + c*np.sin(theta)**2))
    h= np.sqrt(cu/(a*np.sin(theta)**2 - b* np.cos(theta)*np.sin(theta) + c*np.cos(theta)**2))

    ellipse_model = lambda x,y : a*x**2 + b*x*y + c*y**2 + d*x + e*y + f

    error_sum = np.sum([ellipse_model(x,y) for x,y in data])
    #print('fitting error = %.3f' % (error_sum))

    return (cx,cy,w,h,theta)


##### MAIN ######

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(source_video_path)

if isCamera:
  cap = cv2.VideoCapture(0,cv2.CAP_DSHOW) #settings for Vesti video frenzels (live tracking)

if isStream: #save video input
  frame_width = int(cap.get(3))
  frame_height = int(cap.get(4))
  size = (frame_width, frame_height)
  result = cv2.VideoWriter('filename.avi',cv2.VideoWriter_fourcc(*'MJPG'),10, size)  

#open a file in write mode to store x and y coordinates
output_file = open('data/output.csv', 'w', newline='')
writer = csv.writer(output_file)
print('timestamp'+','+ 'xpos' + ','+ 'ypos', file=open('data\output.csv', 'a'))
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
if isCamera:
#Set the resolution
  #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
  #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 576)
  cap.set(cv2.CAP_PROP_CODEC_PIXEL_FORMAT, 0x32595559)
  print('camera ready')
# Start the thread for the plot function
_thread.start_new_thread(plot,())
start_time = datetime.now().timestamp() #used for camera recording timestamp
file_name_suffix = datetime.now().strftime("%Y-%m-%d--%H%M%S") #use when saving csv and video

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  #cv2.imshow("preview",frame)
  if ret == True:
      ret, frame = cap.read()
      kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
      image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      blur = cv2.GaussianBlur(image_gray,(3,3),0)
      ret,thresh1 = cv2.threshold(blur,50,255,cv2.THRESH_BINARY)
      opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
      closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

      image = 255 - closing
      _,contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
      hull = []

      for i in range(len(contours)):
          hull.append(cv2.convexHull(contours[i], False)) 
                        
    #   cnt = sorted(hull, key=cv2.contourArea)
    #   maxcnt = cnt[-1]
      for con in hull:
          approx = cv2.approxPolyDP(con, 0.01 * cv2.arcLength(con,True),True)
          area = cv2.contourArea(con)
          if(len(approx) > 10 and area > 1000):
              cx,cy,w,h,theta = fit_rotated_ellipse_ransac(con.reshape(-1,2))
              #add data to csv
              if isCamera:
                timestamp = datetime.now().timestamp() - start_time
              else:
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)/1000 #get timestamp
              print(str(timestamp)+','+str(cx)+ ','+ str(cy), file=open('data\output.csv', 'a'))
              # draw pupil
              cv2.ellipse(frame,(int(cx),int(cy)),(int(w),int(h)),theta*180.0/np.pi,0.0,360.0,(0,0,255),1)
              cv2.drawMarker(frame, (int(cx),int(cy)),(0, 0, 255),cv2.MARKER_CROSS,2,1)
              cv2.imshow('Output -press Q to Quit',frame)
              
              if isStream:
                result.write(frame)
    # Press Q on keyboard to  exit
      
      if cv2.waitKey(25) & 0xFF == ord('q'):
           break   
  # Break the loop
  else: 
    break


# When everything done, release the video capture object and tidy-up copying and saving file/s to data folders
output_file.close()
graph_csv()

#rename csv file to something memorable (has to be generic intially to avoid passing arguments to plot.py thread)
csv_source = 'data/output.csv'
csv_target = 'data/'+ output_filename + ' ' + file_name_suffix+'.csv'
shutil.copy(csv_source, csv_target)

cap.release() #release capture
if isStream:
  result.release() #release writer
  vid_source = 'filename.avi'
  vid_target = 'video_capture/'+ output_filename + ' ' + file_name_suffix+'.avi'
  shutil.copy(csv_source, csv_target)
  
# Closes all the frames
cv2.destroyAllWindows()
