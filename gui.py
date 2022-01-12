### Code has been adapted from
### https://codingshiksha.com/python/python-3-tkinter-webcam-video-recorder-selfie-capture-as-png-image-using-opencv-pillow-library-gui-desktop-app-full-project-for-beginners/

import tkinter as tk
from tkinter.constants import LEFT, RIGHT
import PIL.Image, PIL.ImageTk
import os
import cv2
import time
import datetime as dt
import argparse
from detect import inn3_detector

#rootWindow = tk.Tk()
#rootWindow.title("INN3 Python Detector")
#rootWindow.geometry('500x500')

# create a label and position it
#lbl = tk.Label(rootWindow, text="Hello")
#lbl.grid(column=0, row=0)

#rootWindow.mainloop()

class App:
    def __init__(self, rootWindow, windowTitle, videoSource = 0):
        self.rootWindow = rootWindow
        self.rootWindow.attributes('-fullscreen',True)
        print(self.rootWindow.winfo_reqheight())
        print(self.rootWindow.winfo_reqwidth())
        self.rootWindow.title(windowTitle)
        self.videoSource = videoSource
        self.ok = False
        # btn click flags
        self.emotion = False
        self.age = False
        self.gender = False

        # open ghe video source for testing
        #self.video = VideoCapture(self.videoSource)
        # create canvas for video source
        self.tkCanvas = tk.Canvas(rootWindow, width = , height = 480)
        self.tkCanvas.pack()


        # add Buttons to the window
        self.btnAge = tk.Button(rootWindow, text='Age Detection', fg = "black", width = 15, command = self.ageDetection)
        self.btnAge.pack(side = LEFT)
        self.btnGender = tk.Button(rootWindow, text='Gender Detection', fg = "black", width = 15, command = self.genderDetection)
        self.btnGender.pack(side = LEFT)
        self.btnEmotion = tk.Button(rootWindow, text='Emotion Detection', fg = "black", width = 15, command = self.emotionDetection)
        self.btnEmotion.pack(side = LEFT)
        self.btnExit = tk.Button(rootWindow, text='Close', width = 15, command=quit)
        self.btnExit.pack(side = LEFT)
        self.idet = inn3_detector()
        self.idet.start_pred()
        self.idet.start_threads()
        time.sleep(1)
        self.delay = 1
        self.update()
        self.rootWindow.mainloop()

    # define functions for the buttons
    def ageDetection(self):
        if self.age == True:
            self.age = False
        else:
            self.age = True

        if self.age == True:
            self.btnAge.configure(fg="green")
        else:
            self.btnAge.configure(fg="black")

        self.idet.set_agedet(self.age)

        pass
    
    def genderDetection(self):
        if self.gender == True:
            self.gender = False
        else:
            self.gender = True

        if self.gender == True:
            self.btnGender.configure(fg="green")
        else:
            self.btnGender.configure(fg="black")

        self.idet.set_gendet(self.gender)

        pass

    def emotionDetection(self):
        if self.emotion == True:
            self.emotion = False
        else:
            self.emotion = True

        if self.emotion == True:
            self.btnEmotion.configure(fg="green")
        else:
            self.btnEmotion.configure(fg="black")

        self.idet.set_emodet(self.emotion)

        pass

    def update(self):
        # Get a frame from the video source
        frame = self.idet.get_currentpic()
        if frame.size != 0:
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            im = PIL.Image.fromarray(frame)
            self.photo = PIL.ImageTk.PhotoImage(im)
            self.tkCanvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
        self.rootWindow.after(self.delay,self.update)

class VideoCapture:
    def __init__(self, videoSource = 0):
        # Open the video source
        self.video = cv2.VideoCapture(videoSource)
        if not self.video.isOpened():
            raise ValueError("Unable to open video source", videoSource)

        # Command Line Parser
        args=CommandLineParser().args
        
        #create videowriter

        # 1. Video Type
        VIDEO_TYPE = {
            'avi': cv2.VideoWriter_fourcc(*'XVID'),
            #'mp4': cv2.VideoWriter_fourcc(*'H264'),
            'mp4': cv2.VideoWriter_fourcc(*'XVID'),
        }

        self.fourcc=VIDEO_TYPE[args.type[0]]

        # 2. Video Dimension
        STD_DIMENSIONS =  {
            '480p': (640, 480),
            '720p': (1280, 720),
            '1080p': (1920, 1080),
            '4k': (3840, 2160),
        }
        res=STD_DIMENSIONS[args.res[0]]
        print(args.name,self.fourcc,res)
        self.out = cv2.VideoWriter(args.name[0]+'.'+args.type[0],self.fourcc,10,res)

        #set video sourec width and height
        self.video.set(3,res[0])
        self.video.set(4,res[1])

        # Get video source width and height
        self.width,self.height=res
        #self.width.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        #self.height.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # To get frames
    def get_frame(self):
        if self.video.isOpened():
            ret, frame = self.video.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            else:
                return (ret, None)
        else:
            return (ret, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.video.isOpened():
            self.video.release()
            self.out.release()
            cv2.destroyAllWindows()


class CommandLineParser:
    
    def __init__(self):

        # Create object of the Argument Parser
        parser=argparse.ArgumentParser(description='Script to record videos')

        # Create a group for requirement 
        # for now no required arguments 
        # required_arguments=parser.add_argument_group('Required command line arguments')

        # Only values is supporting for the tag --type. So nargs will be '1' to get
        parser.add_argument('--type', nargs=1, default=['avi'], type=str, help='Type of the video output: for now we have only AVI & MP4')

        # Only one values are going to accept for the tag --res. So nargs will be '1'
        parser.add_argument('--res', nargs=1, default=['1080p'], type=str, help='Resolution of the video output: for now we have 480p, 720p, 1080p & 4k')

        # Only one values are going to accept for the tag --name. So nargs will be '1'
        parser.add_argument('--name', nargs=1, default=['output'], type=str, help='Enter Output video title/name')

        # Parse the arguments and get all the values in the form of namespace.
        # Here args is of namespace and values will be accessed through tag names
        self.args = parser.parse_args()

def main():
    App(tk.Tk(), 'InnoLab Python Detector')

main()