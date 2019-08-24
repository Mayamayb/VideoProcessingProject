from tkinter import Tk, Label, Button
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from tkinter import *
from PIL import ImageTk,Image
import numpy as np
# import functions
from matting import *
from tracking_and_roi import *
from BG_subtraction import *
from stabilization import *


"""
Example for a Graphical User Interface.
Using tkinter library.
"""




def is_image(file_name):
    ret_val = False
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        ret_val = True
    else:
        print("The file {} is not an image file! Please select a valid file".format(file_name))
    return ret_val

def is_video(file_name):
    ret_val = False
    if file_name.lower().endswith(('.avi', '.mp4')):
        ret_val = True
    else:
        print("The file {} is not a video file! Please select a valid file".format(file_name))
    return ret_val


class ExampleGUI:
    def __init__(self, master_window):
        self.parname =self.get_parname()
        self.master_window = master_window
        master_window.title("Video Processing Project")
        self.filename = {}
        self.filename['Input'] = os.path.join(self.parname,'Input','INPUT.avi')
        self.filename['Background'] = os.path.join(self.parname, 'Input', 'background.jpg')
        self.crop_image_roi = [510, 78, 71, 252]


        self.get_image_label_Input = Label(master_window, text="Get files:").pack()
        self.get_image_path_button_Input = Button(master_window, text="Browse for Input video", command=lambda: self.select_image_video_via_browser('Input'), height=1, width=30, anchor=W).pack()
        self.get_image_path_button_Stabilization = Button(master_window, text="Browse for Stabilized video", command=lambda: self.select_image_video_via_browser('Stabilization'), height=1, width=30, anchor=W).pack()
        self.get_image_path_button_Binary = Button(master_window, text="Browse for Binary video", command=lambda: self.select_image_video_via_browser('Binary'), height=1, width=30, anchor=W).pack()
        self.get_image_path_button_Extracted = Button(master_window, text="Browse for Extracted video", command=lambda: self.select_image_video_via_browser('Extracted'), height=1, width=30, anchor=W).pack()
        self.get_image_path_button_background = Button(master_window, text="Browse for background image", command=lambda: self.select_image_video_via_browser('Background'), height=1, width=30,anchor=W).pack()
        self.get_image_path_button_Matted = Button(master_window, text="Browse for Matted video", command=lambda: self.select_image_video_via_browser('Matted'), height=1, width=30,anchor=W).pack()


        self.show_image_label_Input = Label(master_window, text="View Video/Image:").pack()
        self.show_image_button_Input = Button(master_window, text="Show Input Video", command=lambda: self.show_video('Input'), height=1, width=30,anchor=W).pack()
        self.show_image_button_Stabilization = Button(master_window, text="Show Stabilization Video", command=lambda: self.show_video('Stabilization'), height=1, width=30,anchor=W).pack()
        self.show_image_button_Binary = Button(master_window, text="Show Binary Video", command=lambda: self.show_video('Binary'), height=1, width=30,anchor=W).pack()
        self.show_image_button_Extracted = Button(master_window, text="Show Extracted Video", command=lambda: self.show_video('Extracted'), height=1, width=30,anchor=W).pack()
        self.show_image_button_background = Button(master_window, text="Show Background Image", command=lambda: self.show_image('Background'), height=1, width=30,anchor=W).pack()
        self.show_image_button_Matted = Button(master_window, text="Show Matted Video", command=lambda: self.show_video('Matted'), height=1, width=30,anchor=W).pack()
        self.show_image_button_Output = Button(master_window, text="Show Output Video", command=lambda: self.show_video('Output'), height=1, width=30,anchor=W).pack()


        self.view_selected_region_label = Label(master_window, text='Function').pack()
        self.view_selected_region_button_Stabilization = Button(master_window, text="Run Stabilization function", command=self.Stabilization, height=1, width=30,anchor=W).pack()
        self.view_selected_region_button_background_subtraction = Button(master_window, text="Run Background Subtracion function",command=self.background_subtraction, height=1, width=30,anchor=W).pack()
        self.view_selected_region_button_Matted = Button(master_window, text="Run Matting function",command=self.matted, height=1, width=30,anchor=W).pack()
        self.view_selected_region_button_tracking = Button(master_window, text="Run to select manual ROI",command=self.roi_sign, height=1, width=30,anchor=W).pack()
        self.view_selected_region_button_tracking = Button(master_window, text="Run Tracking function",command=self.create_tracking, height=1, width=30,anchor=W).pack()
        self.view_selected_region_button_run_all = Button(master_window, text="Run the Full Algorithm",command=self.run_all, height=1, width=30,anchor=W).pack()

    """
    Definition of functions that would be called in a case of a button click
    """
    def get_parname(self):
        dirname = os.path.dirname(__file__)
        return os.path.dirname(dirname)

    def select_image_video_via_browser(self, name):
        self.filename[name] = askopenfilename()
        print(name)
        print(self.filename[name])

    def show_image(self,name):
        if name in self.filename.keys():
            if is_image(self.filename[name]):
                image = cv2.imread(self.filename[name])
                name_win = 'For exit press q'
                cv2.namedWindow(name_win, cv2.WINDOW_NORMAL)
                # cv2.resizeWindow(name_win, image.shape[0], image.shape[1])
                print('For exit press q')
                while True:
                    try:

                        cv2.imshow(name_win, image)
                        k = cv2.waitKey(1) & 0xFF
                        if k == ord('q'):
                            break
                    except KeyboardInterrupt:
                        break
                cv2.destroyAllWindows()
        else:
            print('File for {} doesn\'t exist'.format(name))

    def show_video(self, name):
        if name in self.filename.keys():
            if is_video(self.filename[name]):
                cap = cv2.VideoCapture(self.filename[name])
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                wait_time = int(np.ceil(cap.get(cv2.CAP_PROP_FPS)))
                # print(wait_time)
                while (cap.isOpened()):
                    ret, frame = cap.read()
                    if ret==True:
                        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        cv2.namedWindow('For exit press q', cv2.WINDOW_NORMAL)
                        cv2.resizeWindow('For exit press q', w, h)
                        cv2.imshow('For exit press q', frame)
                        # cv2.waitKey(1000//wait_time)
                        if cv2.waitKey(1000//wait_time) & 0xFF == ord('q'):
                            break
                    else:
                        break

                cap.release()
                cv2.destroyAllWindows()
        else:
            print('File for {} doesn\'t exist'.format(name))

    def select_interest_region_from_matted(self, name):
        # name = 'Matted'
        if name in self.filename.keys():
            if is_video(self.filename[name]):
                cap = cv2.VideoCapture(self.filename[name])
                ret, image = cap.read()
                # Select ROI
                from_center = False
                cv2.namedWindow("Drag the rect from the top left to the bottom right corner of the forground object,"
                                " then press ENTER.", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Drag the rect from the top left to the bottom right corner of the forground object,"
                                 " then press ENTER.", image.shape[1], image.shape[0])
                roi = cv2.selectROI("Drag the rect from the top left to the bottom right corner of the forground object,"
                                    " then press ENTER.",
                                    image, from_center)
                # Crop image
                self.crop_image_roi = roi
                print('new ROI:',roi)

                cv2.destroyAllWindows()
                cv2.waitKey(0)
                cap.release()
        else:
            print('File {} doesn\'t exist'.format(name))

    def view_selected_region(self):
        # Display cropped image
        name = 'Matted'
        if name in self.filename.keys():

            if self.crop_image_roi:
                if is_video(self.filename[name]):
                    cap = cv2.VideoCapture(self.filename[name])
                    ret, image = cap.read()
                    crop_image = image[int(self.crop_image_roi[1]):int(self.crop_image_roi[1]+self.crop_image_roi[3]),
                                       int(self.crop_image_roi[0]):int(self.crop_image_roi[0]+self.crop_image_roi[2])]
                    cv2.imshow("Selected Part", crop_image)
                    cv2.waitKey(0)
            else:
                print('Region of interest was not selected')
        else:
            print('File doesn\'t exist')


    def Stabilization(self):
        self.filename['Stabilization'] = os.path.join(self.parname, 'Output', 'stabilized.avi')
        try:
            stabilize(video_name_input=self.filename['Input'],
                   video_name_output=self.filename['Stabilization'])
        except KeyError as e:
            print('problem with {} file check it'.format(e))
        except IOError as e:
            print('{}'.format(e))
        print('Stabilization done! ')

    def background_subtraction(self):
        self.filename['Binary'] = os.path.join(self.parname, 'Output', 'binary.avi')
        self.filename['Extracted'] = os.path.join(self.parname, 'Output', 'extracted.avi')
        try:
            BG_subtraction(video_name_input=self.filename['Stabilization'],
                           video_name_output=self.filename['Binary'],
                           video_name_extract=self.filename['Extracted'],
                           BGth=30, kopen=7, kclose=7)
        except KeyError as e:
            print('problem with {} file check it'.format(e))
        except IOError as e:
            print('{}'.format(e))
        print('Background Subtraction done!')


    def matted(self):
        self.filename['Matted'] = os.path.join(self.parname, 'Output', 'matted.avi')
        try:
            matted(video_img_name_input = self.filename['Stabilization'],
                    video_trimap_name_input = self.filename['Binary'],
                    background_img_name_input = self.filename['Background'],
                    video_name_output = self.filename['Matted'])
        except KeyError as e:
            print('problem with {} file check it'.format(e))
        except IOError as e:
            print('{}'.format(e))
        print('Done Matting!')


    def roi_sign(self):
        try:
            self.select_interest_region_from_matted('Matted')
        except:
            print('Problem in ROI selection function')
        print('Done selecting manual ROI!')

    def create_tracking(self):
        self.filename['Output'] = os.path.join(self.parname, 'Output', 'Output.avi')
        try:
            # self.select_interest_region_from_matted('Matted')
            create_tracking(self.crop_image_roi,
                            video_name_input = self.filename['Matted'],
                            video_name_output = self.filename['Output'])
        except:
            print('Problem in create_tracking function')
        print('Done Tracking!')

    def run_all(self):
        self.Stabilization()
        self.background_subtraction()
        self.matted()
        self.create_tracking()



if __name__ == "__main__":
    root = Tk()
    root.geometry("400x600")  # You want the size of the app to be 500x500
    example_gui = ExampleGUI(root)
    root.mainloop()




