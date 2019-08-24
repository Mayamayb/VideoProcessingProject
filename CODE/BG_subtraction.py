
import numpy as np
import cv2
import os
from tqdm import tqdm

def BG_subtraction(video_name_input = 'stabilized.avi',video_name_output = 'binary.avi',video_name_extract = 'extracted.avi',  BGth = 30, kopen=11, kclose=11):

    cap = cv2.VideoCapture(video_name_input)

    # Get frame count
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get width and height of video stream
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec for output video
    out_mask = cv2.VideoWriter(video_name_output, cv2.VideoWriter_fourcc(*'DIVX'), fps, (w, h))
    out_extract = cv2.VideoWriter(video_name_extract, cv2.VideoWriter_fourcc(*'DIVX'), fps, (w, h))

    medstack = []
    # save frames for median calculation
    for i in range(n_frames - 1):
        success, frame = cap.read()
        if not success:
            break
        medstack.append(frame)

    cap = cv2.VideoCapture(video_name_input)

    # CALCULATE the median filter for all frames
    medst = np.uint8(np.median(medstack, axis=0))
    pbar = tqdm(total=n_frames - 1, desc='Background Subtraction')
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kopen, kopen))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kclose, kclose))

    for i in range(n_frames - 1):
        success, frame = cap.read()
        if not success:
            break
        # calculate absalute difference between current frame and median frame
        difframe = cv2.absdiff(frame, medst)
        # convert to gray scale
        gray = cv2.cvtColor(difframe,cv2.COLOR_BGR2GRAY)
        # apply binary threshold
        th ,binary = cv2.threshold(gray, BGth, 255, cv2.THRESH_BINARY )

        # apply morphological operations
        # cv2.imwrite('frame_{}_before_morph.png'.format(i), binary)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
        # cv2.imwrite('frame_{}_after_morph_opn{}_cls{}.png'.format(i, kopen,kclose), binary)

        # find largest connected component --> object
        new_img = np.zeros_like(binary)
        labels, stats = cv2.connectedComponentsWithStats(binary, cv2.CV_32S)[1:3]
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        new_img[labels == largest_label] = 255  #object recieves the value 255
        # cv2.imwrite('frame_{}_after_CC.png'.format(i), new_img)

        # convert back to BGR for saving output frame
        mask = cv2.cvtColor(new_img,cv2.COLOR_GRAY2BGR)
        out_mask.write(mask)
        # extract object video
        extracted = np.multiply((mask//255), frame)
        out_extract.write(extracted)
        # cv2.imwrite('frame_{}_extracted.png'.format(i), extracted)

        pbar.update(1)
    pbar.close()
    print('Saved background subtraction to ',video_name_output)
