import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from tqdm import tqdm
# from HW3_functions import *
import os

def is_image(file_name):
    ret_val = False
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        ret_val = True
    else:
        print("The file {} is not an image file! Please select a valid file".format(file_name))
    return ret_val

def select_interest_region_im(image):

    # Select ROI
    from_center = False
    roi = cv2.selectROI("Drag the rect from the top left to the bottom right corner of the forground object,"
                        " then press ENTER.",
                        image, from_center)
    # Crop image
    cv2.destroyAllWindows()
    cv2.waitKey(0)

    return {'roi':roi,'x':roi[0]+roi[2]//2,'y':roi[1]+roi[3]//2,'w':roi[2],'h':roi[3]}


def compNormHist(I, s):
    """INPUT  = I (image) AND s (1x6 STATE VECTOR, CAN ALSO BE ONE COLUMN FROM S)
    OUTPUT = normHist (NORMALIZED HISTOGRAM 16x16x16 SPREAD OUT AS A 4096x1
                VECTOR. NORMALIZED = SUM OF TOTAL ELEMENTS IN THE HISTOGRAM = 1) """
    numBins = 16

    cX = s[0]
    cY = s[1]
    hW = s[2]
    hH = s[3]

    [sizeY, sizeX] = I.shape[:-1]

    xMin = round(max(cX - hW, 0))
    xMax = round(min(cX + hW, sizeX))
    yMin = round(max(cY - hH, 0))
    yMax = round(min(cY + hH, sizeY))

    ROI = I[yMin:yMax, xMin:xMax, :]
    quantizedROI = np.round(ROI*(numBins/255))

    quantizedR = quantizedROI[:,:, 0] *(numBins ** 2)
    quantizedG = quantizedROI[:,:, 1] *numBins
    quantizedB = quantizedROI[:,:, 2]

    quantizedTotal = quantizedR + quantizedG + quantizedB

    normHist, bins = np.histogram([quantizedTotal[:]], bins=np.arange(numBins**3+1), density=True)
    return np.expand_dims(normHist, axis=1)

def predictParticles(S_next_tag):
    """INPUT  = S_next_tag (previously sampled particles)
    OUTPUT = S_next (predicted particles. weights and CDF not updated yet) """
    S_next = np.zeros(S_next_tag.shape)

    # Generate white Gaussian noise
    std_dev = 0.6
    n = std_dev * np.random.randn(2, S_next_tag.shape[1])
    # Update velocity
    S_next[4, :] = S_next_tag[4, :] + n[0, :]
    S_next[5, :] = S_next_tag[5, :] + n[1, :]

    # Update center location
    S_next[0, :] = S_next_tag[0, :] + S_next[4, :]
    S_next[1, :] = S_next_tag[1, :] + S_next[5, :]

    # Tracking window width and height has not changed
    S_next[2, :] = S_next_tag[2, :]
    S_next[3, :] = S_next_tag[3, :]
    return S_next
def compBatDist(p, q):
    """INPUT  = p , q (2 NORMALIZED HISTOGRAM VECTORS SIZED 4096x1)
    OUTPUT = THE BHATTACHARYYA DISTANCE BETWEEN p AND q (1x1)
    IMPORTANT - YOU WILL USE THIS FUNCTION TO UPDATE THE INDIVIDUAL WEIGHTS
    OF EACH PARTICLE. AFTER YOU'RE DONE WITH THIS YOU WILL NEED TO COMPUTE
    THE 100 NORMALIZED WEIGHTS WHICH WILL RESIDE IN VECTOR W (1x100)
    AND THE CDF (CUMULATIVE DISTRIBUTION FUNCTION, C. SIZED 1x100)
    NORMALIZING 100 WEIGHTS MEANS THAT THE SUM OF 100 WEIGHTS = 1 """
    w = np.exp(20 * np.matmul(np.sqrt(p).T, np.sqrt(q)))
    return w


def sampleParticles(S_prev, C):
    """INPUT  = S_prev (PREVIOUS STATE VECTOR MATRIX), C (CDF)
    OUTPUT = S_next_tag (NEW X STATE VECTOR MATRIX) """
    S_next_tag = np.zeros(S_prev.shape)

    for i in range(S_prev.shape[1]):
        r = np.random.uniform(0, 1)
        j = np.where(C >= r)[0][0]
        S_next_tag[:, i] = S_prev[:, j]
    return S_next_tag


def compNormWeights(I, S, q):
    N = S.shape[1]
    W = np.zeros((1, N))

    for i in range(N):
        p = compNormHist(I, S[:, i].tolist())
        W[:,i] = compBatDist(p, q)
    W = W / np.sum(W)
    return W


def get_x_y_w_h(I,S,W):
    hW = S[2, 0]
    hH = S[3, 0]
    # calculate average weight rectangle
    cX = np.matmul(S[0,:], W.T)
    cY = np.matmul(S[1,:], W.T)
    x = int(np.round(np.max((cX - hW, 0))))
    w = int(np.round(np.min((hW * 2, I.shape[1] - x))))
    y = int(np.round(np.max((cY - hH, 0))))
    h = int(np.round(np.min((hH * 2, I.shape[0] - y))))
    return x ,y ,w, h


def create_tracking(roi_in, video_name_input = 'matted.avi',video_name_output = 'video_out_tracking.avi',):

    cnt =-1
    # SET NUMBER OF PARTICLES
    N = 100

    cap = cv2.VideoCapture(video_name_input)
    length_vid_input = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # take first frame of the video
    ret,I = cap.read()
    height, width, layers = I.shape
    # video = cv2.VideoWriter(video_name, 0, 1, (width,height))
    video = cv2.VideoWriter(video_name_output,cv2.VideoWriter_fourcc(*'DIVX'), fps, (width,height))
    # LOAD FIRST IMAGE
    roi  = {'roi':roi_in,'x':roi_in[0]+roi_in[2]//2,'y':roi_in[1]+roi_in[3]//2,'w':roi_in[2],'h':roi_in[3]}

    # Initial Settings
    s_initial = [roi['x'],    # x center
                 roi['y'],    # y center
                 roi['w']//2,    # half width
                 roi['h']//2,    # half height
                   0,    # velocity x
                   0]    # velocity y

    # CREATE INITIAL PARTICLE MATRIX 'S' (SIZE 6xN)

    S = predictParticles((s_initial * np.ones((N,1))).T)


    # COMPUTE NORMALIZED HISTOGRAM
    q = compNormHist(I, s_initial)

    # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
    # YOU NEED TO FILL THIS PART WITH CODE:

    W = compNormWeights(I, S, q)
    C = np.cumsum(W)

    # MAIN TRACKING LOOP
    pbar = tqdm(total=length_vid_input, desc= 'tracking function')

    while(1):
        cnt+=1

        ret, I = cap.read()

        if ret == True:
            # find new tracking every 3 frames for efficiency reasons
            # (fps is high and tracking person walks slow enough).
            if cnt % 2 == 0:
                S_prev = S
                # SAMPLE THE CURRENT PARTICLE FILTERS
                try:
                    S_next_tag = sampleParticles(S_prev, C)
                except:
                    print(':-)')

                # PREDICT THE NEXT PARTICLE FILTERS (YOU MAY ADD NOISE)
                S = predictParticles(S_next_tag)

                # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
                W = compNormWeights(I, S, q)
                C = np.cumsum(W)
                # calculate average estimation
                x, y, w, h, = get_x_y_w_h(I,S, W)
            # draw bounding box
            img2 = cv2.rectangle(I, (x,y), (x+w,y+h), 255,1)
            video.write(img2)
            pbar.update(1)

        else:
            break
    pbar.update(1)
    pbar.close()

if __name__ == '__main__':
    create_tracking(roi, video_name_input = 'matted.avi',video_name_output = 'video_out_tracing.avi')