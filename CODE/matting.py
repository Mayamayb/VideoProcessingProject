import cv2
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm
import matplotlib.pyplot as plt
try:
    import lib._wdt
    fortran_lib = True
except ImportError:
    print("No Fortran modules found, falling back on Python implementation.\nDid you run `python3 setup.py install`?")
    fortran_lib = False
import math
import heapq
from scipy.misc import imread
import matplotlib.pyplot as plt

DIR_STRINGS = ["left", "down", "right", "up"]
DIRS = ((-1, 0), (0, -1), (1, 0), (0, 1))
BW=0.5

def _wdt_python(cost_field):
    """
    See `get_weighted_distance_transform`
    :param cost_field: 2D array
    :return: Weighted distance transform array with same shape as `cost_field`
    """
    nx, ny = cost_field.shape
    # Cost for moving along horizontal lines
    costs_x = np.ones([nx + 1, ny], order='F') * np.inf
    costs_x[1:-1, :] = (cost_field[1:, :] + cost_field[:-1, :]) / 2
    # Cost for moving along vertical lines
    costs_y = np.ones([nx, ny + 1], order='F') * np.inf
    costs_y[:, 1:-1] = (cost_field[:, 1:] + cost_field[:, :-1]) / 2

    # Initialize locations (known/unknown/exit/obstacle)
    weighted_distance_transform = np.ones_like(cost_field, order='F') * np.inf
    exit_locs = np.where(cost_field == 0)
    obstacle_locs = np.where(cost_field == np.inf)
    weighted_distance_transform[exit_locs] = 0

    # Initialize Cell structures
    all_cells = {(i, j) for i in range(nx) for j in range(ny)}
    known_cells = {cell for cell in zip(exit_locs[0], exit_locs[1])}
    unknown_cells = all_cells - known_cells - {cell for cell in zip(obstacle_locs[0], obstacle_locs[1])}
    new_candidate_cells = set()
    for cell in known_cells:
        new_candidate_cells |= _get_new_candidate_cells(cell, unknown_cells)
    cand_heap = [(np.inf, cell) for cell in new_candidate_cells]
    # Loop until all unknown cells have a distance value
    while True:
        # by repeatedly looping over the new candidate cells
        for cell in new_candidate_cells:
            # Compute a distance for each cell based on its neighbour cells
            distance = _propagate_distance(cell, [costs_x, costs_y], weighted_distance_transform)
            # Store this value in the heap (for fast lookup)
            # Don't check whether we have the distance already in the heap; check on outcome
            heapq.heappush(cand_heap, (distance, cell))
        # See if the heap contains a good value and if so, add it to the field. If not, finish.
        # Since we can store multiple distance values for one cell, we might need to pop a couple of times
        while True:
            min_distance, best_cell = heapq.heappop(cand_heap)
            if weighted_distance_transform[best_cell] == np.inf:
                # Got a good one: no assigned distance in wdt yet
                break
            elif min_distance == np.inf:  # No more finite values; done
                return weighted_distance_transform
        # Good value found, add to the wdt and
        weighted_distance_transform[best_cell] = min_distance
        unknown_cells.remove(best_cell)
        new_candidate_cells = _get_new_candidate_cells(best_cell, unknown_cells)


def _exists(index, nx, ny):
    """
    Checks whether an index exists an array
    :param index: 2D index tuple
    :return: true if lower than tuple, false otherwise
    """
    return (0 <= index[0] < nx) and (0 <= index[1] < ny)


def _get_new_candidate_cells(cell, unknown_cells):
    """
    Compute the new candidate cells (cells for which we have no definite distance value yet
    For more information on the algorithm: check fast marching method
    :param cell: tuple of index; a new cell that has been added to the distance field
    :param unknown_cells: set of tuples; all cells still unknown
    :return: Set of new candidate cells for which to compute the distance
    """
    new_candidate_cells = set()
    for direction in DIRS:
        nb_cell = (cell[0] + direction[0], cell[1] + direction[1])
        if nb_cell in unknown_cells:
            new_candidate_cells.add(nb_cell)
    return new_candidate_cells


def _propagate_distance(cell, costs, wdt_field):
    """
    Compute the weighted distance in a cell using costs and distances in other cells
    :param cell: tuple, index of a candidate cell
    :param costs: list of cost arrays in X and Y direction
    :param wdt_field: the weighted distance transform field up until now
    :return: a approximate distance based on the neighbour cells
    """
    nx, ny = wdt_field.shape
    # Find the minimal directions along a grid cell.
    # Assume left and below are best, then overwrite with right and up if they are better
    adjacent_distances = np.ones(4) * np.inf
    pots_from_axis = [0, 0]  # [x direction, y direction]
    costs_from_axis = [np.inf, np.inf]  #
    for i, dir_s in enumerate(DIR_STRINGS):
        # Direction for which we check the cost
        normal = DIRS[i]
        nb_cell = (cell[0] + normal[0], cell[1] + normal[1])
        if not _exists(nb_cell, nx, ny):
            continue
        pot = wdt_field[nb_cell]
        # distance in that neighbour field
        if dir_s == 'left':
            face_index = (nb_cell[0] + 1, nb_cell[1])
        elif dir_s == 'down':
            face_index = (nb_cell[0], nb_cell[1] + 1)
        else:
            face_index = nb_cell
        # Left/right is x, up/down is y
        cost = costs[i % 2][face_index]
        # Proposed cost along this direction
        adjacent_distances[i] = pot + cost
        # If it is cheaper to go from the opposite direction
        if adjacent_distances[i] < adjacent_distances[(i + 2) % 4]:
            pots_from_axis[i % 2] = pot
            costs_from_axis[i % 2] = cost
        hor_pot, ver_pot = pots_from_axis
        hor_cost, ver_cost = costs_from_axis
        # Coefficients of quadratic equation (upwind discretization)
    a = 1. / hor_cost ** 2 + 1. / ver_cost ** 2
    b = -2 * (hor_pot / hor_cost ** 2 + ver_pot / ver_cost ** 2)
    c = (hor_pot / hor_cost) ** 2 + (ver_pot / ver_cost) ** 2 - 1

    D = b ** 2 - 4 * a * c
    # Largest root represents upwind approximation
    x_high = (2 * c) / (-b - math.sqrt(D))
    return x_high


def get_color_image_f(img, mask):
# this function get image and mask and return the get_color_image_ f
    mask_r_f = np.array((img[:,:,0] * (mask>0)).flatten(),np.float64)
    mask_g_f = np.array((img[:,:,1] * (mask>0)).flatten(),np.float64)
    mask_b_f = np.array((img[:,:,2] * (mask>0)).flatten(),np.float64)

    R_kde = sm.nonparametric.KDEUnivariate(mask_r_f)
    R_kde.fit(bw=BW)
    G_kde = sm.nonparametric.KDEUnivariate(mask_g_f)
    G_kde.fit(bw=BW)
    B_kde = sm.nonparametric.KDEUnivariate(mask_b_f)
    B_kde.fit(bw=BW)
    #pdf -
    R_pdf = R_kde.density
    G_pdf = G_kde.density
    B_pdf = B_kde.density

    Ri  = np.array((img[:, :, 0] / 255) * R_pdf.size, dtype=int) - 1
    Gi  = np.array((img[:, :, 1] / 255) * G_pdf.size, dtype=int) - 1
    Bi  = np.array((img[:, :, 2] / 255) * B_pdf.size, dtype=int) - 1

    color_given_f = np.multiply(np.multiply(R_pdf[Ri], G_pdf[Gi]), B_pdf[Bi])

    return color_given_f


    # return [pdfR,pdfG,pdfB]


# gray, green, black = closing_bg,bg_erosion,fg_erosion
def map_image_to_costs(gray,green,black):
    """
    Read image data and convert it to a marginal cost function,
    a 2D array containing costs for moving through each pixel.
    This cost field forms the input for the weighted distance transform
    zero costs denote exits, infinite costs denote fully impenetrable obstacles.
    In this example, we follow Mercurial standards: obstacles are in black, exits in green,
    accessible space is in white, less accessible space has less white.
    Adapt to your own needs.
    :param image: String of image file or open file descriptor of image
    :return: 2D array representing the cost field
    """
    obstacles = np.where(black > 0)
    exits = np.where(green>0)
    # Boolean index array for places without exits and obstacles
    space = np.ones(gray.shape, dtype=np.bool)
    space[obstacles] = False
    space[exits] = False
    # Cost field: Inversely proportional to uin8 grayscale values
    cost_field = np.empty(gray.shape)
    cost_field[obstacles] = np.inf
    cost_field[exits] = 0
    cost_field[space] = gray[space]+1
    return cost_field

def get_weighted_distance_transform(cost_field):
    """
    Compute the weighted distance transform from the cost field using a fast marching algorithm.
    We compute the distance transform with costs defined on a staggered grid for consistency.
    This means that we use costs defined on the faces of cells, found by averaging the values of the two adjacent cells.

    Starting from the exit, we march over all the pixels with the lowest weighted distance iteratively,
    until we found values for all pixels in reach.
    :param cost_field: nonnegative 2D array with cost in each cell/pixel, zero and infinity are allowed values.
    :return: weighted distance transform field
    """
    if fortran_lib:
        # Fortran does not allow for infinite float.
        nx, ny = cost_field.shape
        # Float that is (probably far) higher than the highest reachable potential
        obstacle_value = np.max(cost_field[cost_field < np.inf]) * nx * ny * 2

        cost_field[cost_field == np.inf] = obstacle_value
        # Run the Fortran module
        wdt_field = lib._wdt.weighted_distance_transform(cost_field, nx, ny, obstacle_value)
        wdt_field[wdt_field >= obstacle_value] = np.inf
        return wdt_field
    else:
        # Run python implementation
        return _wdt_python(cost_field)


def matte(frame_img, trimap,i=0):
    # this function get frame image and mask and retrun alpha

    image_bg = trimap.copy()
    image_fg = trimap.copy()
    image_bg = cv2.cvtColor(image_bg,cv2.COLOR_RGB2GRAY)
    image_fg = cv2.cvtColor(image_fg,cv2.COLOR_RGB2GRAY)
    image_fg[np.where(image_fg>100)]=255
    image_bg[np.where(image_bg<=100)]=255
    image_fg[np.where(image_fg<=100)]=0
    image_bg[np.where(image_bg<255)]=0


    #calculate scribble
    kernel = np.ones((5, 5), np.uint8)
    bg_erosion = cv2.erode(image_bg, kernel, iterations=3)
    fg_erosion = cv2.erode(image_fg, kernel, iterations=3)

    #calculate map of probabilities
    color_given_fg = get_color_image_f(frame_img, image_fg)
    color_given_bg = get_color_image_f(frame_img, image_fg)

    Pf= color_given_fg/(color_given_bg+color_given_fg)
    Pb= color_given_bg/(color_given_bg+color_given_fg)

    #calculate derivative
    laplacian_Pf = np.abs(cv2.Laplacian(Pf,cv2.CV_64F))
    laplacian_Pb = np.abs(cv2.Laplacian(Pb,cv2.CV_64F))
    closing_fg = cv2.morphologyEx(laplacian_Pf, cv2.MORPH_CLOSE, kernel)
    closing_bg = cv2.morphologyEx(laplacian_Pb, cv2.MORPH_CLOSE, kernel)

    # calculate cost probabilities
    cost_Pf = map_image_to_costs(closing_fg,fg_erosion,bg_erosion)
    cost_Pb = map_image_to_costs(closing_bg,bg_erosion,fg_erosion)
# calculate distances map
    distance_transform_fg = get_weighted_distance_transform(cost_Pf)
    distance_transform_bg = get_weighted_distance_transform(cost_Pb)
    # plt.imshow(distance_transform_fg)
    # plt.savefig('/home/pihash/Desktop/PROJECT/final_code/distance_transform_bg{}.png'.format(i))
    # plt.close()
    # plt.imshow(distance_transform_bg)
    # plt.savefig('/home/pihash/Desktop/PROJECT/final_code/distance_transform_fg{}.png'.format(i))
    # plt.close()
    r=2
    cost_Pf[cost_Pf==np.inf]=1e18
    cost_Pb[cost_Pb==np.inf]=1e18
    Wf=((distance_transform_fg+1e-10)**-r)*cost_Pf
    Wb=((distance_transform_bg+1e-10)**-r)*cost_Pb

    fg_erosion1 = cv2.dilate(fg_erosion, kernel, iterations=1)
    #calculate alpha

    alpha = np.minimum(Wf/(Wf+Wb+1e-17)+np.array(fg_erosion1>0,'uint8'),1)
    return alpha
    # plt.imshow(alpha)

def matted(video_img_name_input = '../stabilized.avi',\
           video_trimap_name_input = '../binary.avi',\
           background_img_name_input ='../background.jpg',
           video_name_output = '../matted.avi'):
    try:
        cap_img = cv2.VideoCapture(video_img_name_input)
    except (IOError ,FileNotFoundError):
        print('Stabilization video file doesn\'t exist ')
        raise IOError('Stabilization video file doesn\'t exist ')
    try:
        cap_trimap = cv2.VideoCapture(video_trimap_name_input)
    except (IOError, FileNotFoundError):
        print('Binary video file doesn\'t exist')
        raise IOError('Binary video file doesn\'t exist')
    length_vid_input = int(cap_trimap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap_trimap.get(cv2.CAP_PROP_FPS)
    ret_img, frame_img = cap_img.read()
    ret_trimap, frame_trimap = cap_trimap.read()
    height, width, layers = frame_img.shape

    try:
        background_img = cv2.resize(cv2.imread(background_img_name_input),(int(width),int(height)))
    except (IOError, FileNotFoundError):
        print('Background file doesn\'t exist')
        raise IOError('Background file doesn\'t exist')

    video = cv2.VideoWriter(video_name_output, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))


    pbar = tqdm(total=length_vid_input, desc='matted function')
    # i=1
    while (1):


        # For eche alpha, do color blending
        if ret_trimap == True:

            # alpha = matte(frame_img, frame_trimap,i)
            alpha = matte(frame_img, frame_trimap)


            alpha_3ch = np.stack([alpha for _ in range(3)], axis=2)
            frame_matted = np.asarray((1-alpha_3ch)*background_img+(alpha_3ch*frame_img),dtype='uint8')
            # cv2.imwrite('/home/pihash/Desktop/PROJECT/final_code/new_source{}.png'.format(i), frame_matted)
            # cv2.imwrite('/home/pihash/Desktop/PROJECT/final_code/new_alpha{}.png'.format(i), alpha_3ch*255)
            # i+=1
            video.write(frame_matted)
            pbar.update(1)

            ret_img, frame_img = cap_img.read()
            ret_trimap, frame_trimap = cap_trimap.read()


        else:
            break
    pbar.update(1)
    pbar.close()
