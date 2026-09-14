import os, argparse
from datetime import datetime
import numpy as np
from timeit import default_timer as timer
import skvideo.io
from pathlib import Path
import skimage.transform as transform
from skimage.morphology import skeletonize
from skimage import exposure
from scipy.io import loadmat
import time
import matplotlib.patches as mpatches
try:
    import scipy.fft as fft
except:
    import scipy.fftpack as fft
from scipy.stats import ks_2samp, kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import scipy.signal as signal
import glob
import matplotlib.pyplot as pl
from matplotlib.widgets import EllipseSelector
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label as sk_label, regionprops
from scipy.ndimage import distance_transform_edt
from datetime import datetime
from skimage.measure import label as sk_label, regionprops
from scipy.ndimage import gaussian_filter, median_filter, binary_opening, binary_closing, binary_dilation, label, \
    laplace
from configparser import ConfigParser
import math
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.ion()  # Turn on interactive mode
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as pl
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter, median_filter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib import patches




def fit_circle(xy):
    # xy is a list of 4 points, where each point is (x, y)
    x = xy[:, 0]
    y = xy[:, 1]

    # Function to calculate the algebraic distance between points and a circle
    def calc_R(xc, yc):
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    # Function to minimize (to find the best circle)
    def fun(c):
        Ri = calc_R(c[0], c[1])
        return np.sum((Ri - Ri.mean()) ** 2)

    # Initial guess is the center of mass of the points
    x_m = np.mean(x)
    y_m = np.mean(y)
    result = minimize(fun, (x_m, y_m), method='Nelder-Mead', tol=1e-6)

    xc, yc = result.x
    Ri = calc_R(xc, yc)
    R = Ri.mean()  # the radius is the mean distance to the center

    return xc, yc, R


def calculate_animal_age(dob, imaging_date):
    # Convert input strings to date objects
    dob = datetime.strptime(dob, "%Y%m%d")
    imaging_date = datetime.strptime(imaging_date, "%Y%m%d")

    # Calculate the difference between the dates in days
    age_in_days = (imaging_date - dob).days

    # Calculate weeks and remaining days
    weeks = age_in_days // 7
    days = age_in_days % 7

    return age_in_days


# Create the GUI for clicking 4 points
class CircleFittingGUI:
    def __init__(self, data):
        self.data = data
        self.points = []
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.data, cmap='jet')
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.circle_params = None

    def onclick(self, event):
        # Collect points on click (limit to 4 points)
        if len(self.points) < 4:
            ix, iy = event.xdata, event.ydata
            self.points.append([ix, iy])
            self.ax.plot(ix, iy, 'ro')  # mark the clicked point
            self.fig.canvas.draw()

        if len(self.points) == 4:
            # Fit the circle to the selected points
            self.points = np.array(self.points)
            xc, yc, R = fit_circle(self.points)
            self.circle_params = (xc, yc, R)

            # Draw the fitted circle on the plot
            circle = plt.Circle((xc, yc), R, color='blue', fill=False, linewidth=2)
            self.ax.add_artist(circle)
            self.fig.canvas.draw()

            # Create a mask for the circle
            self.create_mask(xc, yc, R)

    def create_mask(self, xc, yc, R):
        # Generate mask for the region inside the circle
        y, x = np.ogrid[:self.data.shape[0], :self.data.shape[1]]
        mask = (x - xc) ** 2 + (y - yc) ** 2 <= R ** 2

        # Apply the mask to the data
        masked_data = np.ma.masked_where(~mask, self.data)

        # Plot the masked image
        plt.figure()
        plt.imshow(masked_data, cmap='jet')
        plt.title('Masked Image')
        plt.colorbar()
        plt.show()

    def show(self):
        plt.show()

def load_data(fn, start_frame=0, end_frame=0, resize=None):
    '''Load the video file specified in path.
       File should be in .mj2 format.
       Uses ffmpeg backend and scikit-video to load video file.

       INPUT
       fn           : Filepath to video file (.mj2 format)
       start_frame  : First frame to read (count from 0)
       end_frame    : Last frame to read
       resize       : Factor to resize image to. Skip if None

       OUTPUT
       vid          : Video file as numpy.array (frames x dim1 x dim2)
    '''

    # Create input and output parameters for ffmpeg
    inputparameters = {'-ss': '%s' % (start_frame)}  # -ss seeks to the position/frame in video file
    outputparameters = {'-pix_fmt': 'gray16be'}  # specify to import as uint16, otherwise it's uint8
    if end_frame == 0:
        num_frames = 0
    else:
        num_frames = end_frame - start_frame

    # # Import video file as numpy.array
    vidreader = skvideo.io.vreader(fn, inputdict=inputparameters, outputdict=outputparameters, num_frames=end_frame)
    # for (i, frame) in enumerate(vidreader):  # vreader is faster than vread, but frame by frame
    #     if i == 0:
    #         imagesize = (int(frame.shape[0] * resize), int(frame.shape[1] * resize))
    #         vid = np.zeros((end_frame, imagesize[0], imagesize[1]))
    #
    #     vid[i, :, :] = np.squeeze(transform.resize(frame, imagesize))  # Resize in place for performance

    for (i, frame) in enumerate(vidreader):
        frame = np.squeeze(frame)

        if i == 0:
            if resize is None or resize == 1:
                vid = np.zeros((end_frame, frame.shape[0], frame.shape[1]), dtype=frame.dtype)
            else:
                imagesize = (int(frame.shape[0] * resize), int(frame.shape[1] * resize))
                vid = np.zeros((end_frame, imagesize[0], imagesize[1]))

        if resize is None or resize == 1:
            vid[i, :, :] = frame
        else:
            vid[i, :, :] = transform.resize(frame, imagesize, preserve_range=True)

    return vid


def load_mj2_as_arr(path_to_folder, start_frame=0, n_frames=0):

    file_name = path_to_folder.split('\\')[-1]

    # Open the MJ2 video file
    cap = cv2.VideoCapture(path_to_folder)

    # Get the width and height of frames
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            if i == start_frame:
                vid = np.zeros((n_frames, height, width))
                vid_frame = 0
            if i >= start_frame and i < ((n_frames + start_frame)):
                vid[vid_frame,:,:] = np.nanmean(frame, axis = 2)
                vid_frame += 1
        else:
            break

        i += 1

    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()

    return vid


def convert_mj2_to_avi(path_to_folder, output_format='avi'):

    file_name = path_to_folder.split('\\')[-1]
    input_file = path_to_folder
    output_file = os.path.join(os.path.dirname(path), f'{file_name[:-4]}.{output_format}')

    if os.path.exists(output_file):
        print(f'{output_file} already exists')

    elif not os.path.exists(input_file):
        print(f'Input path {input_file} does not exist')

    else:
        print(f'Creating {output_file} video')

        # Open the MJ2 video file
        cap = cv2.VideoCapture(input_file)

        # Get the width and height of frames
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Define the codec and create VideoWriter object
        if output_format == 'avi':
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        elif output_format == 'mp4':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        else:
            raise ValueError("Output format must be 'avi' or 'mp4'")

        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                out.write(frame)
            else:
                break

        # Release everything if job is finished
        cap.release()
        out.release()

    cv2.destroyAllWindows()


def measure_visual_areas(sign_thresh, amplitude, mask, pixel_size_mm=1.0, min_area=50, k_amp_threshold=1.0):
    """
    Measures total visual area and the largest negative patch area from a thresholded sign map,
    but only counts pixels with amplitude above a specified threshold.

    Parameters
    ----------
    sign_thresh : 2D np.array
        Thresholded sign map. Typically:
           +1 for positive patch,
           -1 for negative patch,
            0 for background.
    amplitude : 2D np.array
        Amplitude map corresponding to the sign map.
    pixel_size_mm : float
        Size of each pixel in mm (if you want mm^2 output). Use 1.0 for pixel counts.
    min_area : int
        Minimum area (in pixels) for a connected region to be considered (to filter noise).
    amp_threshold_ratio : float
        Ratio of the maximum amplitude to use as a threshold. Only pixels with amplitude
        >= (amp_threshold_ratio * max(amplitude)) are retained.

    Returns
    -------
    total_area_mm2 : float
        Total area (in mm^2 or pixels^2) of all patches (both positive and negative) that meet the criteria.
    largest_neg_area_mm2 : float
        Area of the single largest negative patch meeting the criteria.
    """
    # Compute the amplitude threshold.
    #amp_threshold = amp_threshold_ratio * np.max(amplitude)
    amp_threshold = np.mean(amplitude) + k_amp_threshold * np.std(amplitude)
    # Create a mask for pixels with sufficiently high amplitude.
    amp_mask = amplitude >= amp_threshold
    # plt.figure()
    # plt.imshow(np.where(final_mask, amplitude, 0))
    # plt.show()

    # mean amplitude of masked region (according to amp mask)
    masked_amp_mean = np.where(amp_mask, amplitude, 0).mean()

    # Combine the amplitude mask with the sign mask.
    final_mask = (sign_thresh != 0) & amp_mask

    # Label connected components in the final mask.
    labeled_visual = sk_label(final_mask)
    visual_props = regionprops(labeled_visual)

    # Sum the area of connected regions that exceed the minimum area.
    total_area_pixels = sum(rp.area for rp in visual_props if rp.area >= min_area)
    total_area_mm2 = total_area_pixels * (pixel_size_mm ** 2)

    # size of largest negative patch (V1)
    neg_mask = (sign_thresh == -1) & amp_mask # where is sign mask -1 and amp mask above threshold?
    labeled_neg = sk_label(neg_mask) # labels connected components in the binary pos_mask mask
    neg_props = regionprops(labeled_neg)    #extracts properties (e.g., area, centroid, bounding box) of each labeled patch
    neg_areas = [rp.area for rp in neg_props if rp.area >= min_area] # filter out small patches
    if neg_areas:
        largest_neg_area_pixels = max(neg_areas)
        largest_neg_area_mm2 = largest_neg_area_pixels * (pixel_size_mm ** 2)
    else:
        largest_neg_area_mm2 = 0

    # size of largest positive patch (LM)
    pos_mask = (sign_thresh == 1) & amp_mask # where is sign mask +1 and amp mask above threshold?
    labeled_pos = sk_label(pos_mask)   # labels connected components in the binary pos_mask mask

    # plt.figure()
    # plt.imshow(sk_label(pos_mask))
    # plt.show()
    # plt.figure()
    # plt.imshow(labeled_pos)
    # plt.show()
    # plt.figure()
    # plt.imshow(sk_label(neg_mask))
    # plt.show()
    # plt.figure()
    # plt.imshow(neg_mask)
    # plt.show()


    # shape (192, 256) > (Y, X)
    combined = np.zeros_like(pos_mask, dtype=np.int8)
    combined[pos_mask > 0 & mask] = 1
    combined[neg_mask > 0 & mask] = -1

    # nan out the parts of the combined mask that dont fall within the mask we drew
    combined = combined.astype(float)
    masked = combined.copy()
    masked[~mask] = np.nan

    # plt.figure()
    # plt.imshow(masked)
    # plt.colorbar()
    # plt.title('combined mask')
    # plt.show()

    rows, cols = np.where(mask)
    min_y, max_y = rows.min(), rows.max()
    min_x, max_x = cols.min(), cols.max()
    cropped = masked[min_y:max_y + 1, min_x:max_x + 1]

    pos_props = regionprops(labeled_pos) #extracts properties (e.g., area, centroid, bounding box) of each labeled patch
    pos_areas = [rp.area for rp in pos_props if rp.area >= min_area] # filter out small patches
    if pos_areas:
        largest_pos_area_pixels = max(pos_areas)
        largest_pos_area_mm2 = largest_pos_area_pixels * (pixel_size_mm ** 2)
    else:
        largest_pos_area_mm2 = 0

    return total_area_mm2, largest_neg_area_mm2, masked_amp_mean, largest_pos_area_mm2, masked,amp_threshold

def load_maps(fn):
    '''Load analysis maps to calculate sign map'''
    data = np.load(fn)
    phase_a, amp_a = data[0, :, :], data[1, :, :]
    phase_e, amp_e = data[2, :, :], data[3, :, :]
    reference = data[4, :, :]

    return phase_a, amp_a, phase_e, amp_e, reference


def load_parameters(fn):
    '''Load analysis parameters given in filepath fn'''
    config = ConfigParser()
    f = open(fn, 'r')
    config.read_file(f)
    params = {'inputfolder': config.get('config', 'inputfolder'),
              'savefolder': config.get('config', 'savefolder'),
              'savename': config.get('config', 'savename'),
              'template_a': config.get('config', 'template_a'),
              'template_e': config.get('config', 'template_e'),
              'evt_suffix': config.get('config', 'evt_suffix'),
              'framerate': config.get('config', 'framerate'),
              'nrep': config.get('config', 'nrep'),
              'timetot_a': config.get('config', 'timetot_a'),
              'timetot_e': config.get('config', 'timetot_e'),
              'sigma_p': config.get('config', 'sigma_p'),
              'sigma_s': config.get('config', 'sigma_s'),
              'shiftPhase': config.get('config', 'shiftPhase'),
              'sigma_t': config.get('config', 'sigma_t'),
              'sigma_c': config.get('config', 'sigma_c'),
              'openIter': config.get('config', 'openIter'),
              'closeIter': config.get('config', 'closeIter'),
              'dilateIter': config.get('config', 'dilateIter'),
              'borderWidth': config.get('config', 'borderWidth'),
              'epsilon': config.get('config', 'epsilon')}

    # Updates
    try:
        params['s_method'] = config.get('config', 's_method')
    except:
        params['s_method'] = 'gaussian'
    try:
        params['rotateMap'] = config.get('config', 'rotateMap')
    except:
        params['rotateMap'] = '1'

    f.close()
    return params


def sbx_get_ttlevents(fn):
    '''Load TTL events from scanbox events file.
       Based on sbx_get_ttlevents.m script.

       INPUT
       fn   : Filepath to events file

       OUTPUT
       evt  : List of TTL trigger time in units of frames (numpy.array)
    '''
    data = loadmat(fn)['ttl_events']
    if not (data.size == 0):
        # evt = data[:, 2] * 256 + data[:, 1]  # Not sure what this does
        evt = data[:, 2].astype(np.uint16) * 256 + data[:, 1].astype(np.uint16)
    else:
        evt = np.array([])

    return evt


def circular_smoothing(data, sigma):
    '''Apply Gaussian filter to circular data.

       INPUT
       data  : Input data (numpy.array)
       sigma : SD of Gaussian

       OUTPUT
       data_smoothed : Smoothed data (numpy.array)
    '''

    data_sin = np.sin(data)
    data_cos = np.cos(data)

    data_sinf = gaussian_filter(data_sin, sigma=sigma)
    data_cosf = gaussian_filter(data_cos, sigma=sigma)

    data_smoothed = np.arctan2(data_sinf, data_cosf)
    return data_smoothed


def rotate_image(im, rotate=1):
    '''Rotate image to reference orientation'''
    # Images will be plotted using origin='lower'
    if rotate:
        return np.fliplr(np.rot90(np.rot90(np.rot90(im))))
    else:
        return im


def circle_to_real(theta):
    '''Maps the angle theta to the real line'''
    x, y = np.cos(theta), np.sin(theta)
    if x == -1:
        t = np.inf
    else:
        t = y / (x + 1)

    return t


def real_to_circle(t):
    '''Maps real number to angle'''
    if t == np.inf:
        theta = np.arcsin(0)
    else:
        theta = np.arcsin(2 * t / (1 + t ** 2))

    return theta

class FOVDrawer:
    def __init__(self, amp_map):
        self.amp_map = amp_map
        self.mask = None
        self.ellipse_params = None  # Will hold (center_x, center_y, width, height)

        # Create the figure and axis
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(amp_map, cmap='gray')
        self.ax.set_title("Draw an ellipse (drag mouse) and press 'enter' to confirm")

        # Create the EllipseSelector widget
        self.selector = EllipseSelector(self.ax, self.onselect,
                                        interactive=True,
                                        minspanx=5, minspany=5, spancoords='pixels',
                                        button=[1],  # left mouse button only
                                        useblit=True)
        self.cid = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def onselect(self, eclick, erelease):
        """
        Called whenever the user draws an ellipse.
        Records the current ellipse parameters.
        """
        # eclick and erelease are mouse events at start and end of the drag.
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        # Calculate center, width and height from the two corner points.
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        self.ellipse_params = (center_x, center_y, width, height)

    def on_key_press(self, event):
        """
        When the user presses 'enter', finalize the ellipse and create the mask.
        """
        if event.key == 'enter':
            if self.ellipse_params is None:
                print("No ellipse drawn!")
                return
            center_x, center_y, width, height = self.ellipse_params
            print(
                f"Ellipse parameters: center=({center_x:.2f}, {center_y:.2f}), width={width:.2f}, height={height:.2f}")
            self.create_mask(center_x, center_y, width, height)
            plt.draw()

    def create_mask(self, center_x, center_y, width, height):
        """
        Create a boolean elliptical mask for the amplitude map.
        Pixels inside the ellipse are True; outside are False.
        """
        rows, cols = self.amp_map.shape
        Y, X = np.ogrid[:rows, :cols]
        # Semi-axis lengths
        a = width / 2.0
        b = height / 2.0
        # Equation of ellipse: ((X - center_x)/a)^2 + ((Y - center_y)/b)^2 <= 1
        self.mask = (((X - center_x) ** 2) / (a ** 2) + ((Y - center_y) ** 2) / (b ** 2)) <= 1

        # Overlay the mask on the amplitude map for visualization
        self.ax.imshow(self.mask, cmap='jet', alpha=0.3)
        print("Mask created and applied.")



import numpy as np
import matplotlib.pyplot as plt

class LineDrawingGUI:
    def __init__(self, azimuth_map, elevation_map):
        self.azimuth_map = azimuth_map
        self.elevation_map = elevation_map

        self.az_points = []
        self.el_points = []
        self.az_line = None
        self.el_line = None
        self.done = False

        self.fig, self.ax = plt.subplots(1, 2, figsize=(12, 5))

        self.ax[0].contourf(self.azimuth_map, cmap=pl.cm.jet, levels=10, zorder=-1)
        # self.ax[0].imshow(self.azimuth_map, cmap='jet')
        self.ax[0].set_title('Azimuth')
        self.ax[0].set_xticks([])
        self.ax[0].set_yticks([])

        # self.ax[1].imshow(self.elevation_map, cmap='jet')
        self.ax[1].contourf(self.elevation_map, cmap=pl.cm.jet, levels=10, zorder=-1)
        self.ax[1].set_title('Elevation')
        self.ax[1].set_xticks([])
        self.ax[1].set_yticks([])

        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def onclick(self, event):
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return

        x, y = event.xdata, event.ydata

        if event.inaxes == self.ax[0]:
            if len(self.az_points) < 2:
                self.az_points.append((x, y))
                self.ax[0].plot(x, y, 'ro')

            if len(self.az_points) == 2 and self.az_line is None:
                p0, p1 = self.az_points
                self.az_line = np.array([p0, p1])
                self.ax[0].plot([p0[0], p1[0]], [p0[1], p1[1]], 'k-', linewidth=2)

        elif event.inaxes == self.ax[1]:
            if len(self.el_points) < 2:
                self.el_points.append((x, y))
                self.ax[1].plot(x, y, 'ro')

            if len(self.el_points) == 2 and self.el_line is None:
                p0, p1 = self.el_points
                self.el_line = np.array([p0, p1])
                self.ax[1].plot([p0[0], p1[0]], [p0[1], p1[1]], 'k-', linewidth=2)

        self.fig.canvas.draw_idle()

        if self.az_line is not None and self.el_line is not None:
            self.done = True

    def show(self):
        plt.show(block=False)