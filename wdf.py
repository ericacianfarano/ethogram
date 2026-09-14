'''
Retinotopic_mapping.py
Heavily inspired by Emily Mace's analysis script and the NeuroAnalysisTools library

Rudi Tong, Trenholm Lab, 2020

Version 1.1 (09.10.2020)

Additional dependencies:
    ffmpeg is required for skvideo backend
'''

import os, argparse
from datetime import datetime
import numpy as np
from timeit import default_timer as timer
import skvideo.io
import skimage.transform as transform
from skimage.morphology import skeletonize
from skimage import exposure
from scipy.io import loadmat
try:
    import scipy.fft as fft
except:
    import scipy.fftpack as fft
import scipy.signal as signal
import glob
import matplotlib.pyplot as pl
from scipy.ndimage import gaussian_filter, median_filter, binary_opening, binary_closing, binary_dilation, label, \
    laplace
from configparser import ConfigParser

version = 'v1.1'


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

    # Import video file as numpy.array
    vidreader = skvideo.io.vreader(fn, inputdict=inputparameters, outputdict=outputparameters, num_frames=end_frame)
    for (i, frame) in enumerate(vidreader):  # vreader is faster than vread, but frame by frame
        if i == 0:
            imagesize = (int(frame.shape[0] * resize), int(frame.shape[1] * resize))
            vid = np.zeros((end_frame, imagesize[0], imagesize[1]))

        vid[i, :, :] = np.squeeze(transform.resize(frame, imagesize))  # Resize in place for performance

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



#convert_mj2_to_avi(r'E:\retinotopic mapping\test\20231104\wdf\wdf_000_a11.mj2')


angular_dir = 'a'
#config_file = r'E:\retinotopic mapping\test\20231104\wdf\config.txt'
#config_file = r'E:\retinotopic mapping\EC_GNAT_02\20240807\wdf_big25\config.txt'
config_file = r'EI:\retmap\EC_GCaMP6s_09\20241009\big100\config.txt'
params = load_parameters(config_file)
framerate = int(params['framerate'])
fn = os.path.join(os.path.dirname(config_file), params[f'template_{angular_dir}'].replace('*', '11') + '_events')

evt = sbx_get_ttlevents(fn)
start_frame = evt[evt > 0][0] - 1  # -1 to start counting from 0

if angular_dir == 'a':
    timetot = int(params ['timetot_a'])
elif angular_dir == 'e':
    timetot = int(params ['timetot_e'])
nimagtot = int(np.round(timetot*framerate))

vid_path = os.path.join(os.path.dirname(config_file), params[f'template_{angular_dir}'].replace('*', '11'))
data_vid = load_mj2_as_arr(vid_path, start_frame=start_frame, n_frames=nimagtot)

baseline = np.nanmean(data_vid[0:10], axis=0)

# Remove DC component of image (i.e. average intensity value)
#data_ds_rel = data_ds - np.nanmean(data_ds, axis=0)

baseline = np.mean(data_vid, axis=0)[np.newaxis, :, :]  # mean across time (height x width)
std_dev = np.std(data_vid, axis=0)[np.newaxis, :, :]  # standard deviation across time (height x width)

#Z score normalization for each frame - standardizes the data
# z scores indicate how many standard deviations each pixels' value is away from the mean
deviation = (data_vid - baseline) / std_dev

fig, ax = plt.subplots(2, 3, figsize=(8,4))
ax = ax.ravel()

for i in range (6):
    # pixels above x standard deviations of the mean
    #threshold = 4
    threshold = i + 2
    threshold_mask = np.abs(deviation) > threshold

    # Identify pixels that are modulated at least once during the experiment
    threshold_pixels = np.any(threshold_mask, axis=0)

    cax = ax[i].imshow(threshold_pixels, cmap='plasma', interpolation='nearest')
    fig.colorbar(cax, ax=ax[i], shrink = 0.5)
    ax[i].set_title(f'threshold: {threshold} STDs')
plt.suptitle(config_file.split('\\')[-4])
plt.tight_layout()
plt.show()

# divide the video into 10 equal parts
n_repeats = 10
repeats = np.mean(data_vid.reshape(n_repeats, -1, data_vid.shape[-2], data_vid.shape[-1]), axis = 0)

tifffile.imwrite(os.path.join(os.path.dirname(config_file), 'a11_repeats.tiff'), repeats)


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
        evt = data[:, 2] * 256 + data[:, 1]  # Not sure what this does
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


class RetinotopicMap():
    '''Identify retinotopic regions based on phase maps'''

    def __init__(self, savefolder, savename, sigma_p, sigma_s, s_method, sigma_t, sigma_c, openIter, closeIter,
                 dilateIter, shiftPhase, borderWidth, epsilon, rotateMap):
        self.savefolder = savefolder  # Path to folder to save output/read input
        self.savename = savename  # Name for phase map file
        self.sigma_p = sigma_p  # SD of gaussian to smooth phase map
        self.sigma_s = sigma_s  # SD of gaussian to smooth sign map
        self.s_method = s_method  # Method to smooth sign map (uses sigma_s)
        self.sigma_t = sigma_t  # SD of cutoff for thresholding
        self.sigma_c = sigma_c  # SD of cutoff for contour plot
        self.openIter = openIter  # Number of iterations for morph. opening
        self.closeIter = closeIter  # Number of iterations for morph. closure
        self.dilateIter = dilateIter  # Number of iterations for morph. dilation
        self.shiftPhase = shiftPhase  # If 1, shift phase angles to [0, 2pi]
        self.borderWidth = borderWidth  # Witdh of border between patches
        self.epsilon = epsilon  # Sensitivity for point search
        self.rotateMap = rotateMap  # Rotate sign map

        # Internal variables
        self.sign = None
        self.patch = None
        self.sign_thresh = None
        self.vis_border = None
        self.raw_patches = None
        self.patch_i = None
        self.phase1, self.phase2 = None, None
        self.ref = None
        self.mask = None

    def _load_data(self):
        '''Load in phase and amplitude maps'''
        phase_a, amp_a, phase_e, amp_e, ref = load_maps(os.path.join(self.savefolder, self.savename + '_fftmaps.npy'))
        self.amp1 = amp_a
        self.amp2 = amp_e
        self.phase1 = circular_smoothing(phase_a, sigma=0)
        self.phase2 = circular_smoothing(phase_e, sigma=0)
        self.ref = exposure.equalize_adapthist(ref, clip_limit=0.03,
                                               kernel_size=(int(ref.shape[0] / 20), int(ref.shape[1] / 20)))
        edge = laplace(self.ref)
        self.ref -= edge
        self.ref[self.ref < 0] = 0
        self.ref = exposure.adjust_log(self.ref)
        return True

    def _get_sign_map(self):
        '''Calculate the sign map from two orthogonal phase maps.'''

        # Check if phase maps have same dimensions
        if self.phase1.shape != self.phase2.shape:
            raise LookupError("Phase maps should be the same size!")

        # Apply circular gaussian smoothing (seems to also get rid of phase artefacts..)
        phase1 = circular_smoothing(self.phase1, sigma=self.sigma_p)
        phase2 = circular_smoothing(self.phase2, sigma=self.sigma_p)

        # Calculate the gradient of each phase - output is list of gradx and grady
        grad1 = np.gradient(phase1)
        grad2 = np.gradient(phase2)

        # Calculate gradient direction
        graddir1 = np.arctan2(grad1[1], grad1[0])
        graddir2 = np.arctan2(grad2[1], grad2[0])

        # Calculate phase difference
        vdiff = np.multiply(np.exp(1j * graddir1), np.exp(-1j * graddir2))
        sign_map = np.sin(np.angle(vdiff))

        self.sign = sign_map

        return True

    def _get_patch_map(self):
        '''Given a sign map, threshold the map relative to its standard deviation. Then create binary patches.'''

        # Smooth sign map
        if self.s_method == 'gaussian':
            sign = gaussian_filter(self.sign, sigma=self.sigma_s)
        elif self.s_method == 'median':
            sign = median_filter(self.sign, int(self.sigma_s))

        # Calculate cutoff
        cutoff = self.sigma_t * np.nanstd(sign)

        # Treshold
        sign_thresh = np.zeros_like(sign)
        sign_thresh[sign > cutoff] = 1
        sign_thresh[sign < -cutoff] = -1
        self.sign_thresh = sign_thresh

        # Remove noise
        sign_thresh = binary_opening(np.abs(sign_thresh), iterations=self.openIter).astype(np.int)

        # Identify patches
        patches, patch_i = label(sign_thresh)

        # Close each region
        patch_map = np.zeros_like(patches)
        for i in range(patch_i):
            curr_patch = np.zeros_like(patches)
            curr_patch[patches == i + 1] = 1
            patch_map += binary_closing(curr_patch, iterations=self.closeIter).astype(np.int)

        self.raw_patches = patches
        self.patch_i = patch_i
        print('Identified %s visual patches.' % (self.patch_i))

        # Expand patches - directly adapted from NeuroAnalysisTools
        total_area = binary_dilation(patch_map, iterations=self.dilateIter).astype(np.int)
        patch_border = total_area - patch_map

        patch_border = skeletonize(patch_border)

        if self.borderWidth >= 1:
            patch_border = binary_dilation(patch_border, iterations=self.borderWidth - 1).astype(np.float)

        patch_border[patch_border == 0] = np.nan
        self.patch = patch_border

        return True

    def _get_visual_border(self):
        '''Given a patch map, find the global borders of visual cortex'''
        self.vis_borders = binary_dilation(
            binary_opening(binary_closing(np.abs(self.sign_thresh), iterations=self.closeIter),
                           iterations=self.openIter), iterations=self.dilateIter).astype(np.int)
        return True

    def _get_contours(self):
        '''Create contour maps for azimuth and elevation phases'''
        azimuth_map = np.copy(self.phase1)
        azimuth_map = circular_smoothing(azimuth_map, sigma=self.sigma_c)
        azimuth_map = np.ma.masked_array(azimuth_map, mask=self.vis_borders == 0)

        elevation_map = np.copy(self.phase2)
        elevation_map = circular_smoothing(elevation_map, sigma=self.sigma_c)
        elevation_map = np.ma.masked_array(elevation_map, mask=self.vis_borders == 0)

        return azimuth_map, elevation_map

    def _find_points(self, x, y):
        '''Find points on map tht correspond to the same retinotopic location'''
        # Calculate distance of each pixel to position given in x, y
        dist = np.sqrt((self.phase1 - x) ** 2 + (self.phase2 - y) ** 2)
        # Apply absolute threshold given in self.epsilon
        dist[dist > self.epsilon] = np.nan
        # Iterate over each patch and find the minimum point
        points = [(np.nan, np.nan)]
        for i in range(self.patch_i):
            dist_masked = np.copy(dist)
            dist_masked[self.raw_patches != i + 1] = np.nan
            try:
                point = np.nanargmin(dist_masked)
                points.append(np.unravel_index(point, dist_masked.shape))
            except:
                pass

        return np.array(points)

    def _onclick(self, event):
        '''Update overlay plot when clicked'''
        xind, yind = int(np.round(event.xdata)), int(np.round(event.ydata))
        x, y = self.phase1[xind, yind], self.phase2[xind, yind]
        print('Azimuth angle =', x, '\tElevation angle =', y)
        self.mask = self._find_points(x, y)
        self.ax[1, 2].clear()
        self.ax[1, 2].imshow(rotate_image(self.ref), cmap=pl.cm.gray, origin='lower', vmin=0)
        self.ax[1, 2].imshow(rotate_image(self.patch), cmap=pl.cm.hsv, origin='lower')
        self.ax[1, 2].plot(self.mask[:, 0], self.mask[:, 1], 'kx', mew=2)
        self.ax[1, 2].set_title('Overlay')
        fig.canvas.draw()

    def run(self):
        # Load in data
        self._load_data()
        # Create maps
        self._get_sign_map()
        self._get_patch_map()
        self._get_visual_border()

        # Save
        np.save(os.path.join(self.savefolder, self.savename + '_signmap.npy'), self.sign)

    def plot(self):
        # Plot sign map - flip by 90 degrees ccw
        self.fig, self.ax = pl.subplots(nrows=2, ncols=3, figsize=(12, 8))
        self.ax[0, 0].imshow(rotate_image(self.sign, self.rotateMap), cmap=pl.cm.jet)
        self.ax[0, 0].set_title('Raw sign map')

        # Plot patch map - flip by 90 degrees ccw
        if self.s_method == 'gaussian':
            self.ax[0, 1].imshow(rotate_image(gaussian_filter(self.sign, sigma=self.sigma_s), self.rotateMap),
                                 cmap=pl.cm.jet)
        elif self.s_method == 'median':
            self.ax[0, 1].imshow(rotate_image(median_filter(self.sign, int(self.sigma_s)), self.rotateMap),
                                 cmap=pl.cm.jet)
        self.ax[0, 1].set_title('Sign map')

        # Plot reference map (Zhuang et al. 2017, Figure 3C)
        try:
            reference = pl.imread(
                r'C:\Users\TrenholmLab\TrenholmLab_Analysis_Scripts\Retinotopic_Mapping\Reference_Zhuang_2017_Fig3.png')
        except:
            reference = np.zeros_like(self.sign)
        self.ax[0, 2].imshow(reference)
        self.ax[0, 2].get_xaxis().set_visible(False)
        self.ax[0, 2].get_yaxis().set_visible(False)
        self.ax[0, 2].set_title('Reference')

        # Plot contour plots
        azimuth_map, elevation_map = self._get_contours()
        # Plot azimuth contour
        azi_contour = self.ax[1, 0].contourf(rotate_image(azimuth_map, self.rotateMap), cmap=pl.cm.jet, levels=10,
                                             zorder=-1)
        self.ax[1, 0].imshow(rotate_image(self.patch, self.rotateMap), cmap=pl.cm.gray)
        pl.colorbar(azi_contour, ax=self.ax[1, 0])
        self.ax[1, 0].set_title('Azimuth Contours')

        # Plot elevation contour
        self.ax[1, 1].contourf(rotate_image(elevation_map, self.rotateMap), cmap=pl.cm.jet, levels=10, zorder=-1)
        self.ax[1, 1].imshow(rotate_image(self.patch, self.rotateMap), cmap=pl.cm.gray)
        self.ax[1, 1].set_title('Elevation Contours')

        # Interactive plot
        self.ax[1, 2].imshow(rotate_image(self.ref, self.rotateMap), cmap=pl.cm.gray, vmin=0)
        self.ax[1, 2].imshow(rotate_image(self.patch, self.rotateMap), cmap=pl.cm.hsv)
        self.ax[1, 2].set_title('Overlay')

        cid = self.fig.canvas.mpl_connect('button_press_event', self._onclick)

        return self.fig, self.ax

    def plot_raw(self):
        '''Plot raw phase and amplitude maps'''
        self.figraw, self.axraw = pl.subplots(nrows=2, ncols=2, figsize=(9, 8))
        # Plot amplitude map
        self.axraw[0, 0].imshow(rotate_image(self.amp1, self.rotateMap), cmap=pl.cm.jet)
        self.axraw[0, 0].set_title('Raw amplitude azimuth')

        self.axraw[0, 1].imshow(rotate_image(self.amp2, self.rotateMap), cmap=pl.cm.jet)
        self.axraw[0, 1].set_title('Raw amplitude elevation')

        # Plot phase map
        self.axraw[1, 0].imshow(rotate_image(self.phase1, self.rotateMap), cmap=pl.cm.jet)
        self.axraw[1, 0].set_title('Raw phase azimuth')

        self.axraw[1, 1].imshow(rotate_image(self.phase2, self.rotateMap), cmap=pl.cm.jet)
        self.axraw[1, 1].set_title('Raw phase elevation')

        return self.figraw, self.axraw


class PhaseMap():
    '''Generate phase map from widefield calcium imaging data'''

    def __init__(self, inputfolder, savefolder, savename, templates, evt_suffix, framerate, nrep, timetot_a, timetot_e):
        self.inputfolder = inputfolder  # Path to folder with data files
        self.savefolder = savefolder  # Path to folder to save output
        self.savename = savename  # Name for output analysis file
        self.templates = templates  # Template for naming data files
        self.evt_suffix = evt_suffix  # File suffix for events file
        self.framerate = framerate  # in Hz
        self.nrep = nrep  # Number of trial repetitions
        self.timetot_a = timetot_a  # Total time for azimuth scan (sec)
        self.timetot_e = timetot_e  # Total time for elevation scan (sec)

    def _get_filenames(self):
        '''Extract filenames for azimuth and elevation data (2 each) + events file
           self.filetemplate specifies the template used for naming data files.'''

        self.filepath_a = sorted(glob.glob(os.path.join(self.inputfolder, self.templates[0])))
        self.filepath_e = sorted(glob.glob(os.path.join(self.inputfolder, self.templates[1])))

        return True

    def _get_phase_map(self, fn, angular_dir):
        '''Get phase map for one angular direction (azimuth or elevation)

           INPUT
           fn           : Path to experiment file
           angulat_dir  : 'a' for azimuth; 'e' for elevation

           OUTPUT
           data_fft_field    : Complex field output of fft at first harmonic
           '''

        # Calculate stimulus-dependent parameters
        if angular_dir == 'a':
            timetot = self.timetot_a
        elif angular_dir == 'e':
            timetot = self.timetot_e
        else:
            raise (ValueError("angular_dir has to be either 'a' or 'e' for azimuth and elevation, respectively"))

        T = timetot / self.nrep
        nimagtot = int(np.round(timetot * self.framerate))
        fstim = 1 / T

        # Load data - only load image frames during stimulus presentation. Frame number extracted from events file
        # Also downsample data for performance
        print('Loading file %s' % (fn))
        start = timer()
        evt = sbx_get_ttlevents(fn + self.evt_suffix)
        start_frame = evt[evt > 0][0] - 1  # -1 to start counting from 0
        data_ds = load_data(fn, start_frame=start_frame, end_frame=nimagtot, resize=0.5)
        # Save a copy of the background image
        self._reference = np.nanmean(data_ds[0:10], axis=0)
        end = timer()
        print('File loaded - %.2f s' % (end - start))

        # Remove DC component of image (i.e. average intensity value)
        start = timer()
        data_ds_rel = data_ds - np.nanmean(data_ds, axis=0)

        # Apply high pass filter (subtract out frequencies below 2 cycles of the stimulus)
        cutfreq = 1 / (2 * T)
        b, a = signal.butter(1, cutfreq / (0.5 * self.framerate), 'low')
        data_lowfreq = signal.filtfilt(b, a, data_ds_rel, axis=0)
        data_filt = data_ds_rel - data_lowfreq
        end = timer()
        print('Preprocessing completed - %.2f s' % (end - start))

        # Perform Fourier transform
        start = timer()
        data_fft = fft.fft(data_filt, axis=0)
        freq = fft.fftfreq(data_filt.shape[0], d=1 / self.framerate)
        end = timer()
        print('Fourier completed - %.2f s' % (end - start))

        # Extract complex field at first harmonic frequency of the stimulus
        f_ind = np.where(freq >= fstim)[0][0]  # Index of the stimulus frequency
        data_fft_field = data_fft[f_ind, :, :]

        return data_fft_field

    def _combine_maps(self, fft_dir1, fft_dir2):
        '''Given two complex fields, combine them to correct for response delay

           INPUT
           fft_dir1 : Complex field of direction 1 (numpy.array)
           fft_dir2 : Complex field of direction 2 (numpy.array)

           OUTPUT
           phase_combined       : Phase map (numpy.array)
           amplitude_combined   : Amplitude map (numpy.array)
        '''

        # Calculate difference in phase in complex field
        phase_combined = (np.angle(fft_dir1 / fft_dir2) / 2)

        # Calculate amplitude average
        amplitude_combined = (np.abs(fft_dir1) + np.abs(fft_dir2)) / 2

        return phase_combined, amplitude_combined

    def run(self):
        '''Run full analysis'''
        # Get filenames
        self._get_filenames()

        # Perform Fourier analysis
        data_a = [self._get_phase_map(fn, 'a') for fn in self.filepath_a]
        data_e = [self._get_phase_map(fn, 'e') for fn in self.filepath_e]

        # Combine phase maps
        if len(data_a) == 2:
            phase_map_a, amp_map_a = self._combine_maps(data_a[0], data_a[1])
        else:
            print('Only one azimuth file; calculate phase map without correction.')
            phase_map_a, amp_map_a = np.angle(data_a[0]), np.abs(data_a[0])

        if len(data_e) == 2:
            phase_map_e, amp_map_e = self._combine_maps(data_e[0], data_e[1])
        else:
            print('Only one elevation file; calculate phase map without correction.')
            phase_map_e, amp_map_e = np.angle(data_e[0]), np.abs(data_e[0])

        # Save phase and amplitude maps as 3D matrix in order phase_a, amp_a, phase_b, amp_b
        save_matrix = np.array([phase_map_a, amp_map_a, phase_map_e, amp_map_e, self._reference])
        np.save(os.path.join(self.savefolder, self.savename + '_fftmaps.npy'), save_matrix)

        return True


if __name__ == '__main__':
    # Setup parser
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="File path to config file")
    parser.add_argument("-m", "--mode",
                        help="Choose analysis mode: 1 - Create complex fields\t 2 - Create and plot sign map")
    args = parser.parse_args()

    # Open config file
    try:
        params = load_parameters(args.config)
    except:
        raise ValueError("Cannot open config file or config file does not exist!")

    # Log info
    print('Running Retinotopic_mapping.py %s\t' % (version), datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    print('Config file: ', args.config)
    print('Running mode %s' % (args.mode))
    print('Analysis folder: ', params['savefolder'], '\n')

    # Determine mode
    if args.mode == '1':
        phasemap = PhaseMap(inputfolder=params['inputfolder'],
                            savefolder=params['savefolder'],
                            savename=params['savename'],
                            templates=(params['template_a'], params['template_e']),
                            evt_suffix=params['evt_suffix'],
                            framerate=float(params['framerate']),
                            nrep=int(params['nrep']),
                            timetot_a=float(params['timetot_a']),
                            timetot_e=float(params['timetot_e']))

        phasemap.run()

    elif args.mode == '2':
        # Load relevant params
        patchmap = RetinotopicMap(savefolder=params['savefolder'],
                                  savename=params['savename'],
                                  sigma_p=float(params['sigma_p']),
                                  sigma_s=float(params['sigma_s']),
                                  s_method=params['s_method'],
                                  shiftPhase=int(params['shiftPhase']),
                                  sigma_t=float(params['sigma_t']),
                                  sigma_c=float(params['sigma_c']),
                                  openIter=int(params['openIter']),
                                  closeIter=int(params['closeIter']),
                                  dilateIter=int(params['dilateIter']),
                                  borderWidth=int(params['borderWidth']),
                                  epsilon=float(params['epsilon']),
                                  rotateMap=int(params['rotateMap']))

        patchmap.run()
        figraw, axraw = patchmap.plot_raw()
        fig, ax = patchmap.plot()

        pl.show()

    else:
        raise ValueError("Mode has to be either 1 (Create complex fields) or 2 (Create and plot sign map)!")

