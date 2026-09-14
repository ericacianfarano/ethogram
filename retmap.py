'''
Retinotopic_mapping.py
Heavily inspired by Emily Mace's analysis script and the NeuroAnalysisTools library

Additional dependencies:
    ffmpeg is required for skvideo backend
'''
import matplotlib.pyplot as plt

from retmap_helpers import *

version = 'v1.1'
monitor_width_cm, monitor_height_cm = 53.34, 30
half_monitor_width_cm, half_monitor_height_cm = monitor_width_cm/2, monitor_height_cm/2
monitor_distance_eye_cm = 13
monitor_visual_degrees_width = np.rad2deg(2 * math.atan(half_monitor_width_cm / monitor_distance_eye_cm))
monitor_visual_degrees_height = np.rad2deg(2 * math.atan(half_monitor_height_cm / monitor_distance_eye_cm))

total_visual_area = {}
largest_neg_patch_area = {}
largest_pos_patch_area = {}
total_visual_area_days = {}
largest_neg_patch_area_days = {}
largest_pos_patch_area_days = {}
mean_amplitude = {}
amplitude_threshold = {}

class RetinotopicMap():
    '''Identify retinotopic regions based on phase maps'''

    def __init__(self, savefolder, savename, sigma_p, sigma_s, s_method, sigma_t, sigma_c, openIter, closeIter,
                 dilateIter, shiftPhase, borderWidth, epsilon, rotateMap, min_area, animal_dob = None, animal=None, day=None, draw_mask = True, draw_lines = True):
        savefolder = savefolder.replace('I:', 'E:')
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
        self.animal = animal
        self.animal_dob = animal_dob
        self.day = day
        self.min_area = min_area
        self.draw_mask = draw_mask
        self.draw_lines = draw_lines

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

        # plot_autocorrelation(compute_spatial_autocorrelation(self.sign), title='Spatial Autocorrelation')

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
        sign_thresh = binary_opening(np.abs(sign_thresh), iterations=self.openIter).astype(int)

        # Identify patches
        patches, patch_i = label(sign_thresh)

        # Close each region
        patch_map = np.zeros_like(patches)
        for i in range(patch_i):
            curr_patch = np.zeros_like(patches)
            curr_patch[patches == i + 1] = 1
            patch_map += binary_closing(curr_patch, iterations=self.closeIter).astype(int)

        self.raw_patches = patches
        self.patch_i = patch_i
        print('Identified %s visual patches.' % (self.patch_i))

        # Expand patches - directly adapted from NeuroAnalysisTools
        total_area = binary_dilation(patch_map, iterations=self.dilateIter).astype(int)
        patch_border = total_area - patch_map

        patch_border = skeletonize(patch_border)

        if self.borderWidth >= 1:
            patch_border = binary_dilation(patch_border, iterations=self.borderWidth - 1).astype(float)

        patch_border[patch_border == 0] = np.nan
        self.patch = patch_border

        avg_amp = (self.amp1 + self.amp2) / 2.0

        mask_path = os.path.join(self.savefolder, self.savename + '_mask.npy')
        if self.draw_mask: # if true
            drawer = FOVDrawer(avg_amp)
            plt.show(block = False)

            while drawer.mask is None:
                plt.pause(0.1)
                time.sleep(0.1)

            mask = drawer.mask
            np.save(mask_path, mask)

        else:
            mask = np.load(mask_path)

        self.mask = mask
        self.masked_amp_map = np.where(self.mask, avg_amp, 0)
        self.amp_threshold = np.mean(self.masked_amp_map) + 3 * np.std(self.masked_amp_map)
        self.amp_mask = avg_amp > self.amp_threshold

        pixel_size_mm = 1.0  # <--- adjust if you know your mm/pixel calibration
        # if pixel_size_mm = 1.0, the output areas will be in pixel counts
        # if we know the FOV in mm: pixel_size_mm = FOV (mm) / n_pixels_across FOV
        total_area, largest_neg_area, masked_amp_mean, largest_pos_area, self.masked_patch_map, amp_threshold = measure_visual_areas(self.sign_thresh, self.masked_amp_map, self.mask, pixel_size_mm=1/68, min_area = self.min_area, k_amp_threshold = 1)

        total_visual_area[self.animal] = total_area
        mean_amplitude[self.animal] = masked_amp_mean
        largest_neg_patch_area[self.animal] = largest_neg_area
        largest_pos_patch_area[self.animal] = largest_pos_area
        amplitude_threshold[self.animal] = amp_threshold

        print(f"Total visual area = {total_area:.2f} pixels")
        print(f"Largest negative patch area = {largest_neg_area:.2f} pixels")

        return True

    def _get_visual_border(self):
        '''Given a patch map, find the global borders of visual cortex'''
        self.vis_borders = binary_dilation(
            binary_opening(binary_closing(np.abs(self.sign_thresh), iterations=self.closeIter),
                           iterations=self.openIter), iterations=self.dilateIter).astype(int)
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

        self.fig, self.ax = pl.subplots(nrows=1, ncols=4, figsize=(18, 5))

        # Plot patch map - flip by 90 degrees ccw
        #np.where(self.mask, avg_amp, 0)
        from matplotlib.colors import TwoSlopeNorm
        norm = TwoSlopeNorm(vcenter=0) #TwoSlopeNorm(vmin = -0.5, vmax = 0.5, vcenter=0)
        if self.s_method == 'gaussian':
            sign_map = np.where(self.mask, rotate_image(gaussian_filter(self.sign, sigma=self.sigma_s), self.rotateMap),np.nan)
            im = self.ax[0].imshow(sign_map, cmap=pl.cm.jet, vmin = -1, vmax = 1, norm = norm)
        elif self.s_method == 'median':
            sign_map = np.where(self.mask, rotate_image(median_filter(self.sign, int(self.sigma_s)), self.rotateMap),np.nan)
            # sign_map = np.where(self.mask, median_filter(self.sign, int(self.sigma_s)),np.nan)
            # sign_map = np.where(self.mask, self.sign, np.nan)
            im = self.ax[0].imshow(sign_map,cmap=pl.cm.jet, norm = norm)

        cbar = pl.colorbar(im, ax=self.ax[0], fraction=0.04)
        self.ax[0].set_title('Sign map')
        self.ax[0].set_xticks([])
        self.ax[0].set_yticks([])

        # mask_plot = rotate_image(self.mask.astype(float), self.rotateMap) > 0.5
        # mask_plot = self.mask.astype(float) > 0.5

        # Plot contour plots
        azimuth_map, elevation_map = self._get_contours()
        #
        # az_plot = np.where(mask_plot, rotate_image(azimuth_map, self.rotateMap), np.nan)
        # el_plot = np.where(mask_plot, rotate_image(elevation_map, self.rotateMap), np.nan)
        # p0_az, p1_az, _ = dominant_gradient_line(np.nan_to_num(az_plot, nan=0.0), self.mask, smooth_sigma=1.5)
        # p0_el, p1_el, _ = dominant_gradient_line(np.nan_to_num(el_plot, nan=0.0), self.mask, smooth_sigma=1.5)

        # Plot azimuth contour
        phase_range = rotate_image(azimuth_map, self.rotateMap).max() - rotate_image(azimuth_map, self.rotateMap).min()
        deg_per_phase_width = monitor_visual_degrees_width / phase_range
        print(f'Each unit of phase corresponds to ~{np.round(deg_per_phase_width, 2)} deg of visual angle (azimuth)')
        azi_contour = self.ax[2].contourf(np.where(self.mask, rotate_image(azimuth_map, self.rotateMap),np.nan), cmap=pl.cm.jet, levels=10,zorder=-1)
        self.ax[2].imshow(rotate_image(self.patch, self.rotateMap), cmap=pl.cm.gray, norm = norm)
        # self.ax[2].imshow(rotate_image(self.patch, self.rotateMap), cmap=pl.cm.gray, vmin=-75, vmax=75)
        #pl.colorbar(azi_contour, ax=self.ax[2], fraction = 0.04)
        cbar = pl.colorbar(azi_contour, ax=self.ax[2], fraction=0.04)
        # Convert colorbar ticks to visual degrees
        tick_vals = cbar.get_ticks()
        cbar.set_ticks(tick_vals)
        cbar.set_ticklabels(np.round(tick_vals * deg_per_phase_width, 2))  # Convert to degrees
        self.ax[2].set_title('Azimuth Contours')
        self.ax[2].set_xticks([])
        self.ax[2].set_yticks([])


        # Plot elevation contour
        phase_range = rotate_image(elevation_map, self.rotateMap).max() -rotate_image(elevation_map, self.rotateMap).min()
        deg_per_phase_height = monitor_visual_degrees_height / phase_range
        print(f'Each unit of phase corresponds to ~{np.round(deg_per_phase_height, 2)} deg of visual angle (elevation)')
        elev_contour = self.ax[3].contourf(np.where(self.mask, rotate_image(elevation_map, self.rotateMap), np.nan), cmap=pl.cm.jet, levels=10, zorder=-1)
        self.ax[3].imshow(rotate_image(self.patch, self.rotateMap), cmap=pl.cm.gray, norm = norm)
        #pl.colorbar(azi_contour, ax=self.ax[3], fraction = 0.04)
        cbar = pl.colorbar(elev_contour, ax=self.ax[3], fraction=0.04)
        # Convert colorbar ticks to visual degrees
        tick_vals = cbar.get_ticks()
        cbar.set_ticks(tick_vals)
        cbar.set_ticklabels(np.round(tick_vals * deg_per_phase_height, 2))  # Convert to degrees
        self.ax[3].set_title('Elevation Contours')
        self.ax[3].set_xticks([])
        self.ax[3].set_yticks([])

        # Plot sign map - flip by 90 degrees ccw
        self.ax[1].imshow(np.where(self.mask, rotate_image(self.ref, self.rotateMap), np.nan), cmap=pl.cm.gray, vmin=0) # plot image
        self.ax[1].imshow(np.where(self.mask,rotate_image(self.patch, self.rotateMap), np.nan), cmap=pl.cm.hsv) # plot contours
        self.ax[1].set_title('Overlay')
        self.ax[1].set_xticks([])
        self.ax[1].set_yticks([])

        animal_name, day, subfile = self.savefolder.split('\\')[2],self.savefolder.split('\\')[3],self.savefolder.split('\\')[4]
        pl.suptitle(f'{animal_name}, {day}, {subfile}' )

        if not os.path.exists(os.path.join(fr'E:\retmap_figures', animal_name)):
            os.makedirs(os.path.join(fr'E:\retmap_figures', animal_name))
        plt.savefig(fr'E:\retmap_figures\{animal_name}\{animal_name}{day}{subfile}.png')
        plt.savefig(fr'E:\retmap_figures\{animal_name}\{animal_name}{day}{subfile}.svg')

        # plt.figure()
        # Create a mask for pixels with sufficiently high amplitude.
        # sign_map = np.where(self.mask, (gaussian_filter(self.sign, int(self.sigma_s), np.nan)))
        # sign_map = np.where(self.mask, self.sign, np.nan)
        sign_map = np.where(self.mask, median_filter(self.sign, int(self.sigma_s)), np.nan)
        # sign_map = np.where(self.mask, median_filter(self.sign, int(self.sigma_s)), np.nan)

        sign_thesholded_mask = sign_map >= self.amp_threshold
        # im = plt.imshow(sign_map, cmap=pl.cm.jet, norm=norm)
        # plot where the hand-drawn window mask + amp masks are defined
        plt.figure()
        plt.title(self.animal)
        plt.imshow(np.where(self.mask, sign_map, np.nan),cmap=pl.cm.jet, vmin = -1, vmax = 1)
        plt.show()

        fig, ax = pl.subplots(nrows=1, ncols=2, figsize=(15, 5))
        a = ax[0].imshow(np.where(self.mask, self.amp1, np.nan), cmap = pl.cm.jet) #np.where((self.amp_mask & self.mask)
        e = ax[1].imshow(np.where(self.mask, self.amp2, np.nan), cmap = pl.cm.jet)
        ax[0].set_title('amp map')
        ax[1].set_title('el map')
        fig.colorbar(a, ax=ax[0], label='Amplitude')
        fig.colorbar(e, ax=ax[1], label='Elevation')
        plt.show()

        fig, ax = pl.subplots(nrows=1, ncols=2, figsize=(15, 5))
        # a = ax[0].imshow(np.where(self.mask, self.amp1, np.nan), cmap = pl.cm.jet) #np.where((self.amp_mask & self.mask)
        avg_amp = (
                np.nanmean(
                    np.stack([
                        np.where(self.mask, self.amp1, np.nan),
                        np.where(self.mask, self.amp2, np.nan)
                    ]),
                    axis=0
                ) / 70
        )

        e = ax[1].imshow(
            avg_amp,
            cmap=pl.cm.jet,
            vmin=0,
            vmax=1,
        )

        ax[1].set_title('Normalized avg amp map')
        fig.colorbar(e, ax=ax[1], label='Amplitude / 70')
        plt.show()

        plt.figure(figsize = (5,5))
        # a = ax[0].imshow(np.where(self.mask, self.amp1, np.nan), cmap = pl.cm.jet) #np.where((self.amp_mask & self.mask)
        plt.imshow(sign_map, cmap = pl.cm.jet,vmin = -1, vmax = 1)
        # plt.set_title('amp map')
        # ax[1].set_title('el map')
        plt.colorbar(shrink = 0.4)

        plt.show()

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

        plt.show()

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
        # data_ds_rel = data_ds - np.nanmean(data_ds, axis=0)

        # do delta F / F
        data_ds_rel = (data_ds - np.nanmean(data_ds, axis=0)) / np.nanmean(data_ds, axis=0)

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


def run_retmap(config_path, mode, min_area = 100, animal_dob=None,animal=None, day=None, draw_mask = True, draw_lines = True):
    """ Run the Retinotopic Mapping script with given parameters. """
    params = load_parameters(config_path)

    # Optionally add the animal and day to the params dictionary for later use
    if animal is not None:
        params['animal'] = animal
    if day is not None:
        params['day'] = day
        params['animal_dob'] = animal_dob

    params['savefolder'] = params['savefolder'].replace(r"F:\Erica", r"J:")
    print('Running Retinotopic_mapping.py %s\t' % (version), datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    print('Config file: ', config_path)
    print('Running mode %s' % (mode))
    print('Analysis folder: ', params['savefolder'], '\n')

    if mode == '1':
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

    elif mode == '2':
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
                                  rotateMap=int(params['rotateMap']),
                                  min_area = min_area,
                                  animal_dob = animal_dob,
                                  animal = animal,
                                  day = day,
                                  draw_mask = draw_mask,
                                  draw_lines = draw_lines)
        patchmap.run()
        #figraw, axraw = patchmap.plot_raw() # amplitude map
        fig, ax = patchmap.plot() # sign map

        pl.show()
    else:
        raise ValueError("Mode has to be either 1 (Create complex fields) or 2 (Create and plot sign map)!")


def batch_retmap(animals_days_dict, screen, path_to_retmap, birth_dates=None, draw_mask=False, draw_lines = True):
    for animal in animals_days_dict.keys():
        for day in animals_days_dict[animal]:
            for subfile in [item for item in os.listdir(os.path.join(path_to_retmap, animal, day)) if
                            os.path.isdir(os.path.join(path_to_retmap, animal, day, item)) and (screen in item)]:
                config_path = os.path.join(path_to_retmap, animal, day, subfile, 'config.txt')
                print(config_path)
                mode = "2"
                if birth_dates:
                    run_retmap(config_path, mode, min_area=250, animal_dob=birth_dates[animal], animal=animal,
                               day=day, draw_mask=draw_mask, draw_lines = draw_lines)
                else:
                    run_retmap(config_path, mode, min_area=250, animal=animal, day=day, draw_mask=draw_mask, draw_lines = draw_lines)
    #plt.close('all')

