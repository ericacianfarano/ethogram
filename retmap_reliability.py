import os, argparse
from datetime import datetime
import numpy as np
from timeit import default_timer as timer
import skvideo.io
import skimage.transform as transform
from skimage.morphology import skeletonize
from skimage import exposure
from scipy.io import loadmat
import numpy as np
from itertools import combinations
from math import comb

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
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft, signal
from scipy.ndimage import gaussian_filter, binary_fill_holes
from skimage.morphology import convex_hull_image

from Retinotopic_mapping import (
    load_parameters,
    load_data,
    sbx_get_ttlevents,
    load_maps,
    rotate_image
)

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft, signal
from scipy.ndimage import gaussian_filter, binary_fill_holes
from skimage.morphology import convex_hull_image

from Retinotopic_mapping import (
    load_parameters,
    load_data,
    sbx_get_ttlevents,
    load_maps,
    rotate_image
)


def get_filtered_data(fn, evt_suffix, framerate, timetot, nrep, resize=1):
    '''
    Based on _get_phase_map() function from original Retinotopic_mapping.py script. Only does first part of this original function.

    This function loads the movie during the stimulus period and preprocesses/filters (subtracts mean image over time, removes slow drift, returns cleaned movie_.
    :param fn: path to one mj2 file
    :param evt_suffix: suffix for scanbox ttl/event file
    :param framerate:
    :param timetot: total stimulus time for that direction (timetot_a for azimuth, timetot_e for elevation)
    :param nrep: number of stim sweeps/reptitions
    :param resize: how much we downsample moving for speed
    :return:
        data_filt: filtered movie, shape frames x height x width
        T: timetot / n_rep > duration of one sweep in seconds
    '''

    T = timetot / nrep
    nimagtot = int(round(timetot * framerate))

    # Data loading: load TTLs and image frames
    # Only load image frames during stimulus presentation. Frame number extracted from events file
    # Also downsample data for performance
    evt = sbx_get_ttlevents(fn + evt_suffix)
    start_frame = evt[evt > 0][0] - 1  # -1 to start counting from 0
    data = load_data(fn, start_frame=start_frame, end_frame=nimagtot,
                     resize=resize)  # resisze is 0.5 in original function

    print('File loaded')

    # Remove DC component of image (i.e. average intensity value) (i.e., remove slow, low-frequency drift)
    # data_rel = data - np.nanmean(data, axis=0)

    # do delta F / F
    print('CALCULATING DELTA F OVER F VERSION')
    data_rel = (data - np.nanmean(data, axis=0)) / np.nanmean(data, axis=0)

    # Apply high pass filter (subtract out frequencies below 2 cycles of the stimulus)
    cutfreq = 1 / (2 * T)
    b, a = signal.butter(1, cutfreq / (0.5 * framerate), 'low')
    data_lowfreq = signal.filtfilt(b, a, data_rel, axis=0)
    data_filt = data_rel - data_lowfreq

    print('Preprocessing completed')

    return data_filt, T


def compute_phase_reliability(data_filt, framerate, T):
    '''
    Based on _get_phase_map() function from original Retinotopic_mapping.py script. Only does second part of this original function.

    :param data_filt: filtered movie, output from get_filtered_data (shape frames x height x width)
    :param framerate:
    :param T: timetot / n_rep > output from get_filtered_data,  duration of one sweep in seconds
    :return:
        - R: phase reliability map, shape (height x width)
            R -> 1 means phase is consistent across sweeps.
            R -> 0 means phase is random/jittery across sweeps
        - phase_sweeps: phase map for every individual sweep, shape (nrep x height x width) -> trial-by-trial phase maps
        - amp_sweeps: amplitude map for every indidivudal sweep, shape (nrep x height x width) -> trial-by-trial amp map
    '''
    frames_per_sweep = int(round(T * framerate))  # x sec / sweep * frames / sec = frames / sweep
    nrep = data_filt.shape[0] // frames_per_sweep  # how many full sweeps/repeats fit inside movie

    data_filt = data_filt[:nrep * frames_per_sweep]  # trim off excess frames at the end

    # reshape movie from shape (frames x height x weidth) -> (repeat x frames per sweep x height x weight)
    sweeps = data_filt.reshape(
        nrep,
        frames_per_sweep,
        data_filt.shape[1],
        data_filt.shape[2]
    )

    # Perform Fourier transform  > across the time axis within each sweep / repeat
    sweep_fft = fft.fft(sweeps, axis=1)  # shape (n_rep x frequency x height x width)
    freq = fft.fftfreq(frames_per_sweep, d=1 / framerate)  # frequency values corresponding to fft bins
    print('Fourier completed')

    # Extract complex field at first harmonic frequency of the stimulus
    fstim = 1 / T  # stim frequency
    f_ind = np.argmin(np.abs(
        freq - fstim))  # index of stim frequency > find the FFT bin closest to stim frequency (in original script, its np.where(freq >= fstim)[0][0])
    fft_at_stim = sweep_fft[:, f_ind, :,
                  :]  # extract complex fourier response at the stim frequency for every sweep/repeat and pixel (shape nrep x height x weidth)

    phase_sweeps = np.angle(fft_at_stim)  # phase angle for each sweep/pixel
    amp_sweeps = np.abs(fft_at_stim)  # amplitude for each sweep/pixel > how strongly did the pixel respond?

    # Caclulate Phase Reliability > did the pixel respond at the same phase/position every sweep? will be low if unreliable, even if amplitude is sufficient
    R = np.abs(np.nanmean(np.exp(1j * phase_sweeps), axis=0))
    # p.exp(1j * phase_sweeps) turns each phase angle into a unit-length complex vector (discarding amplitude) (both weak & strong responses have a length of 1)
    # if we then average the vectors, with nanmean, an unreliable phase map will bounce around negative and positive values, and average out very weakly
    # then np.abs takes the length of that average arrow (R)

    return R, phase_sweeps, amp_sweeps


def make_responsive_roi(amp_map, percentile=75, use_convex_hull=True):
    thresh = np.nanpercentile(amp_map, percentile)
    roi = amp_map > thresh

    roi = binary_fill_holes(roi)

    if use_convex_hull:
        roi = convex_hull_image(roi)

    return roi

def compute_fill_fraction(responsive_roi, segmented_map):
    segmented = segmented_map > 0

    numerator = np.sum(segmented & responsive_roi)
    denominator = np.sum(responsive_roi)

    if denominator == 0:
        return np.nan

    return numerator / denominator

def plot_maps(amp_map, R_map, responsive_roi, segmented_map, save_path, title=''):
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))

    im0 = ax[0].imshow(amp_map, cmap='gray')
    ax[0].set_title('Amplitude map')
    plt.colorbar(im0, ax=ax[0], fraction=0.046)

    im1 = ax[1].imshow(R_map, cmap='viridis', vmin=0, vmax=1)
    ax[1].set_title('Phase reliability R')
    plt.colorbar(im1, ax=ax[1], fraction=0.046)

    ax[2].imshow(responsive_roi, cmap='gray')
    ax[2].set_title('Responsive ROI')

    ax[3].imshow(responsive_roi, cmap='gray')
    ax[3].imshow(segmented_map > 0, alpha=0.5)
    ax[3].set_title('Segmented fill')

    for a in ax:
        a.axis('off')

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


if __name__ == '__main__':

    screen = 'big100'
    path = r'J:\retmap'

    animals_days = {
        'EC_phpeb_11': ['20241111'],
        'EC_phpeb_12': ['20250121'],
        'EC_phpeb_13': ['20250121'],
        'EC_phpeb_14': ['20260410'],
        'EC_phpeb_15': ['20260410'],
        'EC_phpeb_17': ['20260421'],
        'EC_phpeb_18': ['20260421'],
        'EC_phpeb_19': ['20260421'],
        'EC_GNAT_01': ['20240805'],
        'EC_GNAT_02': ['20240805'],
        'EC_GNAT_03': ['20240923'],
        'EC_GNAT_04': ['20241003'],
        'EC_GNAT_05': ['20240923'],
        'EC_GNAT_06': ['20240923']}

    d = {animal: {'azimuth': [], 'elevation': [] } for animal in animals_days}

    for animal in animals_days.keys():
        for day in animals_days[animal]:
            for subfile in [item for item in os.listdir(os.path.join(path, animal, day)) if
                            os.path.isdir(os.path.join(path, animal, day, item)) and (screen in item)]:
                config_path = os.path.join(path, animal, day, subfile, 'config.txt')

                print(f'Processing {animal}, {day}, {subfile}')

                # config_path = r'PATH_TO_YOUR_CONFIG.txt'
                params = load_parameters(config_path)

                # change the drive, and also remove the folder pointer
                inputfolder = params['inputfolder'].replace('F:', 'J:').replace(r'\Erica', '')
                savefolder = params['savefolder'].replace('F:', 'J:').replace(r'\Erica', '')

                savename = params['savename']

                framerate = float(params['framerate'])
                nrep = int(params['nrep'])
                evt_suffix = params['evt_suffix']

                timetot_a = float(params['timetot_a'])
                timetot_e = float(params['timetot_e'])

                template_a = params['template_a']
                template_e = params['template_e']

                # Load existing maps
                phase_a, amp_a, phase_e, amp_e, reference = load_maps(
                    os.path.join(savefolder, savename + '_fftmaps.npy')
                )

                # load drawn on mask
                mask = np.load(os.path.join(savefolder, savename + '_mask.npy'))

                # Load segmentation from mode 2 output
                sign_map_path = os.path.join(savefolder, savename + '_signmap.npy')
                sign_map = np.load(sign_map_path)

                # For now, you may want to replace this with raw_patches
                # saved from your RetinotopicMap object later.
                segmented_map = np.abs(sign_map) > np.nanstd(sign_map)

                # Find azimuth files
                import glob

                files_a = sorted(glob.glob(os.path.join(inputfolder, template_a)))
                files_e = sorted(glob.glob(os.path.join(inputfolder, template_e)))

                R_maps_a = []

                # processing the two azimuth files
                for fn in files_a:
                    data_filt, T = get_filtered_data(
                        fn,
                        evt_suffix=evt_suffix,
                        framerate=framerate,
                        timetot=timetot_a,
                        nrep=nrep
                    )

                    R, phase_sweeps, amp_sweeps = compute_phase_reliability(
                        data_filt,
                        framerate=framerate,
                        T=T
                    )

                    R = np.where(mask, R, np.nan)
                    R_maps_a.append(R)

                d[animal]['azimuth'] = np.array(R_maps_a)

                R_a = np.nanmean(R_maps_a, axis=0)

                R_maps_e = []

                # processing the two elevation files
                for fn in files_e:
                    data_filt, T = get_filtered_data(
                        fn,
                        evt_suffix=evt_suffix,
                        framerate=framerate,
                        timetot=timetot_e,
                        nrep=nrep
                    )

                    R, phase_sweeps, amp_sweeps = compute_phase_reliability(
                        data_filt,
                        framerate=framerate,
                        T=T
                    )

                    R = np.where(mask, R, np.nan)
                    R_maps_e.append(R)

                d[animal]['elevation'] = np.array(R_maps_e)

                R_e = np.nanmean(R_maps_e, axis=0)

    for animal in d:
        a1, a2 = d[animal]['azimuth'][0], d[animal]['azimuth'][1]
        e1, e2 = d[animal]['elevation'][0], d[animal]['elevation'][1]

        # # plot all 4 reliability maps
        # fig, ax = plt.subplots(2, 2, figsize=(5, 4))
        # ax = ax.ravel()
        # maps = [a1, a2, e1, e2]
        # for i, m in enumerate(maps):
        #     im = ax[i].imshow(m, vmin=0, vmax=1)
        #     fig.colorbar(im, ax=ax[i], shrink = 0.6)
        # plt.tight_layout()
        # plt.suptitle(animal)
        # plt.show()

        # plot the average of the 4 reliability maps
        plt.figure(figsize=(4, 3))
        maps = np.array([a1, a2, e1, e2]).mean(axis=0)
        plt.imshow(maps, vmin=0, vmax=1, cmap=pl.cm.jet)
        plt.colorbar(shrink=0.7)
        plt.tight_layout()
        plt.title(animal)
        plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    groups = np.unique([a.split('_')[1] for a in d])[::-1]

    colors = {
        'phpeb': '#EE2926',
        'GNAT': '#477BBF'
    }

    for i, group in enumerate(groups):

        color = colors.get(group, 'gray')

        animals_in_group = [a for a in d if a.split('_')[1] == group]

        for animal in animals_in_group:
            azimuth_mean = np.nanmean(d[animal]['azimuth'])
            elevation_mean = np.nanmean(d[animal]['elevation'])

            x = i + np.random.uniform(-0.08, 0.08)

            ax[0].scatter(x, azimuth_mean, color=color)
            ax[1].scatter(x, elevation_mean, color=color)

    for a in ax:
        a.set_xticks(range(len(groups)))
        a.set_xticklabels(groups)
        a.set_ylim(0, 1)

    ax[0].set_title("Azimuth")
    ax[1].set_title("Elevation")
    ax[0].set_ylabel("Mean phase reliability")

    plt.tight_layout()
    plt.show()

    def fisher_pitman_exact(x, y):
        """
        Exact two-sided Fisher-Pitman permutation test using the
        difference in means as the test statistic.

        Returns
        -------
        statistic : float
            Observed mean(x) - mean(y).
        z : float
            Standardized Fisher-Pitman statistic.
        p : float
            Exact two-sided permutation p-value.
        n_permutations : int
            Number of possible group assignments.
        """

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]

        if len(x) == 0 or len(y) == 0:
            raise ValueError("Both groups must contain at least one valid value.")

        combined = np.concatenate([x, y])

        n_x = len(x)
        n_y = len(y)
        n_total = n_x + n_y

        observed = np.mean(x) - np.mean(y)

        permutation_statistics = []

        for x_indices in combinations(range(n_total), n_x):
            x_indices = np.asarray(x_indices)

            mask = np.ones(n_total, dtype=bool)
            mask[x_indices] = False

            perm_x = combined[x_indices]
            perm_y = combined[mask]

            permutation_statistics.append(
                np.mean(perm_x) - np.mean(perm_y)
            )

        permutation_statistics = np.asarray(permutation_statistics)

        # Exact two-sided p-value
        tolerance = 1e-12
        p = np.mean(
            np.abs(permutation_statistics)
            >= np.abs(observed) - tolerance
        )

        # Standardized Fisher-Pitman Z statistic
        total_mean = np.mean(combined)

        T = np.sum(x)
        E = n_x * total_mean

        V = (
                (n_x * n_y)
                / (n_total * (n_total - 1))
                * np.sum((combined - total_mean) ** 2)
        )

        if V > 0:
            z = (T - E) / np.sqrt(V)
        else:
            z = np.nan

        return observed, z, p, comb(n_total, n_x)

    #######################################################

    plt.figure(figsize=(4, 4))
    groups = np.unique([a.split('_')[1] for a in d])[::-1]

    colors = {
        'phpeb': '#EE2926',
        'GNAT': '#477BBF'
    }

    # put data into list so that we can do stats on them and plot
    group_data = []
    for group in groups:
        values = []
        animals_in_group = [a for a in d if a.split('_')[1] == group]
        for animal in animals_in_group:
            values.append(np.mean([np.nanmean(d[animal]['azimuth']), np.nanmean(d[animal]['elevation'])]))
        group_data.append(values)

    bp = plt.boxplot(
        group_data,
        positions=np.arange(len(groups)),
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        whis=1.5,
        medianprops=dict(
            color='black',
            linewidth=1.5
        )
    )

    for patch, group in zip(bp['boxes'], groups):
        patch.set_facecolor(colors.get(group, 'gray'))
        patch.set_alpha(0.3)
        patch.set_edgecolor(colors.get(group, 'gray'))

    # add scatter plots and SEM
    for i, (group, values) in enumerate(zip(groups, group_data)):
        color = colors.get(group, 'gray')

        x = np.random.uniform(i - 0.08, i + 0.08, len(values))
        plt.scatter(x, values, color=color, s=35, zorder=3, alpha=0.7)

    if len(group_data) == 2:

        a = np.asarray(group_data[0], dtype=float)
        b = np.asarray(group_data[1], dtype=float)

        statistic, z, p, n_permutations = fisher_pitman_exact(a, b)

        print(
            f"{groups[0]} vs {groups[1]}: "
            f"{len(a)} vs {len(b)} animals, "
            f"{n_permutations} possible assignments"
        )

        print(
            f"Exact Fisher-Pitman permutation test: "
            f"mean difference = {statistic:.4f}, "
            f"Z = {z:.3f}, "
            f"p = {p:.4f}"
        )

        # Significance bar
        clean_a = a[np.isfinite(a)]
        clean_b = b[np.isfinite(b)]

        ymax = max(np.max(clean_a), np.max(clean_b))
        y = ymax + 0.05

        plt.plot(
            [0, 0, 1, 1],
            [y, y + 0.02, y + 0.02, y],
            color='black'
        )

        if p < 0.001:
            text = "***"
        elif p < 0.01:
            text = "**"
        elif p < 0.05:
            text = "*"
        else:
            text = f"p = {p:.2f}"

        plt.text(
            0.5,
            y + 0.025,
            text,
            ha='center',
            va='bottom'
        )

    plt.xticks(range(len(groups)), groups)
    plt.ylabel("Mean phase reliability")
    plt.title("Phase reliability (A&E)")
    plt.ylim(0.35, 1.05)

    plt.tight_layout()
    plt.show()

    ####################################################

    plt.figure(figsize=(4, 4))

    groups = np.unique([a.split('_')[1] for a in d])[::-1]

    colors = {
        'phpeb': '#EE2926',
        'GNAT': '#477BBF'
    }

    display_names = {
        'phpeb': 'WT',
        'GNAT': r'GNAT1/2$^{\mathit{mut}}$'
    }

    # Put data into list so that we can do stats and plot
    group_data = []

    for group in groups:

        values = []

        animals_in_group = [
            a for a in d
            if a.split('_')[1] == group
        ]

        for animal in animals_in_group:
            values.append(
                np.mean([
                    np.nanmean(d[animal]['azimuth']),
                    np.nanmean(d[animal]['elevation'])
                ])
            )

        group_data.append(values)

    # Bring the two groups closer together
    positions = np.array([0.0, 0.65])

    bp = plt.boxplot(
        group_data,
        positions=positions,

        # Thicker boxes
        widths=0.30,

        patch_artist=True,
        showfliers=False,
        whis=1.5,

        medianprops=dict(
            color='black',
            linewidth=2.0
        ),

        # Thinner black box outline
        boxprops=dict(
            color='black',
            linewidth=1.5
        ),

        whiskerprops=dict(
            color='black',
            linewidth=1.5
        ),

        capprops=dict(
            color='black',
            linewidth=1.5
        )
    )

    for patch, group in zip(bp['boxes'], groups):
        patch.set_facecolor(
            colors.get(group, 'gray')
        )

        # Solid box fill
        patch.set_alpha(1)

        # Black outline
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)

    # Reproducible point jitter
    rng = np.random.default_rng(42)

    # Add scatter plots
    for i, (group, values) in enumerate(
            zip(groups, group_data)
    ):
        color = colors.get(group, 'gray')

        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]

        # Small jitter
        jitter = rng.uniform(
            -0.04,
            0.04,
            len(values)
        )

        # Shift points slightly to the left of the box
        point_offset = -0.22

        x = (
                np.full(len(values), positions[i])
                + point_offset
                + jitter
        )

        plt.scatter(
            x,
            values,
            color=color,
            s=42,
            zorder=3,
            alpha=1,
            edgecolors='none'
        )

    # Exact Fisher-Pitman permutation test
    if len(group_data) == 2:

        a = np.asarray(group_data[0], dtype=float)
        b = np.asarray(group_data[1], dtype=float)

        statistic, z, p, n_permutations = fisher_pitman_exact(
            a,
            b
        )

        print(
            f"{groups[0]} vs {groups[1]}: "
            f"{len(a)} vs {len(b)} animals, "
            f"{n_permutations} possible assignments"
        )

        print(
            f"Exact Fisher-Pitman permutation test: "
            f"mean difference = {statistic:.4f}, "
            f"Z = {z:.3f}, "
            f"p = {p:.4f}"
        )

        # Significance bar
        clean_a = a[np.isfinite(a)]
        clean_b = b[np.isfinite(b)]

        ymax = max(
            np.max(clean_a),
            np.max(clean_b)
        )

        y = ymax + 0.05

        plt.plot(
            [
                positions[0],
                positions[0],
                positions[1],
                positions[1]
            ],
            [
                y,
                y + 0.02,
                y + 0.02,
                y
            ],
            color='black',
            linewidth=1.5
        )

        if p < 0.001:
            text = "***"
        elif p < 0.01:
            text = "**"
        elif p < 0.05:
            text = "*"
        else:
            text = f"p = {p:.2f}"

        plt.text(
            np.mean(positions),
            y + 0.025,
            text,
            ha='center',
            va='bottom',
            fontsize=16,
            fontfamily='Arial'
        )

    plt.xticks(
        positions,
        [display_names[group] for group in groups],
        fontsize=14,
        fontfamily='Arial'
    )

    plt.ylabel(
        "Mean phase reliability",
        fontsize=16,
        fontfamily='Arial'
    )

    plt.title(
        "Phase reliability (A&E)",
        fontsize=17,
        fontfamily='Arial',
        pad=14
    )

    plt.ylim(0.35, 1.05)

    # Keep the closer boxes centered
    plt.xlim(-0.45, 1.10)

    ax = plt.gca()

    ax.set_facecolor('white')

    # Only retain the left axis spine
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(2)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.grid(False)

    ax.tick_params(
        axis='y',
        direction='out',
        length=6,
        width=2,
        colors='black',
        labelsize=14
    )

    ax.tick_params(
        axis='x',
        length=0,
        pad=10
    )

    plt.tight_layout()
    plt.show(block=True)

