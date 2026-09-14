import matplotlib.pyplot as plt
from classes import *
from helpers import *
import numpy as np
import scipy.io
import os
import matplotlib.pyplot as plt

animals_days = {'EC_phpeb_11': ['20241113'],
                #'EC_GCaMP6s_12': ['20250123'],
                'EC_phpeb_13': ['20250123'],
                'EC_phpeb_14': ['20260414'],
                'EC_phpeb_15': ['20260416'],
                'EC_phpeb_17': ['20260421'],
                'EC_phpeb_18': ['20260421'],
                'EC_phpeb_19': ['20260421'],
                'EC_GNAT_03': ['20240924'],
                'EC_GNAT_05': ['20240924'],
                'EC_GNAT_06': ['20240924'],
                'EC_GNAT_04': ['20241004'],
                }

#animals_days = {'EC_GCaMP6s_06': ['20241121']}
data_object = DataAnalysis (['G:\\vision_restored'], dict_animals_days = animals_days, response_type = 'fluorescence', zscore_threshold =None,std_threshold = 1.645, dlc = False, show_plots = False)
#1.645 std threshold is 95%
# delete some duplicates
del data_object.dat['EC_phpeb_11']['20241113']['big100_000_001']
del data_object.dat['EC_phpeb_11']['20241113']['big100_000_005']
del data_object.dat['EC_phpeb_11']['20241113']['big100_000_008']
del data_object.dat['EC_phpeb_11']['20241113']['small_000_006']
del data_object.dat['EC_phpeb_11']['20241113']['small_000_007']
del data_object.dat['EC_GNAT_03']['20240924']['big100_000_002']
#del data_object.dat['EC_GCaMP6s_12']

responsive_cells_hist(data_object)


def load_sbx_frame(filepath, frame_idx, count=1):
    """
    Load one or more frames from a .sbx file as a NumPy array.
    filepath: path to file without .sbx extension
    frame_idx: the index of the first frame to load
    count: how many frames to load (default: 1)
    """
    # Load metadata from accompanying .mat file
    mat = scipy.io.loadmat(filepath + '.mat', simplify_cells=True)
    info = mat['info']
    width, height = info['sz']
    chan = info.get('channels', 1)
    nchan = 1 if chan in [1, 2] else 2
    scanmode = info.get('scanmode', 0)
    # if scanmode == 1:
    #     width *= 2

    bytes_per_frame = width * height * nchan * 2  # uint16 = 2 bytes
    offset = frame_idx * bytes_per_frame

    with open(filepath + '.sbx', 'rb') as f:
        f.seek(offset)
        data = np.fromfile(f, dtype=np.uint16, count=count * width * height * nchan)
        data = data.reshape((count, height, width, nchan))
        return np.squeeze(data)

def get_sbx_metadata(filepath):
    """Return metadata dictionary from .mat file"""
    return scipy.io.loadmat(filepath + '.mat', simplify_cells=True)['info']

# Path to your .sbx file, without extension
filename = fr"I:\vision_restored\data\EC_GCaMP6s_11\20250725\small_SFxO\small_SFxO_000_005\experiments\small_SFxO_000_005"

# Load frame 100
frame = load_sbx_frame(filename, frame_idx=100)

# Display
plt.imshow(frame[...,1], cmap='gray')
plt.title("Frame 100")
plt.axis('off')
plt.show()



for animal in data_object.dat:
    for day in data_object.dat[animal]:

        parameter_matrix_plot(data_object, animal, day, subfile, zscore_threshold=3)

fig, ax = plt.subplots (2,1, figsize = (5,8), sharey = True, sharex = True)
for i, animal in enumerate(['EC_phpeb_13', 'EC_GNAT_04']):
    for day in data_object.dat[animal]:
        for recording in [r for r in data_object.dat[animal][day].keys() if 'big100' in r and 'chirps' not in r]:

            colors = plt.cm.plasma(np.linspace(0, 0.8, 5))

            # shape (n_repeats, n_orientations, n_cells, n_timepoints) > (n_repeats, n_cells, n_timepoints)
            matrix = data_object.dat[animal][day][recording]['dfof_matrix_baseline'][:, 4]

            for i_cell in [5]:
                ax[i].plot(np.arange(matrix.shape[-1]), matrix[:, i_cell].mean(axis = 0), color = 'blue')

                for i_repeat in range(matrix.shape[0]):
                    ax[i].plot(np.arange(matrix.shape[-1]), matrix[i_repeat,i_cell], color='grey', alpha = 0.2)

            colors = plt.cm.plasma(np.linspace(0, 0.7, matrix.shape[1]))
            #ymin, ymax = ax[i].get_ylim()  # use get_ylim() instead of ax[i].ylim()
            ax[i].axvspan(data_object.fps, data_object.fps * 4, color='grey', alpha=0.3, label='Sighted')
            ax[i].set_title (animal.split('_')[1])
            #ax[i].set_yticks([])
            ax[i].set_xticks(np.arange(matrix.shape[-1])[::int(data_object.fps)])
            ax[i].set_xticklabels([int(np.round(x)) for x in ( np.arange(-data_object.fps, matrix.shape[-1] - data_object.fps)[::int(data_object.fps)]) / int(data_object.fps)])
            ax[i].set_xlabel ('Time since stimulus onset (s)')
            ax[i].set_ylabel('Z-scored response')
plt.show()

fig, ax = plt.subplots (2,1, figsize = (5,8), sharey = True, sharex = True)
for i, animal in enumerate(['EC_phpeb_13', 'EC_GNAT_04']):
    for day in data_object.dat[animal]:
        for recording in [r for r in data_object.dat[animal][day].keys() if 'big100' in r and 'chirps' not in r]:

            # shape (n_repeats, n_orientations, n_cells, n_timepoints) > (n_cells, n_timepoints)
            matrix = data_object.dat[animal][day][recording]['dfof_matrix_baseline'].mean(axis = (0,1))
            cells = np.array(random.sample(range(1, matrix.shape[0]), 15))  # pick 25 random cells
            for cell in cells:
                ax[i].plot(np.arange(matrix.shape[-1]), matrix[cell], color='grey', alpha = 0.5)

            ax[i].plot(np.arange(matrix.shape[-1]), matrix[cells].mean(axis = 0), color = 'blue')
            ymin, ymax = ax[i].get_ylim()  # use get_ylim() instead of ax[i].ylim()
            ax[i].axvspan(data_object.fps, data_object.fps * 4, color='grey', alpha=0.3, label='Sighted')
            ax[i].set_title (animal.split('_')[1])
            ax[i].set_xticks(np.arange(matrix.shape[-1])[::int(data_object.fps)])
            ax[i].set_xticklabels([int(np.round(x)) for x in ( np.arange(-data_object.fps, matrix.shape[-1] - data_object.fps)[::int(data_object.fps)]) / int(data_object.fps)])
            ax[i].set_xlabel ('Time since stimulus onset (s)')
            ax[i].set_ylabel('Z-scored response')
plt.show()


for i, animal in enumerate(['EC_GCaMP6s_13', 'EC_GNAT_03']):
    for day in data_object.dat[animal]:
        for recording in [r for r in data_object.dat[animal][day].keys() if 'big100' in r and 'chirps' not in r]:
            print(recording)

            # shape (n_repeats, n_orientations, n_cells, n_timepoints) > (n_cells, n_timepoints)
            matrix = data_object.dat[animal][day][recording]['zscored_matrix_baseline'].mean(axis = (0,1))

            plt.figure()
            plt.plot(np.arange(matrix.shape[-1]),matrix.mean(axis = 0))
            plt.show()

             # control: 1, 6, 10
            # for i in range(20):
            #     plt.figure()
            #     plt.plot(np.arange(matrix.shape[-1]), matrix[i])
            #     plt.title(animal+ str(i))
            #     plt.show()
