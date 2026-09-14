import matplotlib.pyplot as plt
import numpy as np

from imports import *
from plotting import *

def suite2p_files (suite2p_path, response_type = 'deconvolved'):
    '''
    :param response_type: 'deconvolved' or 'fluorescence' - specifies if we want deconvolved (binarized) traces or raw fluorescnece values

    :param suite2p_path: path to the suite2p plane0 folder
    :return: neuronal responses file, is cell file, ops file, stat file

    - F.npy > array of fluorescence traces (ROIs by timepoints)
    - Fneu.npy > array of neuropil fluorescence traces (ROIs by timepoints)
    - spks.npy > array of deconvolved traces (ROIs by timepoints)
    - stat.npy > list of statistics computed for each cell (ROIs by 1)
    - ops.npy > options and intermediate outputs (dictionary)
    -
    '''

    if response_type == 'deconvolved': # spks.npy > array of deconvolved traces (ROIs by timepoints) (generally cleaner)
        responses_file = np.load(os.path.join(suite2p_path, 'spks.npy'), allow_pickle=True)  # array of fluorescence traces (cells x timepoints)
    elif response_type == 'fluorescence': # F.npy > array of fluorescence traces (ROIs by timepoints)
        # if you load in F, you need to subtract the neuropil (F - 0.7*Fneu) > this is the real fluorescence signal
        F = np.load(os.path.join(suite2p_path, 'F.npy'),allow_pickle=True)  # array of fluorescence traces (cells x timepoints)
        Fneu = np.load(os.path.join(suite2p_path, 'Fneu.npy'), allow_pickle=True)  # array of fluorescence traces (cells x timepoints)

        responses_file = F - (0.7 * Fneu)  # array of fluorescence traces (cells x timepoints)

    #iscell.npy > specifies whether an ROI is a cell, first column is 0/1, and second column is probability that the ROI is a cell based on the default classifier
    iscell_file = np.load(os.path.join(suite2p_path, 'iscell.npy'), allow_pickle=True)[:,0]  # specifies whether an ROI is a cell, either 0/1

    # ops.npy > options and intermediate outputs (dictionary)
    ops_file = np.load(os.path.join(suite2p_path, 'ops.npy'), allow_pickle=True).item()

    #stat.npy > list of statistics computed for each cell (ROIs by 1)
    stat_file = np.load(os.path.join(suite2p_path, 'stat.npy'), allow_pickle=True)

    return responses_file, iscell_file, ops_file, stat_file

def load_stims (path_to_log):
    '''
    Reads all lines in log file
    :param path_to_log:
    :return:
    - stim_dict: ex, {'Wait 0': ['Wait600.00'], 'GratingStim 0': ['grating_143_SF0.02_TF1.00', 'grating_323_SF0.02_TF1.00',...]...}
    - list_stim_types: ex, ['Wait', 'GratingStim', 'GratingStim', 'GratingStim', 'GratingStim', 'GratingStim', 'GratingStim', ...]
    - list_stim_names ex, ['Wait600.00', 'grating_143_SF0.02_TF1.00', 'grating_323_SF0.02_TF1.00', 'grating_359_SF0.02_TF1.00',...]
    '''

    # read through all the line in the log file, and add each line to 'list_stims'
    list_stims = []
    with open(path_to_log) as f:
        lines = f.readlines()  # all the lines in the log file

        for i, l in enumerate(lines):  # enumerate through all lines in log file
            if i > 0:  # ignore first line, which just has info on experiment
                l_new = ast.literal_eval(l)
                list_stims.append(l_new)  # store in list

    # store the name and type of each stimulus in sepearate lists
    list_stim_types = [stim['type'] for stim in list_stims]
    list_stim_names = [stim['name'] for stim in list_stims]
    list_stim_params = [stim['params'] for stim in list_stims]

    # number of wait stimuli we show
    n_wait_times = len([stim for stim in list_stim_names if 'Wait' in stim])

    #number of orientations in a 'repeat'
    ntheta = len(np.unique([stim for stim in list_stim_names if 'Wait' not in stim]))
    #ntheta = len(np.unique(list_stim_names)) - 2 # subtract n_wait_times to account for however many wait times
    nrepeats = int((len(list_stim_names)-1)/ntheta)

    # create a dictionary where we sort the stimulus block together with the names of the stimuli (wait stims together, and grating stims together, in blocks of ntheta)
    stim_dict = {}
    wait_count, grating_count = 0, 0
    for i_stim, stim in enumerate(list_stim_types):

        if 'Wait' in stim:
            stim_dict[f'{stim} {wait_count}'] = [list_stim_names[i_stim]]
            wait_count += 1

        elif ('Grating' in stim):
            if (grating_count%ntheta) == 0:
                list_gratings = []
                for i_grating in np.arange(i_stim, i_stim+ntheta):
                    list_gratings.append(list_stim_names[i_grating])
                stim_dict[f'{stim} {grating_count//ntheta}'] = list_gratings
            grating_count += 1

    #return stim_dict, list_stim_types[1:], list_stim_names[1:], ntheta, nrepeats
    return stim_dict, list_stim_types, list_stim_names, ntheta, nrepeats

def check_ttls (obj, ttl_arr, stim_names_list):
    '''
    Check if ttls are aligned

    :param ttl_arr:
    :return: corrected ttl_array, if necessary
    '''
    # if NOT FFF stims
    #if not (('black.png' in stim_names_list) and ('white.png' in stim_names_list)):
    difference = np.diff(ttl_arr)

    n_ttls_needed = len(stim_names_list) * 2

    # since each recording is flanked by a long wait time, the first and last 'difference' period should be very long
    # check that the wait times are each over 400 frames long and that all ttl periods are less than 200 frames long (20fps * 5s = 100frames) (i.e., all stim periods are less than 10 seconds)
    if (difference[0] > 400) and (difference[-1] > 400) and (np.all(np.diff(ttl_arr)[1:-1] < (obj.fps * 10))):
        if len(ttl_arr) + 1 == n_ttls_needed: # missing ONE ttl between end of wait and first stim
            ttl_arr = np.insert(ttl_arr, 2, ttl_arr[1])

    # have 2 less ttls than we need (missing start of wait and end of wait)
    elif (len(ttl_arr) + 2 == n_ttls_needed):
        #add duplicate 'start'
        ttl_arr = np.insert(ttl_arr, 0, ttl_arr[0])

        # there is a ttl missing, so we'd want to add a 0 at the start
        ttl_arr = np.insert(ttl_arr, 0, 0)

    # have 1 less ttls than we need (missing start of first stim / end of wait)
    elif ((len(ttl_arr) + 1 == n_ttls_needed) and (ttl_arr[1]!=ttl_arr[2])):
        #add duplicate 'start'
        ttl_arr = np.insert(ttl_arr, 1, ttl_arr[1])

    # have 1 MORE ttl than we need (extra start?)
    elif ((len(ttl_arr) - 1 == n_ttls_needed) and (ttl_arr[1]!=ttl_arr[2])):
        #remove duplicate 'start'
        ttl_arr = ttl_arr[1:]

    # else:
    #     # have 1 less ttls than we need (missing start of first stim / end of wait)
    #     n_ttls_needed = len(stim_names_list) * 2
    #     if ((len(ttl_arr) + 1 == n_ttls_needed) and (ttl_arr[1]!=ttl_arr[2])):
    #         #add duplicate 'start'
    #         ttl_arr = np.insert(ttl_arr, 1, ttl_arr[1])
    #     # if the first ttl period is longer than 4s
    #     if np.diff(ttl_arr)[0] > obj.fps * 4:
    #         ttl_arr[0] = ttl_arr[1] - obj.fps * 4

    # if we're cutting out the start and the end of the recording, the ttls need to start at '0'
    # if we dont cut out the start and end, dont shift TTLs
    return ttl_arr# - ttl_arr[0]


def zip_ttl_data(obj, object_datsubject_day):
    '''
    Put ttl data into a tuple of (stim_type, stim_name, [ttl_start, ttl_stop], ...)

    :param object_datsubject_day: example: suite2p_obj.dat_subject['20231106']
    :return:
    '''

    stim_names = object_datsubject_day['log']['list_stim_names']                            # list of stimuli names, ex: [ 'Wait600.00', 'ILSVRC2012_val_00000385.JPEG', 'grating_240_SF0.02_TF1.00'...]
    stim_types = object_datsubject_day['log']['list_stim_types']                            # list of stimuli types, ex: [ 'Wait', 'ImageStim', 'GratingStim'...]

    ttl = check_ttls (obj, object_datsubject_day['ttl_data'], stim_names)
    object_datsubject_day['ttl_data'] = ttl                                            # 1d array of TTLs, after being corrected for data collection errors, ex:      [ 102 6086 6086 6094...]

    zipped_ttl_dat = []
    for i, (stim_name, stim_type) in enumerate(zip(stim_names, stim_types)):
        zipped_ttl_dat.append((stim_type, stim_name, [ttl[i * 2], ttl[(2 * i) + 1]]))

    return zipped_ttl_dat

def get_ttl_dicts (obj, obj_dat_subject_day):
    '''

    :param obj_dat_subject_day: example: suite2p_obj.dat_subject['20231106']
    :return:
        - dict_stim_names (dict): dictionary with stimulus type (key) and list of stimuli names (value) for each 'block'
        - dict_stim_ttls (dict): dictionary with stimulus type (key) and array of ttls (value) for each 'block' (shape n_stim_presentations x 2)
    '''

    ttl_tuple = zip_ttl_data(obj, obj_dat_subject_day)

    obj_dat_subject_day['dict_stim_names'] = {}
    obj_dat_subject_day['dict_stim_ttls'] = {}

    # if we are showing gratings
    if (obj_dat_subject_day['n_SF'] and obj_dat_subject_day['n_TF'] and obj_dat_subject_day['n_theta']):

        wait_count, grating_count = 0, 0
        for i_zip in range(len(ttl_tuple)):

            if 'Wait' == ttl_tuple[i_zip][0]:
                type = ttl_tuple[i_zip][0]  # stimulus type, eg 'WaitStim'
                obj_dat_subject_day['dict_stim_names'][f'{type}_{wait_count}'] = ttl_tuple[i_zip][1] #ttl_tuple[i_zip][1] # specifications, eg Grating_90_sf0.02, etc
                obj_dat_subject_day['dict_stim_ttls'][f'{type}_{wait_count}'] = [ttl_tuple[i_zip][2]]  #ttl_tuple[i_zip][2] #list of ttls
                wait_count += 1

            # grating stim, and the remainder is 1 (to account for the wait)
            elif (i_zip % obj_dat_subject_day['ntheta']) == 1:
                #print('ooop', print(i_zip))
                z_names = [ttl_tuple[i][1] for i in range(i_zip, i_zip + obj_dat_subject_day['ntheta'])]
                z_ttls = [ttl_tuple[i][2] for i in range(i_zip, i_zip + obj_dat_subject_day['ntheta'])]

                type = ttl_tuple[i_zip][0]                  # stimulus type, eg 'GratingStim'
                obj_dat_subject_day['dict_stim_names'][f'{type}_{grating_count}'] = z_names # specifications, eg Grating_90_sf0.02, etc
                obj_dat_subject_day['dict_stim_ttls'][f'{type}_{grating_count}'] = z_ttls #list of ttls
                grating_count += 1

    else:   # showing grey/black/white/black/grey
        repeat = 0
        n_stims_per_repeat = 5

        for i_zip in range(len(ttl_tuple)):
            if i_zip % n_stims_per_repeat == 0:
                z_names = [ttl_tuple[i][1] for i in range(i_zip, i_zip + n_stims_per_repeat)]
                z_ttls = [ttl_tuple[i][2] for i in range(i_zip, i_zip + n_stims_per_repeat)]

                obj_dat_subject_day['dict_stim_names'][f'Repeat_{repeat}'] = z_names # specifications, eg Grating_90_sf0.02, etc
                obj_dat_subject_day['dict_stim_ttls'][f'Repeat_{repeat}'] = z_ttls #list of ttls
                repeat += 1

    return obj_dat_subject_day['dict_stim_ttls']

def get_neuronal_responses_ttls (obj, obj_animal_day_session):
    '''
    Indexes the neuronal responses according to the ttl values for each stimulus

    :param obj: ex: suite2p_obj
    :param day (str): yearmonthday, ex: '20231106'
    :return: void - modifies obj in place
    '''

    # load corrected ttl data into a dictionary with stim type (key) and array of ttls (value) (shape: n_stim_presentations x 2)
    dict_stim_ttls = get_ttl_dicts(obj, obj_animal_day_session)

    spikes = obj_animal_day_session['deconvolved_responses']
    fluorescence = obj_animal_day_session['responses']
    zscored_fluorescence = obj_animal_day_session['zscored_responses']

    # create a dictionary that holds the neural activity for each pair of TTLs (in chronological order)
    obj_animal_day_session['responses_ttls'] = {}
    obj_animal_day_session['spikes_ttls'] = {}
    obj_animal_day_session['zscored_responses_ttls'] = {}

    # if we are showing gratings
    if (obj_animal_day_session['n_SF'] and obj_animal_day_session['n_TF'] and obj_animal_day_session['n_theta']):

        obj_animal_day_session['spikes_ttls_whole'] = {}
        obj_animal_day_session['responses_ttls_whole'] = {}
        obj_animal_day_session['zscored_responses_ttls_whole'] = {}

        for stim in dict_stim_ttls.keys():

            obj_animal_day_session['spikes_ttls'][stim] = []
            obj_animal_day_session['spikes_ttls_whole'][stim] = []

            obj_animal_day_session['responses_ttls'][stim] = []
            obj_animal_day_session['responses_ttls_whole'][stim] = []

            obj_animal_day_session['zscored_responses_ttls'][stim] = []
            obj_animal_day_session['zscored_responses_ttls_whole'][stim] = []

            for [start_ttl, end_ttl] in dict_stim_ttls[stim]:
                if 'Wait' in stim:  # take entire wait period
                    obj_animal_day_session['spikes_ttls'][stim].append(spikes[:, int(start_ttl):int(end_ttl)])
                    obj_animal_day_session['responses_ttls'][stim].append(fluorescence[:, int(start_ttl):int(end_ttl)])
                    obj_animal_day_session['zscored_responses_ttls'][stim].append(zscored_fluorescence[:, int(start_ttl):int(end_ttl)])
                elif 'Grating' in stim:
                    # 1s static + 3s moving + 1s off
                    # only use 3 sec moving
                    # print(fluorescence[:, int(start_ttl + (obj.fps * 1)): int(start_ttl + (obj.fps * 4))].shape)
                    obj_animal_day_session['responses_ttls'][stim].append(fluorescence[:, int(start_ttl + (obj.fps * 1)): int(start_ttl + (obj.fps * 4))])
                    obj_animal_day_session['zscored_responses_ttls'][stim].append(zscored_fluorescence[:, int(start_ttl + (obj.fps * 1)): int(start_ttl + (obj.fps * 4))])
                    obj_animal_day_session['spikes_ttls'][stim].append(spikes[:, int(start_ttl + (obj.fps * 1)): int(start_ttl + (obj.fps * 4))])

                    # use 1 sec before stim + 1 sec static + 3 sec moving
                    #obj_animal_day_session['responses_ttls_whole'][stim].append(fluorescence[:, int(start_ttl - (obj.fps * 1)): int(end_ttl - (obj.fps * 1))])

                    # use 1 sec before stim + 1 sec static + 3 sec moving + 1 sec static
                    obj_animal_day_session['spikes_ttls_whole'][stim].append(spikes[:, int(start_ttl - (obj.fps * 1)): int(start_ttl + (obj.fps * 4))])
                    obj_animal_day_session['responses_ttls_whole'][stim].append(fluorescence[:, int(start_ttl - (obj.fps * 1)): int(start_ttl + (obj.fps * 4))])
                    obj_animal_day_session['zscored_responses_ttls_whole'][stim].append(zscored_fluorescence[:, int(start_ttl - (obj.fps * 1)): int(start_ttl + (obj.fps * 4))])
                elif 'Image' in stim:  # 0.5s on + 1.5s off + 1.3-1.7s jitter > only use 0.5s on
                    obj_animal_day_session['spikes_ttls'][stim].append(spikes[:, int(start_ttl):int(start_ttl + (obj.fps * 0.5))])
                    obj_animal_day_session['responses_ttls'][stim].append(fluorescence[:, int(start_ttl):int(start_ttl + (obj.fps * 0.5))])
                    obj_animal_day_session['zscored_responses_ttls'][stim].append(zscored_fluorescence[:, int(start_ttl):int(start_ttl + (obj.fps * 0.5))])

            # broadcasting list into array
            if 'Grating' in stim:  # in order to have an array of shape stimulus x cells x time, everything needs to be the same shape

                max_size = min([l.shape[-1] for l in obj_animal_day_session['spikes_ttls'][stim]])
                obj_animal_day_session['spikes_ttls'][stim] = np.array([n[:, :max_size] for n in obj_animal_day_session['spikes_ttls'][stim]])

                max_size = min([l.shape[-1] for l in obj_animal_day_session['responses_ttls'][stim]])
                obj_animal_day_session['responses_ttls'][stim] = np.array([n[:, :max_size] for n in obj_animal_day_session['responses_ttls'][stim]])

                max_size = min([l.shape[-1] for l in obj_animal_day_session['responses_ttls_whole'][stim]])
                obj_animal_day_session['responses_ttls_whole'][stim] = np.array([n[:, :max_size] for n in obj_animal_day_session['responses_ttls_whole'][stim]])

                max_size = min([l.shape[-1] for l in obj_animal_day_session['spikes_ttls_whole'][stim]])
                obj_animal_day_session['spikes_ttls_whole'][stim] = np.array([n[:, :max_size] for n in obj_animal_day_session['spikes_ttls_whole'][stim]])

                max_size = min([l.shape[-1] for l in obj_animal_day_session['zscored_responses_ttls'][stim]])
                obj_animal_day_session['zscored_responses_ttls'][stim] = np.array([n[:, :max_size] for n in obj_animal_day_session['zscored_responses_ttls'][stim]])

                max_size = min([l.shape[-1] for l in obj_animal_day_session['zscored_responses_ttls_whole'][stim]])
                obj_animal_day_session['zscored_responses_ttls_whole'][stim] = np.array([n[:, :max_size] for n in obj_animal_day_session['zscored_responses_ttls_whole'][stim]])

            obj_animal_day_session['responses_ttls'][stim] = np.array(obj_animal_day_session['responses_ttls'][stim])
            obj_animal_day_session['responses_ttls_whole'][stim] = np.array(obj_animal_day_session['responses_ttls_whole'][stim])
            obj_animal_day_session['spikes_ttls'][stim] = np.array(obj_animal_day_session['spikes_ttls'][stim])
            obj_animal_day_session['spikes_ttls_whole'][stim] = np.array(obj_animal_day_session['spikes_ttls_whole'][stim])
            obj_animal_day_session['zscored_responses_ttls'][stim] = np.array(obj_animal_day_session['zscored_responses_ttls'][stim])
            obj_animal_day_session['zscored_responses_ttls_whole'][stim] = np.array(obj_animal_day_session['zscored_responses_ttls_whole'][stim])

    # just showing on/off stims
    else:
        stim_ttls = np.array([np.array([dict_stim_ttls[key][0][0], dict_stim_ttls[key][-1][-1]]) for key in dict_stim_ttls.keys()])

        obj_animal_day_session['spikes_ttls'] = []
        obj_animal_day_session['responses_ttls'] = []
        obj_animal_day_session['zscored_responses_ttls'] = []

        for [start_ttl, end_ttl] in stim_ttls:
            obj_animal_day_session['spikes_ttls'].append(spikes[:, int(start_ttl):int(end_ttl)])
            obj_animal_day_session['responses_ttls'].append(fluorescence[:, int(start_ttl):int(end_ttl)])
            obj_animal_day_session['zscored_responses_ttls'].append(zscored_fluorescence[:, int(start_ttl):int(end_ttl)])

        # broadcasting list into array
        max_size = min([l.shape[-1] for l in obj_animal_day_session['spikes_ttls']])
        obj_animal_day_session['spikes_ttls'] = np.array([repeat[:,-max_size:] for repeat in obj_animal_day_session['spikes_ttls']])

        max_size = min([l.shape[-1] for l in obj_animal_day_session['responses_ttls']])
        obj_animal_day_session['responses_ttls'] = np.array([repeat[:,-max_size:] for repeat in obj_animal_day_session['responses_ttls']])

        max_size = min([l.shape[-1] for l in obj_animal_day_session['zscored_responses_ttls']])
        obj_animal_day_session['zscored_responses_ttls'] = np.array([repeat[:,-max_size:] for repeat in obj_animal_day_session['zscored_responses_ttls']])

        # # broadcasting list into array
        # if 'Grating' in stim:  # in order to have an array of shape stimulus x cells x time, everything needs to be the same shape
        #     max_size = min([l.shape[-1] for l in obj_animal_day_session['responses_ttls'][stim]])
        #     obj_animal_day_session['responses_ttls'][stim] = np.array(
        #         [n[:, :max_size] for n in obj_animal_day_session['responses_ttls'][stim]])
        #
        #     max_size = min([l.shape[-1] for l in obj_animal_day_session['responses_ttls_whole'][stim]])
        #     obj_animal_day_session['responses_ttls_whole'][stim] = np.array(
        #         [n[:, :max_size] for n in obj_animal_day_session['responses_ttls_whole'][stim]])

        #obj_animal_day_session['responses_ttls'][stim] = np.array(obj_animal_day_session['responses_ttls'][stim])


def responses_gratings(object_day_subfile):
    '''
    In order to calculate orientation tuning selectivity
    - average neural responses to the same orientation within the same block
    - order neural responses to gratings of increasing orientation

    'mean_ordered_grat' is an array of shape (n_blocks, n_orientations_shown), which stores orientation grating angles in increasing order
    'mean_ordered_grat_responses' is an array of shape (n_blocks, n_orientations_shown, n_cells, n_frames), which stores the neural response to each orientation in 'mean_ordered_grat'

    :param object: data object
    :param n_grating_repeats: For each grating stimulus 'block', each orientation is repeated a certain number of times.
    :return:
    '''

    dict_stim = object_day_subfile['dict_stim_names']
    f = object_day_subfile['responses_ttls']                # not zscored
    f_whole = object_day_subfile['responses_ttls_whole']    # not zscored

    # array of shape repeats (4) x n_samples (orientations x sf x tf)
    angles = object_day_subfile['thetas']
    sfs = object_day_subfile['SFs']

    responses = np.array([f[response] for response in f if 'GratingStim' in response])  # array of shape n_repeats (6) x n_orientations_in_repeat (8) x n_neurons x n_frames
    responses_whole = np.array([f_whole[response] for response in f_whole if 'GratingStim' in response])  # array of shape n_repeats (6) x n_orientations_in_repeat (8) x n_neurons x n_frames

    # SORTING ACCORDING TO ORIENTATION
    sort_i = np.argsort(angles, axis=1)  # indices on which to sort the 'angles' & 'responses' array
    #object_day_subfile['grating_sorting'] = sort_i

    # sorting angles / responses in order of increasing orientation
    sorted_angles = np.array([angles[i][sort_i[i]] for i in range(angles.shape[0])])  # ex: 0,0,30,30,60,60,...
    sorted_SFs = np.array([object_day_subfile['SFs'][i][sort_i[i]] for i in range(object_day_subfile['SFs'].shape[0])])  # ex: 0,0,30,30,60,60,...
    sorted_TFs = np.array([object_day_subfile['TFs'][i][sort_i[i]] for i in range(object_day_subfile['TFs'].shape[0])])  # ex: 0,0,30,30,60,60,...
    sorted_responses = np.array([responses[i][sort_i[i]] for i in range(angles.shape[0])])  # indexing neural responses so we can index/sort them similarly to to 'sorted_angles'
    sorted_responses_whole = np.array([responses_whole[i][sort_i[i]] for i in range(angles.shape[0])])  # indexing neural responses so we can index/sort them similarly to to 'sorted_angles'

    object_day_subfile['mean_ordered_grat'] = sorted_angles
    object_day_subfile['ordered_SFs'] = sorted_SFs
    object_day_subfile['ordered_TFs'] = sorted_TFs
    object_day_subfile['mean_ordered_grat_responses'] = sorted_responses
    object_day_subfile['mean_ordered_grat_responses_whole'] = sorted_responses_whole

    n_cells = object_day_subfile['n_cells']

    # reshape into shape n_repeats x n_theta x n_SF x n_TF (+ squeeze gets rid of singleton dimension) x n_cells x n_timepoints
    object_day_subfile['mean_ordered_grat'] = object_day_subfile['mean_ordered_grat'].reshape(
            object_day_subfile['nrepeats'], object_day_subfile['n_theta'], object_day_subfile['n_SF'],object_day_subfile['n_TF'])

    object_day_subfile['mean_ordered_grat_responses'] = object_day_subfile['mean_ordered_grat_responses'].reshape(
            object_day_subfile['nrepeats'], object_day_subfile['n_theta'], object_day_subfile['n_SF'],object_day_subfile['n_TF'], n_cells, -1)

    object_day_subfile['mean_ordered_grat_responses_whole'] = object_day_subfile['mean_ordered_grat_responses_whole'].reshape(
            object_day_subfile['nrepeats'], object_day_subfile['n_theta'], object_day_subfile['n_SF'],object_day_subfile['n_TF'], n_cells, -1)

    object_day_subfile['ordered_SFs'] = object_day_subfile['ordered_SFs'].reshape(
            object_day_subfile['nrepeats'], object_day_subfile['n_theta'], object_day_subfile['n_SF'],object_day_subfile['n_TF'])

    object_day_subfile['ordered_TFs'] = object_day_subfile['ordered_TFs'].reshape(
            object_day_subfile['nrepeats'], object_day_subfile['n_theta'], object_day_subfile['n_SF'],object_day_subfile['n_TF'])

    # # SORTING ACCORDING TO SPATIAL FREQUENCY
    # sort_i = np.argsort(sfs, axis=1)  # indices on which to sort the 'angles' & 'responses' array
    # #object_day_subfile['grating_sorting'] = sort_i
    #
    # # sorting angles / responses in order of increasing orientation
    # sorted_angles = np.array([angles[i][sort_i[i]] for i in range(angles.shape[0])])  # ex: 0,0,30,30,60,60,...
    # sorted_SFs = np.array([object_day_subfile['SFs'][i][sort_i[i]] for i in range(object_day_subfile['SFs'].shape[0])])  # ex: 0,0,30,30,60,60,...
    # sorted_TFs = np.array([object_day_subfile['TFs'][i][sort_i[i]] for i in range(object_day_subfile['TFs'].shape[0])])  # ex: 0,0,30,30,60,60,...
    # sorted_responses = np.array([responses[i][sort_i[i]] for i in range(angles.shape[0])])  # indexing neural responses so we can index/sort them similarly to to 'sorted_angles'
    # sorted_responses_whole = np.array([responses_whole[i][sort_i[i]] for i in range(angles.shape[0])])  # indexing neural responses so we can index/sort them similarly to to 'sorted_angles'
    #
    # print(sorted_angles)
    # print(sorted_SFs)

def normalize_arr(x):
    '''
    Scale/normalize values in array between 0 & 1, along the cell/row axis
    :return:
    '''
    return np.array([(row - row.min()) / (row.max() - row.min()) for row in x])


def circular_diff(angle1, angle2):
    '''
    returns the shortest possible angular difference between angle 1 and angle2 in the range [-180, 180]

    using a circular distance formula  between theta1 and theta2 = (theta1 - theta2 + 180) % 360 - 180
    '''
    return (angle1 - angle2 + 180) % 360 - 180

def get_complex_num(obj_day_session, avg_over_sf = True):
    '''
    calculate tuning curves > sum [r_theta * e ^ (i2theta)] / sum [r_theta]
    theta = direction of the kth condition (array of all the directions)
    r_theta = neuronal responses to the directions

    average the responses
    '''

    # there is only one SF and TF
    if (obj_day_session['n_SF'] ==1) and (obj_day_session['n_TF'] ==1):

        # shape: (n_orientations); ex: [  0.  45.  90. 135. 180. 225. 270. 315.]
        obj_day_session['theta'] = np.squeeze(obj_day_session['mean_ordered_grat'].mean(axis = 0))
        theta_rad = np.deg2rad(obj_day_session['theta'])

        # avg response to the stimulus over time period > shape: (repeats, n_orientations, n_cells)
        obj_day_session['tuning_curves'] = np.squeeze(obj_day_session['mean_ordered_grat_responses']).mean(axis = -1)

        # mean response to the stimulus across repeats > shape: (n_cells, n_orientations)
        responses_theta = obj_day_session['tuning_curves'].mean(axis = 0).T

    elif (obj_day_session['n_theta'] > 1) and (obj_day_session['n_SF'] > 1): # there is more than one orientaiton or SF

        if not avg_over_sf:
            # shape: (n_orientations); ex: [  0.  45.  90. 135. 180. 225. 270. 315.]
            obj_day_session['theta'] = np.squeeze(obj_day_session['mean_ordered_grat'].mean(axis=0)).mean(axis = 1)
            theta_rad = np.deg2rad(obj_day_session['theta'])

            # avg response to the stimulus over time period & SFs > shape: repeats x n_orientations x n_cells
            tuning_curve = np.squeeze(obj_day_session['mean_ordered_grat_responses']).mean(axis = 2).mean(axis = -1)

            # average response across all orientations for each cell. (shape is n_repeats, 1, n_cells)
            baseline = np.nanmean(tuning_curve, axis=1, keepdims=True)

            # subtract baseline response for each cell from its responses to all orientations
            tuning_curve -= baseline

            # ensures that all responses are non-negative after baseline subtraction (any values < 0 are set to zero)
            obj_day_session['tuning_curves'] = np.maximum(tuning_curve, 0)

            # mean response to the stimulus across repeats > shape: (n_cells, n_orientations)
            responses_theta = obj_day_session['tuning_curves'].mean(axis=0).T

        ########## also want to build tuning curve by taking responses to each cell's preferred SF ###############

        # pref_SF = obj_day_session['pref_SF'] # shape n_cell
        #
        # # corresponding SFs for the responses_ttls_gratings array; shape n_repeat, (n_orientation x n_sf)
        # SFs_presented = obj_day_session['SFs']
        #
        # # mean stimulus response across time; shape n_repeat, (n_orientation x n_sf), n_cells, x n_timepoints
        # responses_ttls_gratings = np.array([obj_day_session['responses_ttls'][key] for key in obj_day_session['responses_ttls'].keys() if 'Wait' not in key])
        #
        # # indices pointing to responses to the best SF; shape n_cells, (n_repeats x n_orientations), 2
        # best_SF_idx = np.array(
        #     [np.argwhere(SFs_presented == np.round(pref_SF[i_cell], 3)) for i_cell in range(n_cells)])
        #
        # # for the cell's preferred SF, this is the mean response to each repeat and orientation (shape n_cells x 48 x n_timepoints)
        # pref_response = np.array([np.array([responses_ttls_gratings[i, j, i_cell] for i, j in best_SF_idx[i_cell]]) for i_cell in range(n_cells)])
        #
        # # the orientation shown at each pref SF (shape n_cells x 48)
        # pref_response_theta = np.array([np.array([obj_day_session['thetas'][i, j] for i, j in best_SF_idx[i_cell]]) for i_cell in range(n_cells)])
        #
        # # indices to sort pref_response according to increasing orientation, so we can average across repeats (shape n_cells, n-repeats x n_orientations)
        # indices_theta = np.argsort(pref_response_theta, axis = 1)
        #
        # # array of the orientations in ascending order (shape n_theta), sorted according to indices_theta
        # orientations = np.sort(pref_response_theta).reshape(n_cells, obj_day_session['n_theta'], responses_ttls_gratings.shape[0]).mean(axis = -1).mean(axis = 0)
        #
        # # mean response for each orientation (shape n_repeats, n_theta, n_cells) (mean across timepoints)
        # obj_day_session['tuning_curves'] = np.array(
        #                                             [pref_response[i_cell, indices_theta[i_cell]].reshape(
        #                                                 obj_day_session['n_theta'], responses_ttls_gratings.shape[0],-1)
        #                                                 for i_cell in range(n_cells)]).mean(axis = -1).T
        # # obj_day_session['tuning_curves'] = np.array(
        # #                                             [pref_response[i_cell, indices_theta[i_cell]].reshape(
        # #                                                 obj_day_session['n_theta'], responses_ttls_gratings.shape[0], -1)
        # #                                                 for i_cell in range(n_cells)]).mean(axis=-1).T
    else:
        return

    # shape (n_cells);  NB: np.exp outputs shape n_orientations
    # in python, j is the imaginary component
    obj_day_session['complex'] = (responses_theta * np.exp(1j * 2 * theta_rad)).sum(axis=-1) / (responses_theta.sum(axis=-1))
    # note that e^(i2theta) doubles the angles because orientations are periodic over 180deg, not 360deg.
    # this ensures that responses to 0deg and 180deg are treated as equivalent (represent same orientation)
    # consequently, the resultant vector encodes the doubled angle of the preferred orientation.
    # note that to calculate direction selectivity you wouldnt have the 2 factor

    # shape (n_cells) > magnitude (absolute value) of the complex number > represents the amplitude of the response (how strongly neuron responds) = OSI (should be between 0 & 1)
    obj_day_session['OSI'] = np.abs(obj_day_session['complex'])
    obj_day_session['OSI'] = np.nan_to_num(obj_day_session['OSI'], nan=0) #if the OSI is nan, put it to 0

    # the argument (angle / preferred angle) of the complex number encodes the preferred orientation (the orientation to which the neuron responds most strongly).
    obj_day_session['pref_orientation'] = np.rad2deg(np.angle(obj_day_session['complex']))/2  # preferred angle
    # divide by 2 because of the i*theta*2 in the original equation

def find_pref_SF (obj_day_session):

    n_cells = obj_day_session['responses'].shape[0]

    # corresponding orientations for the responses_ttls_gratings array; shape n_repeat, (n_orientation x n_sf)
    orientations = obj_day_session['thetas']

    if (obj_day_session['n_theta'] > 1) and (obj_day_session['n_SF'] > 1):  # there is more than one orientaiton or SF

        # the orientation that each neuron best responded to > shape n_neurons (numbered from 0 to 7); shape n_cell
        best_orientation = obj_day_session['orientations'][np.array(
            [np.argmin(np.abs(np.array([circular_diff(cell_pref, angle) for angle in obj_day_session['theta']]))) for cell_pref in
             obj_day_session['pref_orientation']])]

        # mean stimulus response across time; shape n_repeat, (n_orientation x n_sf), n_cells
        responses_ttls_gratings = np.array([obj_day_session['responses_ttls'][key] for key in obj_day_session['responses_ttls'].keys() if 'Wait' not in key]).mean(axis = -1)

        # indices pointing to responses to the best orientation; shape n_cells, (n_repeats x n_sfs), 2
        best_orientation_idx = np.array([np.argwhere(orientations == best_orientation[i_cell]) for i_cell in range(n_cells)])

        # for the cell's preferred orientation, this is the mean response to each repeat and SF exposure (shape n_cells x 30)
        pref_response = np.array([np.array([responses_ttls_gratings[i, j, i_cell] for i, j in best_orientation_idx[i_cell]]) for i_cell in range(n_cells)])

        # the spatial frequency shown at each pref orientation (shape n_cells x 30)
        pref_response_sf = np.array([np.array([obj_day_session['SFs'][i, j] for i, j in best_orientation_idx[i_cell]]) for i_cell in range(n_cells)])

        # sort pref_response according to increasing SF, so we can average across repeats
        indices_sf = np.argsort(pref_response_sf, axis = 1)

        # array of the SFs in ascending order
        sfs = np.sort(pref_response_sf).reshape(n_cells,obj_day_session['n_SF'],-1).mean(axis = -1).mean(axis = 0)

        # mean response across repeats for each SF (shape n_cells, n_SF)
        pref_response_sorted_sf = np.array([pref_response[i_cell, indices_sf[i_cell]].reshape(obj_day_session['n_SF'],-1) for i_cell in range(n_cells)]).mean(axis = -1)

        # get the index of the SF that gave the maximum response
        max_pref_response_sorted_sf = np.argmax(pref_response_sorted_sf, axis = 1) # shape n_cells

        # preferred spatial frequency for each cell (shape n_cells)
        obj_day_session['pref_SF'] = sfs[max_pref_response_sorted_sf]

        # responses_ttls_gratings has shape n_repeats, n_orientations x n_spatialfrequencies x n_cells > average response
        # best_orientation_idx has shape n_repeats, n_spatialfrequencies x n_cells x 2 > index for the average response at a specific orientation.

def regress_out_behav(rawY, predX):
    '''
    Project out neural activity predicted from behaviour (from facemap) from the true neural activity, using Linear Regression
    Remove the part of the neural activity that can be explained by the predicted spontaneous behaviour
    :param rawY: raw neural activity, numpy matrix of shape n_neurons x n_frames
    :param predX: predicted neural activity from behavioural variables, numpy matrix of shape n_neurons x n_frames
    :return:
    '''
    n_neurons = rawY.shape[0]
    residuals = np.zeros_like(rawY)

    # for each neuron, perform a linear regression: regress its true activity on its predicted behavioural activity
    for i in range(n_neurons):
        y = rawY[i, :]
        x = predX[i, :]

        # perform linear regression: y = a * x + b > np.polyfit finds the best fit line / coefficients that explain the spontaneous behaviour
        a, b = np.polyfit(x, y, 1)

        # Calculate the predicted activity based on the regression
        y_pred = a * x + b

        # Calculate the residual (stimulus-related activity, or neural activity not explained by the spontaneous behaviour)
        # subtract predicted neural activity (based on regression coefficients) from true neural activity
        residuals[i, :] = y - y_pred

    return residuals

def save_fig (object, folder_title, figure_title):
    folder_path = os.path.join(object.save_path, folder_title)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(os.path.join(folder_path, f'{figure_title}.png'))
    plt.savefig(os.path.join(folder_path, f'{figure_title}.svg'))

def fit_rastermap (response_array):
    '''
    Visualization framework for neural data > sorts neual responses along a one-dimensional manifold
    https://github.com/MouseLand/rastermap
    https://www.biorxiv.org/content/10.1101/2023.07.25.550571v2
    :param response_array: array of neurons x time
    :return:
    '''
    # fit rastermap (https://github.com/MouseLand/rastermap)
    # https://www.biorxiv.org/content/10.1101/2023.07.25.550571v2
    model = Rastermap(n_PCs=200, n_clusters=100, locality=0.75, time_lag_window=5).fit(response_array, compute_X_embedding=True)
    y = model.embedding  # neurons x 1
    isort = model.isort
    X_embedding = model.X_embedding             # visualize binning over neurons

    return X_embedding

def parse_grating_string(s):
    '''
    translate grating string to extract orientation, TF, and SF
    :param s:
    :return:
    '''
    match = re.search(r'grating_(\d+)_SF(\d+\.\d+)_TF(\d+\.\d+)', s)
    if match:
        # return {
        #     'deg': int(match.group(1)),
        #     'SF': float(match.group(2)),
        #     'TF': float(match.group(3))
        # }
        deg = int(match.group(1))
        sf = float(f"{float(match.group(2)):.3f}")  # Format to 2 decimal places
        tf = float(f"{float(match.group(3)):.3f}")  # Format to 2 decimal places
        sf = 0.005 if sf == 0.01 else sf        # for some reasont the 0.005 SF string format is wrong
        return [deg, sf, tf]
    return None

def parse_grating_array (arr):
    np.set_printoptions(suppress=True, precision=4)
    return np.array([parse_grating_string(s) for s in arr])

def on_off_scores (object, animal, day, session):
    # dont' use z socred responses becuase responses need to be positive
    arr = object.dat[animal][day][session]['spikes_ttls']
    # average response of each cell
    grey1 = arr[:, :, :object.fps * 4].mean(axis=(0, -1))
    black1 = arr[:, :, object.fps * 4:object.fps * 8].mean(axis=(0, -1))
    white =  arr[:, :, object.fps * 8:object.fps * 12].mean(axis=(0, -1))
    black2 = arr[:, :,object.fps * 12:object.fps * 16].mean(axis=(0, -1))
    grey2 = arr[:, :, object.fps * 16:].mean(axis=(0, -1))
    on_score = np.array((white / black1, grey2 / black2)).mean(axis=0)
    off_score = np.array((black2 / white, black1 / grey1)).mean(axis=0)

    on_off_index = (on_score - off_score) / (on_score + off_score)

    return on_off_index

def find_threshold_crossing(signal, threshold, time_axis):
    """Finds the first time point where signal crosses the threshold."""
    idx = np.where(signal >= threshold)[0]
    return time_axis[idx[0]] if len(idx) > 0 else 0

def response_speed (object, animal, day, session):
    # dont' use z score responses becuase responses need to be positive
    # grab portion of the response during the white FFF, mean across repeats
    # shape (n_cells, n_timepoints)
    arr = object.dat[animal][day][session]['zscored_matrix_baseline'][:,:, object.fps*8:object.fps*12].mean(axis = 0)

    R_max = arr.max(axis = 1)                               # max response for each cell
    threshold_10, threshold_90 = 0.1 * R_max, 0.9 * R_max   # 10% and 90% thresholds

    time_axis = np.arange(arr.shape[1])  # time axis  (n_timepoints,)

    # Find crossing times for each cell
    t_10 = np.array([find_threshold_crossing(arr[i, :], threshold_10[i], time_axis) for i in range(arr.shape[0])])
    t_90 = np.array([find_threshold_crossing(arr[i, :], threshold_90[i], time_axis) for i in range(arr.shape[0])])

    # Compute slope (change in response over time)
    response_speed = (threshold_90 - threshold_10) / (t_90 - t_10)
    response_speed[np.isinf(response_speed)] = -1

    return response_speed

def response_amplitude_FFF (object, animal, day, session):
    # dont' use z score responses becuase responses need to be positive
    # grab portion of the response during the white FFF, mean across repeats
    arr = object.dat[animal][day][session]['zscored_matrix_baseline'][:,:, object.fps*8:object.fps*12].mean(axis = 0)

    # this is the amplitude of the response (max peak - min peak)
    amplitude = arr.max(axis = 1)

    return amplitude

def save_fig (object, folder_title, figure_title):
    folder_path = os.path.join(object.save_path, folder_title)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(os.path.join(folder_path, f'{figure_title}.png'))
    plt.savefig(os.path.join(folder_path, f'{figure_title}.svg'))

from datetime import datetime, timedelta
def calculate_animal_age(dob, imaging_date):
    '''
    Returns the age of the animal born on DOB, on imaging_date (in post-natal days, and weeks)
    :param dob: format 'YYYY/MM/DD'
    :param imaging_date: 'YYYY/MM/DD'
    :return:
    '''
    # Convert input strings to date objects
    dob = datetime.strptime(dob, "%Y/%m/%d")
    imaging_date = datetime.strptime(imaging_date, "%Y/%m/%d")

    # Calculate the difference between the dates in days
    age_in_days = (imaging_date - dob).days

    # Calculate weeks and remaining days
    weeks = age_in_days // 7
    days = age_in_days % 7

    return f"P {age_in_days} ({weeks} weeks + {days} days)"

def calculate_day_animal_age(dob, target_postnatal_day):
    '''
    Returns day that the animal born on DOB will be of age target_postnatal_day
    :param dob: format 'YYYY/MM/DD'
    :param target_postnatal_day: INT
    :return:
    '''
    # Convert input strings to date objects
    dob = datetime.strptime(dob, "%Y/%m/%d")
    formatted_dob = dob.strftime("%Y/%m/%d")

    dob_plus_target = dob + timedelta(days=target_postnatal_day)
    formatted_date = dob_plus_target.strftime("%Y/%m/%d")

    print( f"Animal born on {formatted_dob} will be P{target_postnatal_day} on {formatted_date})")

def sf_tuning(obj, animal, day, sub_file):
    '''
    - Generates & stores amplitude / tuning curves for the different SFs
    - Generates tuning curve plots by calling 'sf_tuning_curves' function
    '''

    # ordered_sfs : shape n_repeats x n_orientations x n_sfs x 1 > represents the SFs shown for each presentation (presentations sorted according to orientation)
    ordered_SFs = obj.dat[animal][day][sub_file]['ordered_SFs']
    sort_i = np.argsort(ordered_SFs,axis=2)  # indices to sort 'ordered_sfs' in ascending order for each repeat & orientation block

    # sort along the SF axis, and then average across repeats
    sorted_sfs = np.take_along_axis(ordered_SFs, sort_i, axis=2).mean(axis=0)

    # sort the responses along the SF axis, and then average across repeats
    expanded_sort_i = np.expand_dims(sort_i, axis=(4, 5))
    sorted_responses = np.take_along_axis(obj.dat[animal][day][sub_file]['mean_ordered_grat_responses'], expanded_sort_i, axis=2).mean(axis=0)

    # CALCULATE RESPONSE AMPLITUDE FOR EACH SF
    sorted_responses_avg = np.take_along_axis(obj.dat[animal][day][sub_file]['mean_ordered_grat_responses'], expanded_sort_i, axis=2)
    obj.dat[animal][day][sub_file]['SF_tuning'] = np.squeeze(sorted_responses_avg.max(axis=-1)).mean(axis = 1)  # shape SFs x cells > represents the response amplitude for each SF (& each repeat)

    sf_tuning_curves (obj, animal, day, sub_file, obj.dat[animal][day][sub_file]['SF_tuning'], sorted_sfs)


def sf_tf_tuning(obj, animal, day, sub_file):
    '''
    - Generates & stores amplitude / tuning curves for the different SFs x TFs
    - Generates tuning curve plots by calling 'sf_tf_tuning_curves' function
    '''

    SFs, TFs = obj.dat[animal][day][sub_file]['SFs'], obj.dat[animal][day][sub_file]['TFs']
    unique_SFs, unique_TFs = np.unique(SFs), np.unique(TFs)

    responses = obj.dat[animal][day][sub_file]['responses_ttls']
    responses_ttls_gratings = np.array([responses[key] for key in responses.keys() if 'Wait' not in key]) # mean responses across time

    n_cells = responses_ttls_gratings.shape[-2]
    average_max_responses = np.zeros((len(unique_SFs), len(unique_TFs), n_cells))

    for i, sf in enumerate(unique_SFs):
        for j, tf in enumerate(unique_TFs):

            # search for 2d indices where there is a 'true' (which signifies a SF/TF match)
            mask_indices = np.argwhere((SFs == sf) & (TFs == tf))

            # max response amplitude for each sf x tf pair (shape n_cells) (averaged over repeats)
            average_max_responses [i, j] = np.array([responses_ttls_gratings[k, l].max(axis=-1) for (k, l) in mask_indices]).mean(axis = 0)

    obj.dat[animal][day][sub_file]['SF_TF_tuning'] = average_max_responses

    sf_tf_tuning_curves(obj, animal, day, sub_file, unique_SFs, unique_TFs)


def build_parameter_matrix(object, animal, day, subfile, response_window = 'whole', zscore = True):
    '''
    :param object:  data_object
    :param animal:
    :param day:
    :param subfile:
    :param response_window: either 'whole' (1s wait + 1s static + 3s moving + 1s wait) or 'moving' (3 s moving)
    :param zscore: True or False: whether or not we want z scored responses in our calculations
    :return:
    '''

    # 1 spatial frequency and 8 orientations
    if (object.dat[animal][day][subfile]['n_SF'] == 1) and (object.dat[animal][day][subfile]['n_TF'] == 1):

        # each of shape (n_repeats, x nORI) > parameters for stimuli shown for each repeat
        thetas_stim = object.dat[animal][day][subfile]['thetas']

        thetas = np.unique(thetas_stim)

        # dictionary - 1 key per repeat. each repeat contains array of shape (nSF x nOri, timepoints)
        if response_window == 'whole':      # 1s wait + 1s static + 3s moving + 1s wait
            if zscore:
                responses_ttls = object.dat[animal][day][subfile]['zscored_responses_ttls_whole']
            else:
                responses_ttls = object.dat[animal][day][subfile]['responses_ttls_whole']
        elif response_window == 'moving':   # 3s moving
            if zscore:
                responses_ttls = object.dat[animal][day][subfile]['zscored_responses_ttls']
            else:
                responses_ttls = object.dat[animal][day][subfile]['responses_ttls']

        # array shape (n_repeats, nORI, n_cells, timepoints)
        min_t = min(responses_ttls[key].shape[-1] for key in responses_ttls if 'Grating' in key)
        gratings_responses = np.array([responses_ttls[key][..., :min_t] for key in responses_ttls.keys() if 'Grating' in key])

        n_repeats = gratings_responses.shape[0]
        n_orientation = len(thetas)
        cells = gratings_responses.shape[2]
        timepoints = gratings_responses.shape[-1]

        # the reordered array> contains response matrix for each repeat, with sorted orientation and SF
        responses_ordered = np.zeros((n_repeats, n_orientation, cells, timepoints))

        for i_repeat in range(n_repeats):
            for i_theta, theta in enumerate(thetas):
                # find where this sf/orientation combination exists in thetas_repeat
                idx = np.where(thetas_stim[i_repeat] == theta)[0][0]

                # store response at this index
                responses_ordered[i_repeat, i_theta] = gratings_responses[i_repeat, idx]

        # store in object
        object.dat[animal][day][subfile][f'param_matrix_{response_window}{zscore*"_zscore"}'] = responses_ordered

        return responses_ordered, thetas

    # SFxO
    elif object.dat[animal][day][subfile]['n_TF'] == 1:

        # each of shape (n_repeats, nSF x nORI) > parameters for stimuli shown for each repeat
        sfs_stim = object.dat[animal][day][subfile]['SFs']
        tfs_stim = object.dat[animal][day][subfile]['TFs']
        thetas_stim = object.dat[animal][day][subfile]['thetas']

        sfs, tfs, thetas = np.unique(sfs_stim), np.unique(tfs_stim), np.unique(thetas_stim)

        # dictionary - 1 key per repeat. each repeat contains array of shape (nSF x nOri, timepoints)
        if response_window == 'whole':      # 1s wait + 1s static + 3s moving + 1s wait
            if zscore:
                responses_ttls = object.dat[animal][day][subfile]['zscored_responses_ttls_whole']
            else:
                responses_ttls = object.dat[animal][day][subfile]['responses_ttls_whole']
        elif response_window == 'moving':   # 3s moving
            if zscore:
                responses_ttls = object.dat[animal][day][subfile]['zscored_responses_ttls']
            else:
                responses_ttls = object.dat[animal][day][subfile]['responses_ttls']

        # array shape (n_repeats, nSF x nORI, n_cells, timepoints)
        # gratings_responses = np.array([responses_ttls[key] for key in responses_ttls.keys() if 'Grating' in key])
        min_t = min(responses_ttls[key].shape[-1] for key in responses_ttls if 'Grating' in key)
        gratings_responses = np.array([responses_ttls[key][..., :min_t] for key in responses_ttls.keys() if 'Grating' in key])


        n_repeats = gratings_responses.shape[0]
        n_orientation = len(thetas)
        n_sf = len(sfs)
        cells = gratings_responses.shape[2]
        timepoints = gratings_responses.shape[-1]

        # the reordered array> contains response matrix for each repeat, with sorted orientation and SF
        responses_ordered = np.zeros((n_repeats, n_orientation, n_sf, cells, timepoints))

        for i_repeat in range(n_repeats):
            for i_theta, theta in enumerate(thetas):
                for i_sf, sf in enumerate(sfs):
                    # find where this sf/orientation combination exists in sfs_repeat and thetas_repeat
                    idx = np.where((sfs_stim[i_repeat] == sf) & (thetas_stim[i_repeat] == theta))[0][0]

                    # store response at this index
                    responses_ordered[i_repeat, i_theta, i_sf] = gratings_responses[i_repeat, idx]

        # store in object
        object.dat[animal][day][subfile][f'param_matrix_{response_window}{zscore*"_zscore"}'] = responses_ordered

        return responses_ordered, thetas, sfs, tfs

    # SFxTF (only 1 orientation)
    elif object.dat[animal][day][subfile]['n_theta'] == 1:

        # each of shape (n_repeats, nSF x nORI) > parameters for stimuli shown for each repeat
        sfs_stim = object.dat[animal][day][subfile]['SFs']
        tfs_stim = object.dat[animal][day][subfile]['TFs']
        thetas_stim = object.dat[animal][day][subfile]['thetas']

        sfs, tfs, thetas = np.unique(sfs_stim), np.unique(tfs_stim), np.unique(thetas_stim)

        # dictionary - 1 key per repeat. each repeat contains array of shape (nSF x nOri, timepoints)
        if response_window == 'whole':      # 1s wait + 1s static + 3s moving + 1s wait
            if zscore:
                responses_ttls = object.dat[animal][day][subfile]['zscored_responses_ttls_whole']
            else:
                responses_ttls = object.dat[animal][day][subfile]['responses_ttls_whole']
        elif response_window == 'moving':   # 3s moving
            if zscore:
                responses_ttls = object.dat[animal][day][subfile]['zscored_responses_ttls']
            else:
                responses_ttls = object.dat[animal][day][subfile]['responses_ttls']

        # array shape (n_repeats, nSF x nORI, n_cells, timepoints)
        gratings_responses = np.array([responses_ttls[key] for key in responses_ttls.keys() if 'Grating' in key])

        n_repeats, cells, timepoints = gratings_responses.shape[0], gratings_responses.shape[2], gratings_responses.shape[-1]
        n_sf, n_tf = len(sfs), len(tfs)

        # the reordered array> contains response matrix for each repeat, with sorted orientation and SF
        responses_ordered = np.zeros((n_repeats, n_sf, n_tf, cells, timepoints))

        for i_repeat in range(n_repeats):
            for i_sf, sf in enumerate(sfs):
                for i_tf, tf in enumerate(tfs):
                    # find where this sf/orientation combination exists in sfs_repeat and thetas_repeat
                    idx = np.where((sfs_stim[i_repeat] == sf) & (tfs_stim[i_repeat] == tf))[0][0]

                    # store response at this index
                    responses_ordered[i_repeat, i_tf, i_sf] = gratings_responses[i_repeat, idx]

        # store in object
        object.dat[animal][day][subfile][f'param_matrix_{response_window}{zscore*"_zscore"}'] = responses_ordered

        return responses_ordered, thetas, sfs, tfs,

def pref_sf (object, parameter_matrix):
    '''
    :param object:
    :param animal:
    :param day:
    :param subfile:
    :return: array of shape n_cells, representing the index of the preferred SF for each cell (max response across the board)
    '''

    # average across repeats, then grab the response period (static+ 2 sec moving)
    # shape (n_orientations, n_sf, n_cells, n_timepoints)
    repeats_avg = parameter_matrix.mean(axis=0)[:, :, :, object.fps*2:object.fps*4]

    # sum across orientation and time axis (to collapse high dimensional data) (result is shape (n_sf, n_cells))
    # then take the argmax across the SF axis
    preferred_sf_idx = np.argmax(np.sum(repeats_avg, axis=(0, -1)), axis=0)
    #
    # cell_indices = np.arange(parameter_matrix.shape[3])
    #
    # # average across repeats and SFs > average response over the static + first 2 seconds of moving period
    # # index of max orientation
    # max_orientation_index = np.argmax(parameter_matrix.mean(axis = (0,2))[:, :, object.fps:object.fps*4].mean(axis = -1), axis = 0)
    #
    # # response to the best orientation > shape (cells, repeats, sfs, time)
    # max_orientation_response = parameter_matrix[:, max_orientation_index, :, cell_indices, :]
    #
    # # mean across repeats axis > average of first 2 seconds of static + first 2 sec of moving period > shape (n_cells, n_sf)
    # response_across_sfs = max_orientation_response.mean(axis = 1)[:,:,object.fps:object.fps*4].mean(axis = -1)
    #
    # # shape (n_cells)
    # preferred_sf_idx = np.argmax(response_across_sfs, axis = 1)

    return preferred_sf_idx

def pref_theta (object, parameter_matrix):
    '''
    :return:
    - preferred_theta_idx: preferred theta for each cell (index of the preferred SF for each cell (max response across the board)); shape (n_cells)

    '''
    # average across repeats, then grab the response period (static+ 2 sec moving)
    # shape (n_orientations, n_sf, n_cells, n_timepoints) OR (n_orientations, n_cells, n_timepoints) (if o * repeat)
    if parameter_matrix.ndim == 5:
        repeats_avg = parameter_matrix.mean(axis=0)[:, :, :, object.fps*2:object.fps*4]
    elif parameter_matrix.ndim == 4:
        repeats_avg = parameter_matrix.mean(axis=0)[:, :, object.fps*2:object.fps*4]

    # sum across sf and time axis (to collapse high dimensional data) (result is shape (n_sf, n_cells))
    # then take the argmax across the theta axis
    # shape ((n_cells)
    preferred_theta_idx = np.argmax(np.sum(repeats_avg, axis=(1, -1)), axis=0)

    return preferred_theta_idx

def sf_tuning_curve (object, parameter_matrix):
    '''
    :return:
    - response_across_sfs: SF tuning curve at the best orientation; shape (n_cells, n_repeats, n_sf)
    - pref_theta_idx: preferred theta for each cell; shape (n_cells)
    '''

    pref_theta_idx = pref_theta(object, parameter_matrix)

    cell_indices = np.arange(parameter_matrix.shape[3])

    # response to the best orientation > shape (cells, repeats, sfs, time)
    pref_theta_response = parameter_matrix[:, pref_theta_idx, :, cell_indices, :]

    # mean response across time axis > average of first 1 seconds of static + first 2 sec of moving period > shape (n_cells, n_repeats, n_sf)
    response_across_sfs = pref_theta_response[:,:,:,object.fps*2:object.fps*4].mean(axis = -1)

    return response_across_sfs, pref_theta_idx


def grating_response_amplitude (object, response_array):
    '''
    - Across all repeats, take a time window around the peak response
    - Return average (over time) of each window
    :param object:
    :param response_array: ((n_repeats, n_orientation, n_SF, cells, n_timepoints))
    :return: return response_array for a time window specified by 'method', averaged across timepoints
    '''

    #size of window (on each side) (i.e., window frames on either side)
    half_window = object.fps / 4 # quarter of a second (5 frames on either side if 20fps)
    indices = np.arange(-half_window, half_window)

    # grab the entire 'moving period' (shape (n_ori, n_sf, n_cells, n_timepoints))
    moving = response_array[..., 2 * (object.fps):5 * (object.fps):].mean(axis = 0)

    n_ori, n_sf, n_cells, timepoints_moving = moving.shape

    # grab the max index during the moving period (shape (n_ori, n_sf, n_cells, 1))
    max_index = np.argmax(moving, axis = -1)[..., np.newaxis]

    # shape: (_ori, n_sf, n_cells, window_size)
    window_indices = max_index + indices

    # grab the response values at the window indices, excluding anything < 0 (preceeding the start of moving window) or > window length (after end of moving window)
    # then take the mean of the response within the window
    # shape n_ori, n_sf, n_cells
    return np.array([[[np.array([moving[i_ori, i_sf, i_cell, int(idx)]
                                                   for idx in window_indices[i_ori, i_sf, i_cell] if 0 <= idx < moving.shape[-1]]).mean()
                                                   for i_cell in range(n_cells)]
                                                   for i_sf in range(n_sf)]
                                                   for i_ori in range(n_ori)])


def deltaf_over_f(obj, param_matrix):
    '''
    calculate delta f / f (for diff stims, either O, SFxO, SFxTF, or FFF)

    :param param_matrix: shape (n_repeats, n_ori, n_cells, n_timepoints) OR shape (n_repeats, n_ori, n_SF, n_cells, n_timepoints) or (n_repeats, n_SF, n_TF, n_cells, n_timepoints)  OR shape (n_repeats, n_cells, n_timepoints) (if FFF)
    :return: dF/F matrix shape (n_repeats, n_ori, n_cells, n_timepoints) OR shape (n_repeats, n_ori, n_SF, n_cells, n_timepoints) or (n_repeats, n_SF, n_TF, n_cells, n_timepoints)  OR shape (n_repeats, n_cells, n_timepoints) (if FFF)
    '''

    # take the mean over the first second of the resposne, across all dimensions except cells

    # baseline_F is of shape n_cells
    if param_matrix.ndim == 4: # param_matrix: shape (n_repeats, n_ori, n_cells, n_timepoints)
        baseline_F = param_matrix[..., :obj.fps].mean(axis= -1, keepdims=True)
    elif param_matrix.ndim == 5: # either shape (n_repeats, n_ori, n_SF, n_cells, n_timepoints) or (n_repeats, n_SF, n_TF, n_cells, n_timepoints)
        baseline_F = param_matrix[..., :obj.fps].mean(axis=(2,-1), keepdims=True)

    # take first 4sec of response (baseline)
    elif param_matrix.ndim == 3:  # FFF, response matrix: shape (n_repeats, n_cells, n_timepoints)
        baseline_F = param_matrix[..., :obj.fps* 4].mean(axis=-1, keepdims=True)

    return (param_matrix - baseline_F) / baseline_F


def zscore_baseline(obj, param_matrix):
    '''
    z-score the parameter matrix according to the baseline period for each trial (for diff stims, either O, SFxO, SFxTF, or FFF)

    :param param_matrix: shape (n_repeats, n_ori, n_cells, n_timepoints) OR shape (n_repeats, n_ori, n_SF, n_cells, n_timepoints) or (n_repeats, n_SF, n_TF, n_cells, n_timepoints) OR shape (n_repeats, n_cells, n_timepoints) (if FFF)
    :return: z scored matrix shape (n_repeats, n_ori, n_cells, n_timepoints) OR shape (n_repeats, n_ori, n_SF, n_cells, n_timepoints) or (n_repeats, n_SF, n_TF, n_cells, n_timepoints)  OR shape (n_repeats, n_cells, n_timepoints) (if FFF)
    '''

    # Z SCORING MATRIX
    # take first second of response (baseline) > mean/std across time/orientation/repeat axis
    # baseline_mean and baseline_std are of shape (n_repeats,n_ori, n_cells)
    if param_matrix.ndim == 4:  # param_matrix: shape (n_repeats, n_ori, n_cells, n_timepoints)
        baseline_mean = param_matrix[..., :obj.fps].mean(axis=-1, keepdims=True)
        baseline_std = param_matrix[..., :obj.fps].std(axis=-1, keepdims=True)
    elif param_matrix.ndim == 5:  # either shape (n_repeats, n_ori, n_SF, n_cells, n_timepoints) or (n_repeats, n_SF, n_TF, n_cells, n_timepoints)
        baseline_mean = param_matrix[..., :obj.fps].mean(axis=(2, -1), keepdims=True)
        baseline_std = param_matrix[..., :obj.fps].std(axis=(2, -1), keepdims=True)
    elif param_matrix.ndim == 3:   # FFF, response matrix: shape (n_repeats, n_cells, n_timepoints)

        # take first 4sec of response (baseline) > mean/std across time/repeat axis > shape (n_cells)
        # we can assume that we have consistent baselines across repeats and orientations
        baseline_mean = param_matrix[...,:obj.fps*4].mean(axis = (-1), keepdims = True)
        baseline_std = param_matrix[...,:obj.fps*4].std(axis = (-1), keepdims = True)

    # z score the response, and then average across repeats (shape = ((n_repeats, n_ori, n_cells, n_timepoints)
    return ((param_matrix - baseline_mean) / baseline_std)

def get_cell_by_cell_threshold (obj, param_matrix, zscore_threshold = None, std_threshold = None): #TO DO !!!!!
    '''
    :param param_matrix: shape = ((n_repeats, n_ori, n_cells, n_timepoints))
    :param zscore_threshold:
    :param std_threshold:
    :return: cell_theshold, shape ((n_cells))
    '''

    if zscore_threshold:  # compute z score of entire trace, then pick some global threshold (for all cells) of how many std above baseline
        z_scored_response = zscore_baseline(obj, param_matrix) # array of responses, shape ((n_repeats, n_ori, ..., n_cells, timepoints))
        cell_threshold = np.ones((z_scored_response.shape[-2])) * zscore_threshold # shape n_cells

    elif std_threshold: # compute individual (cell-by-cell) threshold
        # either compute DC offset, or deltaF/

        #  to just get the threshold for each cell, taking the entire response matrix and taking std over orientations, repeats and timepoints
        #  and then multiplying by some threshold multiplying

        # DELTA F / F, shape = ((n_repeats, n_ori, n_cells, n_timepoints
        dFoF_response = deltaf_over_f(obj, param_matrix) # array of responses, shape ((n_repeats, n_ori, ..., n_cells, timepoints))

        if param_matrix.ndim == 4:  # param_matrix: shape (n_repeats, n_ori, n_cells, n_timepoints)

            # first second of the response (baseline period) > std over orientations, repeats and timepoints > multiply each cell by threshold
            cell_threshold = dFoF_response[..., :obj.fps].std(axis=(0, 1, -1)) * std_threshold

        elif param_matrix.ndim == 5:  # either shape (n_repeats, n_ori, n_SF, n_cells, n_timepoints) or (n_repeats, n_SF, n_TF, n_cells, n_timepoints)
            #  std over orientations, SFs (and TFs), repeats and timepoints
            cell_threshold = dFoF_response[..., :obj.fps].std(axis=(0, 1, 2, -1)) * std_threshold

        elif param_matrix.ndim == 3:  # FFF, response matrix: shape (n_repeats, n_cells, n_timepoints)
            # take first 4sec of response (baseline period) > std over repeats and timepoints
            cell_threshold = dFoF_response[..., :obj.fps*4].std(axis=(0, -1)) * std_threshold

    return cell_threshold


def zscore_thresholding (object, animal, day, subfile):
    '''
    Considering responses at the best 'sf', calculate a z score based on the baseline
    Determines which cells were responsive

    :param object:
    :param parameter_matrix:  shape ((n_repeats, n_orientation, n_sf, n_cells, n_timepoints))
    :param zscore_threshold: 2
    :param std_threshold: 1.5

    '''
    if 'chirp' not in subfile:

        # since we're z scoring according to the baseline period, want to grab raw responses so we don't z score twice
        # either shape (n_repeats, n_ori, n_cells, timepoints), (n_repeats, n_ori, n_SF, n_cells, n_timepoints) or (n_repeats, n_SF, n_TF, n_cells, n_timepoints)
        parameter_matrix, _ = build_parameter_matrix(object, animal, day, subfile, response_window='whole', zscore=False)

        # 8 orientations x 6 repeats
        if (object.dat[animal][day][subfile]['n_SF'] == 1) and (object.dat[animal][day][subfile]['n_TF'] == 1):

            # ZSCORED RESPONSES, shape = ((n_repeats, n_ori, n_cells, n_timepoints
            z_scored_response = zscore_baseline(object, parameter_matrix)

            # DELTA F / F, shape = ((n_repeats, n_ori, n_cells, n_timepoints
            dFoF_response = deltaf_over_f (object, parameter_matrix)

            print('zscore thresh:', object.zscore_threshold,'; std thresh:', object.std_threshold)
            cell_threshold = get_cell_by_cell_threshold(object, parameter_matrix, zscore_threshold=object.zscore_threshold, std_threshold=object.std_threshold)

            # shape n_cells > take the z scored responses during the 'moving grating' period > average (n_ori, n_cells, n_timepoints) over orientations/repeats to get shape (n_cells)
            #z_scored_response_mean = z_scored_response[..., object.fps * 2: object.fps * 5].mean(axis=(0,-1))
            # shape (n_ori, n_cells)
            z_scored_response_mean = z_scored_response[..., object.fps * 2: object.fps * 5].mean(axis=(0,-1))
            dFoF_response_mean = dFoF_response[..., object.fps * 2: object.fps * 5].mean(axis=(0, -1))
            #z_scored_response_median = np.median(z_scored_response[..., object.fps * 2: object.fps * 5], axis=(0,-1))

            # see if the mean response during moving exceeds the cell std threshold
            #cell_exceeds_threshold = z_scored_response_mean > cell_threshold
            # shape (n_ori, n_cells) > (n_cells)
            # cell_exceeds_threshold = (z_scored_response_mean > cell_threshold[None,:]).any(axis=0) & (
            #             z_scored_response_median > cell_threshold[None,:] * 0.4).any(axis=0)

            # to then actually see which cells pass threshold, im taking the dfOF of the moving period, and averaging across repeats
            # and timepoints to get an array of shape (ntheta, n_cells).
            # then a cell is responsive if, for any theta, it passes the cell threshold defined above
            if object.zscore_threshold is None:
                cell_exceeds_threshold = (dFoF_response_mean > cell_threshold[None, :]).any(axis=0)
            elif object.std_threshold is None:
                cell_exceeds_threshold = (z_scored_response_mean > cell_threshold[None, :]).any(axis=0)

            # # shape (n_ori, n_cells, n_timepoints) > look across any timepoint
            # z_scored_response_mean = z_scored_response[..., object.fps * 2: object.fps * 5].mean(axis=0)
            # # TAKE A WINDOW AROUND THE PEAK
            # window = 3 # n frames we want to take
            # peak_idx = np.argmax(z_scored_response_mean, axis=-1)   # peak idx across timepoints, shape (n_ori, n_cells)
            # window_idx = peak_idx[..., None] + np.arange(-window, window + 1) # shape (n_ori, n_cells, 2*window + 1)
            # valid_mask = (window_idx >= 0) & (window_idx < z_scored_response_mean.shape[-1])
            # window_idx = np.clip(window_idx, 0, z_scored_response_mean.shape[-1] - 1) # clip to valid timerange
            #
            # # index z scored array according to window idx
            # vals = np.take_along_axis(z_scored_response_mean, window_idx, axis=2) #  shape (n_ori, n_cells, 2*window + 1)
            # vals[~valid_mask] = np.nan  # nan out out of bounds values
            #
            # peak_window_means = np.nanmean(vals, axis=-1) # shape n_ori, n_cells
            # cell_exceeds_threshold = (peak_window_means > cell_threshold[None,:]).any(axis=0)


        else: #SFxO or SFx TF

            # ZSCORED RESPONSES, either shape (n_repeats, n_ori, n_SF, n_cells, n_timepoints) or (n_repeats, n_SF, n_TF, n_cells, n_timepoints)
            z_scored_response = zscore_baseline(object, parameter_matrix)

            # DELTA F / F, seither shape (n_repeats, n_ori, n_SF, n_cells, n_timepoints) or (n_repeats, n_SF, n_TF, n_cells, n_timepoints)
            dFoF_response = deltaf_over_f (object, parameter_matrix)

            # shape n_cells > cell-by-cell threshold
            cell_threshold = get_cell_by_cell_threshold(object, parameter_matrix, zscore_threshold=object.zscore_threshold, std_threshold=object.std_threshold)

            # shape n_cells >  z scored responses during the 'moving grating' period > average over repeats (shape = n_cells)
            #z_scored_response_mean = z_scored_response[..., object.fps * 2: object.fps * 5].mean(axis=(0,1,2,-1))
            # take the z scored responses during the 'moving grating' period > average over repeats (shape = (n_ori, n_SF, n_cells) or (n_SF, n_TF, n_cells)
            z_scored_response_mean = z_scored_response[..., object.fps * 2: object.fps * 5].mean(axis=(0,-1))
            z_scored_response_median = np.median(z_scored_response[..., object.fps * 2: object.fps * 5], axis=(0,-1))

            # cell_exceeds_threshold = ((z_scored_response_mean > zscore_threshold) & (
            #             z_scored_response_median > (zscore_threshold - 1))).any(axis=(0, 1, 3))

            #cell_exceeds_threshold = z_scored_response_mean > cell_threshold
            # cell_exceeds_threshold = (z_scored_response_mean > cell_threshold[:, None]).any(axis=1) & (
            #             z_scored_response_median > cell_threshold[:, None] * 0.9).any(axis=1)
            #cell_exceeds_threshold = z_scored_response_mean > cell_threshold
            # cell_exceeds_threshold = (z_scored_response_mean > cell_threshold[None, None, :, None]).any(axis=(0,1,-1)) & (
            #             z_scored_response_median > cell_threshold[None, None,:, None] * 0.9).any(axis=(0,1,-1))

            # # check if across all orientations (SFxO) or SF (SFxTF), mean response cell exceeds threshold
            # cell_exceeds_threshold = (z_scored_response_mean.mean(axis = 1) > cell_threshold[None, :, None]).any(axis=(0,-1)) & (
            #             z_scored_response_median.mean(axis = 1) > cell_threshold[None,:, None] * 0.3).any(axis=(0,-1))
            #
            # check if across all orientations (SFxO) or SF (SFxTF), mean response cell exceeds threshold
            #cell_exceeds_threshold = (z_scored_response_mean > cell_threshold[None, None, :]).any(axis=(0,1)) & (z_scored_response_median > cell_threshold[None, None,:] * 0.4).any(axis=(0,1))

            cell_exceeds_threshold = (z_scored_response_mean > cell_threshold[None, None, :]).any(axis=(0, 1))

    else: # full-field flash
        # exclude first repeat
        response_matrix = object.dat[animal][day][subfile]['responses_ttls'][1:] # shape n_repeats, cells, timepoints

        # ZSCORED RESPONSES, shape =  (shape = ((n_repeats, n_cells, n_timepoints))
        z_scored_response = zscore_baseline(object, response_matrix)

        # DELTA F / F, shape =  (shape = ((n_repeats, n_cells, n_timepoints))
        dFoF_response = deltaf_over_f(object, response_matrix)

        # shape n_cells > cell-by-cell threshold
        cell_threshold = get_cell_by_cell_threshold(object, response_matrix, zscore_threshold=object.zscore_threshold, std_threshold=object.std_threshold)

        # take the z scored responses during the 'ON' period > average over repeats & timepoints (shape = (n_cells))
        # 0-4 = grey, 4-8 = black, 8-12 = white, 12-16 = black, 16-20 = grey
        #z_scored_response_mean = z_scored_response[..., object.fps * 4 : object.fps * 16].mean(axis = (0,-1))
        # shape n_cells, timepoints
        z_scored_response_mean = z_scored_response[..., object.fps * 4: object.fps * 16].mean(axis=0)
        z_scored_response_median = np.median(z_scored_response[..., object.fps * 4: object.fps * 16], axis=0)

        # see if the response at peak exceeds the mean zscore threshold (shape cell > shape cell)
        # take 90% of threshold and see if median crosses it
        cell_exceeds_threshold = (z_scored_response_mean > cell_threshold[:, None]).any(axis=1) & (z_scored_response_median > cell_threshold[:, None]*0.9).any(axis=1)
        #(z_scored_response_mean > zscore_threshold-1).any(axis=(-1))
        #cell_exceeds_threshold = ((z_scored_response_mean > zscore_threshold) & (z_scored_response_median > (zscore_threshold-1) )).any(axis=(-1))


    return z_scored_response, dFoF_response, cell_exceeds_threshold

def build_tuning_curves (object, animal, day, subfile, zscore = False, avg_over_param = None, sf_i = None):
    '''
    :param object:
    :param animal:
    :param day:
    :param subfile:
    :param zscore: fine to z-score when building TCs, but can't z-score if trying to calculate the complex phase
    :param avg_over_param:
        - 'SFs': take the average across SFs,
        - 'orientations': take the average across orientations
        - None: do not average across anything, rather grab one specific SF value at index sf_i
    :return:
        - tuning_curve: array of shape ((n_repeats, n_orientation/n_SF, cells))
        - theta: array of shape n_theta, theta of gratings in increasing order
        - sfs: array of shape n_sfs, SFs of gratings in increasing order
    '''
    if (object.dat[animal][day][subfile]['n_SF'] == 1) and (object.dat[animal][day][subfile]['n_TF'] == 1):
        # responses_ordered: shape ((n_repeats, n_orientation, cells, timepoints))
        # 1s wait + 1s static + 3s moving + 1s static
        responses_ordered, thetas = build_parameter_matrix(object, animal, day, subfile, response_window='whole', zscore=zscore)
        response = responses_ordered
    else:
        # responses_ordered: shape ((n_repeats, n_orientation, n_sf, cells, timepoints))
        # 1s wait + 1s static + 3s moving + 1s static
        responses_ordered, thetas, sfs, tfs = build_parameter_matrix(object, animal, day, subfile, response_window='whole', zscore=zscore)

        if avg_over_param == 'SFs':
            # average over SFs > shape ((n_repeats, n_orientation, cells, timepoints))
            response = responses_ordered.mean(axis=2)
        elif avg_over_param == 'orientations':
            # average over orientations > shape ((n_repeats, n_sf, cells, timepoints))
            response = responses_ordered.mean(axis=2)
        # if avg_over_param is None

        elif (avg_over_param is None) and (sf_i is not None):
            if isinstance(sf_i, np.ndarray):

                # not averaging over SFs - taking a specific index of SFs living at sf_i (specific to each cell)- shape ((n_repeats, n_orientation, cells, timepoints))
                # array of shape n_cells > representing index of preferred sf
                preferred_sf_idx = pref_sf(object, responses_ordered)
                cell_indices = np.arange(responses_ordered.shape[3])
                # for each cell, only taking response at preferred index > shape (n_repeats, n_orientations, n_cells, n_timepoints)
                response = responses_ordered[:, :, preferred_sf_idx, cell_indices, :]
            elif isinstance(sf_i, int):
                response = responses_ordered[:, :, sf_i, :, :]

    # baseline period is the average over the first second of stimulus (WAIT period) > shape ((n_repeats, n_orientation/n_SF, cells))
    baseline = response[:, :, :, :object.fps].mean(axis=-1)

    # moving period is average of the first 2 seconds moving > shape ((n_repeats, n_orientation/n_SF, cells))
    tuning_curve_moving = response[:, :, :, 2 * (object.fps):4 * (object.fps):].mean(axis=-1)

    # baseline subtracted tuning curve, ensuring that all responses are non-negative> shape (on_repeats, n_orientation/n_SF, cells))
    tuning_curve = np.maximum((tuning_curve_moving - baseline), 0)

    if not ((object.dat[animal][day][subfile]['n_SF'] == 1) and (object.dat[animal][day][subfile]['n_TF'] == 1)):

        # store in object > shape ((n_repeats, n_orientation/n_SF, cells))
        if (avg_over_param is None) and (sf_i is not None):

            object.dat[animal][day][subfile][f'tuning_curves_sf_pref'] = tuning_curve

            if object.dat[animal][day][subfile]['n_theta'] == 8:
                #print(',,,....;,', tuning_curve.shape)
                half = tuning_curve.shape[1]//2 # n_theta / 2
                object.dat[animal][day][subfile][f'ori_tuning_curves_sf_pref'] = (tuning_curve[:,:half,:] + tuning_curve[:,half:,:])/2

        else:
            object.dat[animal][day][subfile][f'tuning_curves_avg_over_{avg_over_param}'] = tuning_curve

    return responses_ordered, tuning_curve, thetas

def complex_phase (object, animal, day, subfile, avg_over_param = 'SFs', sf_i = None):

    # store in object > shape ((n_repeats, n_orientation/n_SF, cells))
    param_matrix, tuning_curves, thetas = build_tuning_curves(object, animal, day, subfile, zscore = False, avg_over_param = avg_over_param, sf_i = sf_i)

    # shape n_orientations
    theta_rad = np.deg2rad(thetas)

    # mean response to the stimulus across repeats > shape: (n_cells, n_orientations/n_SF)
    responses_theta = tuning_curves.mean(axis=0).T

    # shape (n_cells);  NB: np.exp outputs shape n_orientations
    # in python, j is the imaginary component
    complex_num_orientation = (responses_theta * np.exp(1j * 2 * theta_rad)).sum(axis=-1) / (responses_theta.sum(axis=-1))
    # note that e^(i2theta) doubles the angles because orientations are periodic over 180deg, not 360deg.
    # this ensures that responses to 0deg and 180deg are treated as equivalent (represent same orientation)
    # consequently, the resultant vector encodes the doubled angle of the preferred orientation.
    # note that to calculate direction selectivity you wouldnt have the 2 factor

    # shape (n_cells) > magnitude (absolute value) of the complex number > represents the amplitude of the response (how strongly neuron responds) = OSI (should be between 0 & 1)
    osi = np.abs(complex_num_orientation)
    osi = np.nan_to_num(osi, nan=0) #if the OSI is nan, put it to 0

    # the argument (angle / preferred angle) of the complex number encodes the preferred orientation (the orientation to which the neuron responds most strongly).
    pref_orientation = np.rad2deg(np.angle(complex_num_orientation))/2  # neuron’s preferred orientation
    # divide by 2 because of the i*theta*2 in the original equation

    # For direction selectivity, the angles (theta_rad) should represent the actual stimulus directions, with a full 360° range.
    complex_num_direction = (responses_theta * np.exp(1j * theta_rad)).sum(axis=-1) / (responses_theta.sum(axis=-1))
    dsi = np.abs(complex_num_direction) # ranges between 0 and 1, where 0 = no direction selectivity, 1 = perfect direction selectivity.
    pref_direction = np.rad2deg(np.angle(complex_num_direction)) #  neuron’s preferred direction.
    # PD is periodic over 360°, distinguishing between opposite directions (e.g., 0° and 180° are different).
    # PO is periodic over 180°, meaning that responses to 0° and 180° are treated as equivalent because they represent the same orientation.

    return param_matrix, tuning_curves, (complex_num_orientation, osi, pref_orientation), (complex_num_direction, dsi, pref_direction), thetas

def complex_phase_from_tuning (tuning_curves, thetas):
    '''
    :param tuning_curves: shape ((n_repeats, n_orientation/n_SF, cells))
    :param thetas: shape ((n_orientation))
    :param sfs: shape ((n_SF))
    :return:
    '''

    # shape n_orientations
    theta_rad = np.deg2rad(thetas)

    # mean response to the stimulus across repeats > shape: (n_cells, n_orientations/n_SF)
    # # HOW DO I GET OSI FOR AVERAGE ACROSS ORIENTATINO)
    responses_theta = tuning_curves.mean(axis=0).T

    # shape (n_cells);  NB: np.exp outputs shape n_orientations
    # in python, j is the imaginary component
    complex_ori = (responses_theta * np.exp(1j * 2 * theta_rad)).sum(axis=-1) / (responses_theta.sum(axis=-1))
    # note that e^(i2theta) doubles the angles because orientations are periodic over 180deg, not 360deg.
    # this ensures that responses to 0deg and 180deg are treated as equivalent (represent same orientation)
    # consequently, the resultant vector encodes the doubled angle of the preferred orientation.
    # note that to calculate direction selectivity you wouldnt have the 2 factor

    # shape (n_cells) > magnitude (absolute value) of the complex number > represents the amplitude of the response (how strongly neuron responds) = OSI (should be between 0 & 1)
    osi = np.abs(complex_ori)
    osi = np.nan_to_num(osi, nan=0) #if the OSI is nan, put it to 0

    # the argument (angle / preferred angle) of the complex number encodes the preferred orientation (the orientation to which the neuron responds most strongly).
    pref_orientation = np.rad2deg(np.angle(complex_ori))/2  # preferred angle
    # divide by 2 because of the i*theta*2 in the original equation

    complex_direction = (responses_theta * np.exp(1j * theta_rad)).sum(axis=-1) / (responses_theta.sum(axis=-1))
    dsi = np.abs(complex_direction)  # ranges between 0 and 1, where 0 = no direction selectivity, 1 = perfect direction selectivity.
    pref_direction = np.rad2deg(np.angle(complex_direction))  # neuron’s preferred direction.

    return (complex_ori, osi, pref_orientation), (complex_direction, dsi, pref_direction)

## need to ttest output of these two functions (tuning curves & complex phase) - make sure output looks the same as previously
# then in coproate it into the poarameter matrix plotting function

#
# def parameter_matrix_plot(object, animal, day, subfile):
#
#     # responses_ordered: shape ((n_repeats, n_orientation, n_sf, cells, timepoints))
#     param_matrix, thetas, sfs, tfs = build_parameter_matrix(object, animal, day, subfile, response_window='whole', zscore=True)
#
#     # PROPERTIES FOR SF PREFERRED
#     param_matrix_sf_pref, tuning_curves_sf_pref, complex_sf_pref, osi_sf_pref, pref_orientation_sf_pref, _, _ = complex_phase(object, animal, day, subfile, sf_i=pref_sf(object, animal, day, subfile))
#
#     if (object.dat[animal][day][subfile]['n_SF'] > 1) and (object.dat[animal][day][subfile]['n_theta'] > 1):
#
#         if not os.path.exists(os.path.join(object.save_path, 'theta x SF + polar', animal)):
#             os.makedirs(os.path.join(object.save_path, 'theta x SF + polar', animal))
#
#         with PdfPages(os.path.join(object.save_path, 'theta x SF + polar', animal,
#                                    f'{animal} theta x SF + polar ({day}, {subfile}).pdf')) as pdf:
#
#             #for cell_i in range(param_matrix.shape[-2]):
#             for cell_i in range(2):
#
#                 fig, ax = plt.subplots(nrows=len(thetas)+1, ncols=len(sfs) + 1, figsize=(14, 9), sharey=True, sharex=True)
#                 colors = plt.cm.plasma(np.linspace(0, 0.85, len(sfs) * len(thetas) + 5))
#                 colors_cells = plt.cm.plasma(np.linspace(0, 0.9, param_matrix.shape[-2]))
#                 counter = 0
#                 for i_theta, theta in enumerate(thetas):
#                     for i_sf, sf in enumerate(sfs):
#                         [ax[i_theta, i_sf].plot(param_matrix[i_repeat, i_theta, i_sf, cell_i], c=colors[counter],
#                                                 alpha=0.5) for i_repeat in range(param_matrix.shape[0])]
#                         ax[i_theta, i_sf].plot(param_matrix[:, i_theta, i_sf, cell_i].mean(axis=0), c='black',
#                                                alpha=0.8)
#
#                         if not i_sf:  # if its 0
#                             ax[i_theta, i_sf].set_ylabel('Time', rotation=0, fontsize=11)
#                             ax[i_theta, i_sf].set_ylabel(f'{int(theta)}\u00b0', rotation=0, fontsize=11)
#                             ax[i_theta, i_sf].yaxis.set_label_coords(-0.25, 0.35)  # Adjust the y-label position
#                         if not i_theta:
#                             ax[i_theta, i_sf].set_title(f'{sf}', fontsize=11)
#
#                         ax[i_theta, i_sf].set_ylim(
#                             [param_matrix[:, :, :, cell_i].min(), param_matrix[:, :, :, cell_i].max()])
#                         ax[i_theta, i_sf].axvline(object.fps, c='grey', alpha=0.3)
#                         ax[i_theta, i_sf].set_xticks([])
#                         ax[i_theta, i_sf].set_yticks([])
#
#                         counter += 1
#
#                 # plot the average response to an orientation, averaged across all SFs
#                 for i_theta, theta in enumerate(thetas):
#                     [ax[i_theta, -1].plot(param_matrix[i_repeat, i_theta, :, cell_i].mean(axis=0),
#                                           c='mediumslateblue', alpha=0.5) for i_repeat in range(param_matrix.shape[0])]
#                     ax[i_theta, -1].plot(param_matrix[:, i_theta, :, cell_i].mean(axis=0).mean(axis=0), c='black',
#                                          alpha=0.8)
#                     # ax[i_theta, -1].set_ylim ([responses_ordered[:, :, :, cell_i, :].mean(axis = 2).min(), responses_ordered[:, :, :, cell_i, :].mean(axis = 2).max()])
#                     ax[i_theta, -1].set_yticks([])
#                     ax[i_theta, -1].axvline(object.fps, c='grey', alpha=0.3)
#
#                 # plot the average response to an SF, averaged across all orientations
#                 polar_axes = []
#                 for i_sf, sf in enumerate(sfs):
#                     ax[-1, i_sf].axis('off')
#                     #ax[-1, i_sf].clear()
#                     # Define the position for each polar plot
#                     polar_ax = fig.add_subplot(len(thetas) + 1, len(sfs) + 1, (len(thetas)) * (len(sfs) + 1) + i_sf + 1,projection='polar')
#                     polar_axes.append(polar_ax)
#
#                     # shape ((n_repeats, n_orientation, cells))
#                     response_sf, _, _ = build_tuning_curves(object, animal, day, subfile, avg_over_param=None, sf_i=i_sf)
#
#                     # r = vector of responses for each direction(response vector)
#                     r = response_sf[:, :, cell_i].mean(axis=0)
#                     r /= r.sum()  # normalizing responses so they're between 0 and 1
#                     theta = np.deg2rad(thetas)
#
#                     # to join the last point and first point
#                     idx = np.arange(r.shape[0] + 1)
#                     idx[-1] = 0
#
#                     polar_ax.plot(theta, r, linewidth=2, color=colors_cells[cell_i],
#                                   alpha=0.6)  # , color=scalarMap.to_rgba(cell), alpha=0.6)
#                     polar_ax.plot(theta[idx], r[idx], linewidth=2, color=colors_cells[cell_i],
#                                   alpha=0.6)  # , color=scalarMap.to_rgba(cell), alpha=0.6)
#                     polar_ax.set_thetagrids([0, 90, 180, 270], y=0.2,
#                                             labels=['0', '\u03c0' + '/2', '\u03c0', '3' + '\u03c0' + '/2'],
#                                             fontsize=8)  # labels = ['0', '','\u03c0','']
#
#                     # [ax[-1, i_sf].plot(responses_ordered[i_repeat, :, i_sf, cell_i].mean(axis=0), c='mediumslateblue',
#                     #                    alpha=0.5) for i_repeat in range(responses_ordered.shape[0])]
#                     # ax[-1, i_sf].plot(responses_ordered[:, :, i_sf, cell_i].mean(axis=0).mean(axis=0), c='black',
#                     #                   alpha=0.8)
#                     # ax[-1, i_sf].set_yticks([])
#                     # ax[-1, i_sf].axvline(object.fps, c='grey', alpha=0.3)
#                     #
#                     # counter += 1
#
#                 ax[-1, -1].axis('off')
#                 #ax[-1, 0].set_ylabel('av. across\n orientations', rotation=0, fontsize=11)
#                 ax[-1, 0].yaxis.set_label_coords(-0.6, 0.2)  # Adjust the y-label position
#                 ax[0, -1].set_title('av. across SFs')
#                 fig.text(0.04, 0.5, 'Orientation', va='center', rotation='vertical', fontsize=12)
#                 fig.text(0.37, 0.93, 'Spatial frequency', va='center', rotation='horizontal', fontsize=12)
#
#                 # # adding polar plot to show tuning curve
#                 # polar_ax = fig.add_axes([0.8, 0.3, 0.15, 0.4], polar=True)
#                 #
#                 # # r = vector of responses for each direction(response vector)
#                 # r = object.dat[animal][day][subfile]['tuning_curves'][:, :, cell_i].mean(axis=0)
#                 # r /= r.sum()  # normalizing responses so they're between 0 and 1
#                 # theta = np.deg2rad(object.dat[animal][day][subfile]['theta'])
#                 #
#                 # # to join the last point and first point
#                 # idx = np.arange(r.shape[0] + 1)
#                 # idx[-1] = 0
#                 #
#                 # polar_ax.plot(theta, r, linewidth=2, color=colors_cells[cell_i],
#                 #               alpha=0.6)  # , color=scalarMap.to_rgba(cell), alpha=0.6)
#                 # polar_ax.plot(theta[idx], r[idx], linewidth=2, color=colors_cells[cell_i],
#                 #               alpha=0.6)  # , color=scalarMap.to_rgba(cell), alpha=0.6)
#                 # polar_ax.set_thetagrids([0, 90, 180, 270], y=0.1,
#                 #                         labels=['0', '\u03c0' + '/2', '\u03c0', '3' + '\u03c0' + '/2'],
#                 #                         fontsize=8)  # labels = ['0', '','\u03c0','']
#                 # polar_ax.set_rlabel_position(45)  # r is normalized response
#                 # # polar_ax.tick_params(axis='y', labelsize=8)
#                 # polar_ax.set_rticks(np.round(np.linspace(0, r.max(), 2), 1))
#                 # polar_ax.grid(True)
#                 # polar_ax.set_title(
#                 #     f'ROI #{cell_i} (OSI = {np.round(object.dat[animal][day][subfile]["OSI"][cell_i], 2)})',
#                 #     fontsize=10, pad=4)
#
#                 plt.tight_layout(rect=[0, 0.05, 0.75, 0.92])
#
#                 pdf.savefig()
#                 plt.show()
#                 if not object.show_plots:
#                     plt.close()
#
# #parameter_matrix_plot(data_object, 'EC_GCaMP6s_09', '20241122', 'small_SFxO_000_012')

def store_metrics(object, animal, day, subfile):
    '''
    Calculates and stores several important metrics to be plotted by plot_parameter_matrix
    '''

    # if only orientations x repeats
    if ((object.dat[animal][day][subfile]['n_SF'] == 1) and (object.dat[animal][day][subfile]['n_TF'] == 1)):
        # param_matrix: shape ((n_repeats, n_orientation, n_sf, cells, timepoints))
        param_matrix, object.dat[animal][day][subfile]['orientations'] = build_parameter_matrix(object, animal, day, subfile, response_window='whole', zscore=True)

        _, tuning_curve, _ = build_tuning_curves(object, animal, day, subfile, zscore=False)

    # if SFxO or SFxTF
    else:
        param_matrix, object.dat[animal][day][subfile]['orientations'], object.dat[animal][day][subfile][
            'spatial_frequencies'],object.dat[animal][day][subfile][
            'temporal_frequencies'] = build_parameter_matrix(object, animal, day, subfile, response_window='whole',
                                                            zscore=True)

        # shape ((n_cells))
        object.dat[animal][day][subfile]['preferred_SF_idx'] = pref_sf(object, param_matrix)

        _, tuning_curve, _ = build_tuning_curves(object, animal, day, subfile, zscore=False, avg_over_param=None,
                                                 sf_i=object.dat[animal][day][subfile]['preferred_SF_idx'])

    orientation, direction = complex_phase_from_tuning(tuning_curve, object.dat[animal][day][subfile]['orientations'])

    _, object.dat[animal][day][subfile]['OSI'], object.dat[animal][day][subfile]['preferred_orientation'] = orientation
    _, object.dat[animal][day][subfile]['DSI'], object.dat[animal][day][subfile]['preferred_direction'] = direction

    object.dat[animal][day][subfile]['zscored_matrix_baseline'], object.dat[animal][day][subfile]['dfof_matrix_baseline'], object.dat[animal][day][subfile]['thresholded_cells'] = zscore_thresholding(object, animal, day, subfile)




    # if SF x O only > n_theta = 8, n_sf = 5, n_tf = 1
    if  ((object.dat[animal][day][subfile]['n_theta'] > 1) and (object.dat[animal][day][subfile]['n_SF'] == 5) and (object.dat[animal][day][subfile]['n_TF'] == 1)):

        # sf_tuning_curve: SF tuning curve at the best orientation; shape (n_cells, n_repeats, n_sf)
        # pref_theta_idx: preferred theta for each cell; shape (n_cells)
        object.dat[animal][day][subfile]['sf_tuning_curve'], pref_theta_idx = sf_tuning_curve(object, param_matrix)
        object.dat[animal][day][subfile]['preferred_theta'] = object.dat[animal][day][subfile]['orientations'][pref_theta_idx]
def parameter_matrix_plot(object, animal, day, subfile, zscore_threshold = 3):

    if ((object.dat[animal][day][subfile]['n_SF'] == 1) and (object.dat[animal][day][subfile]['n_TF'] == 1)):

        # param_matrix: shape ((n_repeats, n_orientation, cells, timepoints))
        param_matrix, thetas = build_parameter_matrix(object, animal, day, subfile, response_window='whole', zscore=True)

        colors = plt.cm.plasma(np.linspace(0, 0.9, len(thetas)))
        colors_cells = plt.cm.plasma(np.linspace(0, 0.9, param_matrix.shape[-2]))

        # shape ((n_repeats, n_orientation, cells))
        responses_ordered, tuning_curve, thetas = build_tuning_curves(object, animal, day, subfile)
        (_, osi, _), (_, dsi, _) = complex_phase_from_tuning(tuning_curve, thetas)

        if not os.path.exists(os.path.join(object.save_path, 'theta x SF + polar', animal)):
            os.makedirs(os.path.join(object.save_path, 'theta x SF + polar', animal))
        with PdfPages(os.path.join(object.save_path, 'theta x SF + polar', animal, f'{animal} theta x SF + polar ({day}, {subfile}).pdf')) as pdf:
            for cell_i in range(param_matrix.shape[-2]):
                fig, ax = plt.subplots(nrows=1, ncols=len(thetas) + 1, figsize=(17, 3), sharey=True, sharex=True)
                counter = 0

                for i_theta, theta in enumerate(thetas):
                    [ax[i_theta].plot(param_matrix[i_repeat, i_theta, cell_i], c=colors[counter], alpha=0.5) for i_repeat in range(param_matrix.shape[0])]
                    ax[i_theta].plot(param_matrix[:, i_theta, cell_i].mean(axis=0), c='black', alpha=0.8)

                    if not i_theta:  # if its 0 (left col)
                        ax[i_theta].set_ylabel('Response (z-score)', rotation=90, fontsize=11)

                    ax[i_theta].set_box_aspect(1)
                    ax[i_theta].set_title(f'{int(theta)}\u00b0', rotation=0, fontsize=11)
                    ax[i_theta].set_ylim([param_matrix[:, :, cell_i].min(), param_matrix[:, :, cell_i].max()])
                    ax[i_theta].axvline(object.fps, c='grey', alpha=0.3)
                    ax[i_theta].axvspan(object.fps * 2, object.fps * 5, color='grey', alpha=0.2)
                    ax[i_theta].set_xticks([])
                    ax[i_theta].set_yticks([])
                    counter += 1

                    # average across repeats
                    rmax = np.array(tuning_curve[:, :, cell_i].mean(axis=0)).max()

                    # plot the average response to an SF, averaged across all orientations
                    ax[-1].axis('off')
                    # define position & size of the polar plot
                    polar_ax = fig.add_axes(
                        [0.73, # x-coordinate
                         0.30,  # y-coordinate
                         0.28,  # width
                         0.38,  # height (increase for larger plots)
                        ], polar=True)

                    # r = vector of responses for each direction(response vector)
                    r = tuning_curve[:, :, cell_i].mean(axis=0)
                    r /= rmax
                    # r /= r.sum()  # normalizing responses so they're between 0 and 1
                    theta = np.deg2rad(thetas)

                    # to join the last point and first point
                    idx = np.arange(r.shape[0] + 1)
                    idx[-1] = 0

                    polar_ax.plot(theta, r, linewidth=2, color=colors_cells[cell_i], alpha=0.6)  # , color=scalarMap.to_rgba(cell), alpha=0.6)
                    polar_ax.plot(theta[idx], r[idx], linewidth=2, color=colors_cells[cell_i], alpha=0.6)  # , color=scalarMap.to_rgba(cell), alpha=0.6)
                    polar_ax.set_thetagrids([0, 90, 180, 270], y=0.15, labels=['0', '\u03c0' + '/2', '\u03c0', '3' + '\u03c0' + '/2'], fontsize=8)  # labels = ['0', '','\u03c0','']
                    polar_ax.set_rlabel_position(45)  # r is normalized response
                    # polar_ax.set_rticks([np.round(r.max(), 1)])
                    polar_ax.set_rticks([1])
                    polar_ax.set_rmax(1)
                    polar_ax.grid(True)

                if object.dat[animal][day][subfile]['thresholded_cells'][cell_i]:
                    plt.suptitle(f'Cell {cell_i}', fontsize = 16, color = 'green', fontweight = 'bold')
                else:
                    plt.suptitle(f'Cell {cell_i}', fontsize=16, color='r')
                #plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                if not object.show_plots:
                    plt.close()

    elif ((object.dat[animal][day][subfile]['n_theta'] != 1) and (object.dat[animal][day][subfile]['n_TF'] == 1)):
        # param_matrix: shape ((n_repeats, n_orientation, n_sf, cells, timepoints))
        param_matrix, thetas, sfs, tfs = build_parameter_matrix(object, animal, day, subfile, response_window='whole', zscore=True)

        # shape ((n_cells))
        preferred_sf = pref_sf(object, param_matrix)

        # sf_tc: SF tuning curve at the best orientation: shape n_cells, n_repeats, n_sfs
        # pref_theta_idx: shape n_cells
        sf_tc, pref_theta_idx = sf_tuning_curve(object, param_matrix)

        colors = plt.cm.plasma(np.linspace(0, 0.9, len(sfs) * len(thetas)))
        colors_cells = plt.cm.plasma(np.linspace(0, 0.9, param_matrix.shape[-2]))

        _, tuning_curves_sf_pref, (complex_sf_pref, osi_sf_pref, pref_orientation_sf_pref), (_,_,_), _, = complex_phase(object, animal, day, subfile, sf_i=preferred_sf)

        if (object.dat[animal][day][subfile]['n_SF'] > 1) and (object.dat[animal][day][subfile]['n_theta'] > 1):
            if not os.path.exists(os.path.join(object.save_path, 'theta x SF + polar', animal)):
                os.makedirs(os.path.join(object.save_path, 'theta x SF + polar', animal))
            with PdfPages(os.path.join(object.save_path, 'theta x SF + polar', animal,
                                       f'{animal} theta x SF + polar ({day}, {subfile}).pdf')) as pdf:
                for cell_i in range(param_matrix.shape[-2]):
                    fig, ax = plt.subplots(nrows=len(thetas)+1, ncols=len(sfs) + 1, figsize=(14, 9), sharey=True, sharex=True)
                    counter = 0

                    for i_theta, theta in enumerate(thetas):
                        for i_sf, sf in enumerate(sfs):
                            [ax[i_theta, i_sf].plot(param_matrix[i_repeat, i_theta, i_sf, cell_i], c=colors[counter],
                                                    alpha=0.5) for i_repeat in range(param_matrix.shape[0])]
                            ax[i_theta, i_sf].plot(param_matrix[:, i_theta, i_sf, cell_i].mean(axis=0), c='black',
                                                   alpha=0.8)
                            if not i_sf:  # if its 0 (left col)
                                ax[i_theta, i_sf].set_ylabel('Time', rotation=0, fontsize=11)
                                if i_theta != pref_theta_idx[cell_i]:
                                    ax[i_theta, i_sf].set_ylabel(f'{int(theta)}\u00b0', rotation=0, fontsize=11)
                                else:
                                    ax[i_theta, i_sf].set_ylabel(f'{int(theta)}\u00b0', rotation=0, fontsize=11, color = 'red', fontweight='bold' )

                                ax[i_theta, i_sf].yaxis.set_label_coords(-0.25, 0.35)  # Adjust the y-label position
                            if not i_theta: # if its 0 (top row)

                                if i_sf == preferred_sf[cell_i]:
                                    ax[i_theta, i_sf].set_title(f'{sf}', fontsize=11, color = 'red', fontweight = 'bold')
                                else:
                                    ax[i_theta, i_sf].set_title(f'{sf}', fontsize=11)

                            ax[i_theta, i_sf].set_ylim(
                                [param_matrix[:, :, :, cell_i].min(), param_matrix[:, :, :, cell_i].max()])
                            ax[i_theta, i_sf].axvline(object.fps, c='grey', alpha=0.3)
                            ax[i_theta, i_sf].axvspan(object.fps * 2, object.fps * 5, color='grey', alpha=0.2)
                            ax[i_theta, i_sf].set_xticks([])
                            ax[i_theta, i_sf].set_yticks([])
                            counter += 1

                    # plot the average response to an orientation, averaged across all SFs
                    for i_theta, theta in enumerate(thetas):
                        [ax[i_theta, -1].plot(param_matrix[i_repeat, i_theta, :, cell_i].mean(axis=0),
                                              c='mediumslateblue', alpha=0.5) for i_repeat in range(param_matrix.shape[0])]
                        ax[i_theta, -1].plot(param_matrix[:, i_theta, :, cell_i].mean(axis=0).mean(axis=0), c='black',
                                             alpha=0.8)
                        # ax[i_theta, -1].set_ylim ([responses_ordered[:, :, :, cell_i, :].mean(axis = 2).min(), responses_ordered[:, :, :, cell_i, :].mean(axis = 2).max()])
                        ax[i_theta, -1].set_yticks([])
                        ax[i_theta, -1].axvline(object.fps, c='grey', alpha=0.3)
                        ax[i_theta, -1].axvspan(object.fps*2, object.fps*5, color='grey', alpha=0.2)

                    # average across repeats
                    rmax = np.array([(build_tuning_curves(object, animal, day, subfile, avg_over_param=None, sf_i=i_sf)[1][:, :, cell_i].mean(axis=0))
                                     for i_sf in range(len(sfs))]).max()

                    # plot the average response to an SF, averaged across all orientations
                    polar_axes = []

                    for i_sf, sf in enumerate(list(sfs) + [None]):
                        ax[-1, i_sf].axis('off')

                        # define position & size of the polar plot
                        polar_ax = fig.add_axes(
                            [
                                0.07 + i_sf * 0.1125,  # x-coordinate
                                0.05,  # y-coordinate
                                0.08,  # width
                                0.18,  # height (increase for larger plots)
                            ],
                            polar=True)

                        polar_axes.append(polar_ax)

                        if sf: #if sf is a valid number (i.e., is not none, we want to look at a specific sf)
                            # shape ((n_repeats, n_orientation, cells))
                            _, response_sf, _ = build_tuning_curves(object, animal, day, subfile, avg_over_param=None, sf_i=i_sf)
                            (_, osi, _), (_, dsi, _) = complex_phase_from_tuning(response_sf, thetas)

                        else: # if sf is none, we want to look at tuning across sfs
                            # shape ((n_repeats, n_orientation, cells))
                            _, response_sf, _ = build_tuning_curves(object, animal, day, subfile, avg_over_param='SFs')
                            (_, osi, _), (_, dsi, _) = complex_phase_from_tuning(response_sf, thetas)

                        # r = vector of responses for each direction(response vector)
                        r = response_sf[:, :, cell_i].mean(axis=0)
                        r /= rmax
                        #r /= r.sum()  # normalizing responses so they're between 0 and 1
                        theta = np.deg2rad(thetas)

                        # to join the last point and first point
                        idx = np.arange(r.shape[0] + 1)
                        idx[-1] = 0
                        polar_ax.plot(theta, r, linewidth=2, color=colors_cells[cell_i],
                                      alpha=0.6)  # , color=scalarMap.to_rgba(cell), alpha=0.6)
                        polar_ax.plot(theta[idx], r[idx], linewidth=2, color=colors_cells[cell_i],
                                      alpha=0.6)  # , color=scalarMap.to_rgba(cell), alpha=0.6)
                        polar_ax.set_thetagrids([0, 90, 180, 270], y=0.15,
                                                labels=['0', '\u03c0' + '/2', '\u03c0', '3' + '\u03c0' + '/2'],
                                                fontsize=8)  # labels = ['0', '','\u03c0','']
                        polar_ax.set_rlabel_position(45)  # r is normalized response
                        #polar_ax.set_rticks([np.round(r.max(), 1)])
                        polar_ax.set_rticks([1])
                        polar_ax.set_rmax(1)
                        polar_ax.grid(True)

                        if i_sf == preferred_sf[cell_i]:
                            polar_ax.set_title(
                                f'OSI: {np.round(osi[cell_i], 2)} \nDSI: {np.round(dsi[cell_i], 2)}',
                                fontsize=12, pad=4, color='red', fontweight='bold')
                        else:
                            polar_ax.set_title(
                                f'OSI: {np.round(osi[cell_i], 2)} \nDSI: {np.round(dsi[cell_i], 2)}',
                                fontsize=12, pad=4)

                    #ax[-1, -1].axis('off')
                    ax[-1, 0].yaxis.set_label_coords(-0.6, 0.2)  # Adjust the y-label position
                    ax[0, -1].set_title('av. across SFs')

                    if object.dat[animal][day][subfile]['thresholded_cells'][cell_i]:
                        plt.suptitle(f'Cell {cell_i}', fontsize = 16, color = 'green', fontweight = 'bold')
                    else:
                        plt.suptitle(f'Cell {cell_i}', fontsize=16, color='r')
                    fig.text(0.0125, 0.5, 'Orientation', va='center', rotation='vertical', fontsize=12)
                    fig.text(0.38, 0.925, 'Spatial frequency', va='center', rotation='horizontal', fontsize=12)

                    #plt.tight_layout(rect=[0.02, 0.2, 0.73, 0.92])
                    plt.subplots_adjust(left=0.07, bottom=0.2, right=0.73, top=0.88)

                    # adding plot to show SF tuning curve
                    #sf_tuning = fig.add_axes([0.765, 0.4, 0.20, 0.2])
                    sf_tuning = fig.add_axes([0.765, 0.45, 0.20, 0.2])

                    [sf_tuning.plot(sf_tc[cell_i, i_repeat], c=colors_cells[cell_i], alpha=0.4) for i_repeat in range(sf_tc.shape[1])]
                    sf_tuning.plot(sf_tc[cell_i].mean(axis=0), c='black',alpha=0.8, linewidth = 2.5)
                    #print(sfs)
                    sf_tuning.set_xticks(np.arange(sf_tc.shape[-1]))
                    sf_tuning.set_xticklabels(sfs)
                    #sf_tuning.set_xticklabels(['0.005', '0.02', '0.04', '0.08', '0.16'])
                    #sf_tuning.set_xticklabels(sfs)
                    sf_tuning.set_yticks([])
                    sf_tuning.set_xlabel('Spatial Frequency')
                    sf_tuning.set_ylabel('Response amplitude')

                    pdf.savefig()

                    if not object.show_plots:
                        plt.close()

#store_metrics(data_object, 'EC_GCaMP6s_09', '20241122', 'small_SFxO_000_012', zscore_threshold = 3)
#store_metrics(data_object, 'EC_RD1_05', '20241122', 'small_SFxO_000_002', zscore_threshold = 3)



def perform_pca (data_trials_normalized, n_components = 10):
    '''
    :param data_trials_normalized: array shape (n_trials, n_features), each trial (each row) is normalized to unit length (l2 normalization)
    '''
    # performing PCA & projection
    pca = PCA(n_components=n_components, svd_solver='full') # full forces the lapack solver
    principal_components = pca.fit_transform(data_trials_normalized)
    explained_variance_ratio = pca.explained_variance_ratio_
    return pca, principal_components, explained_variance_ratio

def participation_ratio (pca):

    # Participation Ratio (Dimensionality)
    eigenvalues = pca.explained_variance_
    pr = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
    return pr

# def filter_active_epochs (data_trials_array, response_threshold = 4, cell_threshold = 2):
#     '''
#     Remove epochs with no activity or only 1-2 cells active
#     :param data_trials_array: numpy array of shape (n_cells, n_chunks)
#     :param threshold: z-score threshold that cells need to be to be considered active
#     :return:
#     '''
#
#     # number of timepoints that exceed the threshold for each cell in every epoch > shape n_epochs, n_cells
#     timepoints_above_threshold = (data_trials_array > response_threshold).sum(axis=2)
#
#     #for each epoch, number of cells that are active in at least one timepoint (shape n_epochs)
#     cells_above_threshold = (timepoints_above_threshold > 0).sum(axis=1)
#
#     # valid epochs where more than 'cell_threshold' cells have at least one timepoint > 'response_threshold' (at least 2 co-active cells in each epoch)
#     valid_epochs = cells_above_threshold > cell_threshold  # Shape: (n_epochs,)
#
#     filtered_data_trials = data_trials_array[valid_epochs]
#
#     return filtered_data_trials, valid_epochs

def perform_pca (data_trials_normalized, n_components = 10):
    '''
    :param data_trials_normalized: array shape (n_trials, n_features), each trial (each row) is normalized to unit length (l2 normalization)
    '''
    # performing PCA & projection
    pca = PCA(n_components=n_components, svd_solver='full') # full forces the lapack solver
    principal_components = pca.fit_transform(data_trials_normalized)
    explained_variance_ratio = pca.explained_variance_ratio_ # eigenvalues
    return pca, principal_components, explained_variance_ratio

def filter_active_epochs (data_trials_array, response_threshold = 3, cell_threshold = 4):
    '''
    Remove epochs with no activity or only 1-2 cells active
    :param data_trials_array: numpy array of shape (n_cells, n_chunks)
    :param threshold: z-score threshold that cells need to be to be considered active
    :return:
    '''
    # check whether each cell goes above response threshold in each epoch
    # then count number of cells (per epoch) that exceed response threshold (shape n_epochs)
    cells_above_threshold = np.any(data_trials_array > response_threshold, axis=-1).sum(axis = -1)

    # valid epochs are only those with > 2 cells co-active (shape n_epochs)
    valid_epochs = cells_above_threshold >= cell_threshold

    return valid_epochs


def decay_eigenspectra(eigenvalues):
    '''
    Fit power-law in log-log space

    Power-law: y = Ax**-alpha, where x is the rank, alpha is the slope (rate of decay), A is constant (intercept in log-log space)
    - to solve for alpha > take log of both sides
    - > log(y) = log(Ax**-alpha)
    - > log(y) = log(A) + log(x**-alpha)
    - > log(y) = log(A) -alpha*log(x)
    - > log(y) ~= -alpha*log(x)
    - > log(y)/log(x) ~= -alpha

    :param eigenvalues: variance explained output from PCA
    :return: -slope
    '''
    x = np.arange(1, len(eigenvalues) + 1)
    log_x = np.log10(x)
    log_y = np.log10(eigenvalues + 1e-10)  # add epsilon to avoid log(0)
    slope, intercept, r_value, _, _ = linregress(log_x, log_y)
    return -slope



# def hist_osi_angle_all_animals (obj, screen = 'big', metric = 'OSI'):
#     '''
#     plots histogram distribution of OSIs/preferred orientation for each day ACROSS ALL ANIMALS
#
#     :param obj:
#     :param metric: 'OSI' or 'DSI' or 'preferred_orientation' or 'preferred_direction'
#     '''
#     list_animals = obj.dat.keys()
#     d = {k: [] for k in np.unique([a.split ('_')[1] for a in list_animals])}
#
#     for animal in list_animals:
#         group = animal.split('_')[1]
#
#         for day in obj.dat[animal].keys():
#             for recording in [r for r in obj.dat[animal][day].keys() if (screen in r) and ('chirps' not in r)]:
#
#                 responsive_indices = obj.dat[animal][day][recording]['thresholded_cells']
#                 metric_dat = obj.dat[animal][day][recording][metric][responsive_indices]  # shape n_cells
#                 # metric_dat = obj.dat[animal][day][recording][metric]#[responsive_indices]  # shape n_cells
#
#                 d[group].append(metric_dat)
#
#     for group in d.keys():
#         d[group] = np.concatenate(d[group])
#
#
#     colours = {'phpeb': 'black', 'GNAT': 'red'}
#     fig, ax = plt.subplots (nrows = 1, ncols = 1, figsize = (5,4), sharex=True, sharey=True)
#
#     for i_group, group in enumerate(d.keys()):
#
#             if (metric == 'preferred_orientation') or (metric == 'preferred_direction'):
#                 bins = np.linspace(-180,180,8)
#             elif metric == 'OSI' or metric == 'DSI':
#                 bins = np.linspace(0, 1, 20)
#
#             weights = np.ones_like(d[group]) * (100.0 / d[group].size)
#             ax.hist(
#                 d[group], bins=bins, weights=weights,
#                 density=False, color=colours.get(group, 'gray'),
#                 label = group, histtype='step', alpha=0.8, linewidth=3)
#
#     #
#     # ax.set_ylabel(
#     #     rf'$\bf{{{group}}}$' + '\n % cells',
#     #     fontsize=14)
#     ax.set_xlabel(metric)
#     ax.set_title(f'{group}')
#     ax.legend(frameon=False)
#
#     # font_kwargs = dict(fontsize="large")
#     # add_headers(fig, col_headers=["Preferred Orientation", "Orientation Selectivity Index"], row_headers=['Cell count \n (' + day + ')' for day in list(obj.dat_subject.keys())], **font_kwargs)
#     plt.suptitle(metric)
#     # plt.tight_layout()
#     # folder_path = os.path.join(os.path.dirname(obj.path), 'figures', 'population_all_animals','histograms')
#     # if not os.path.exists(folder_path):
#     #     os.makedirs(folder_path)
#     # plt.savefig(os.path.join(folder_path, f'hist_osi_across_days_{metric}.svg'))
#     # plt.savefig(os.path.join(folder_path, f'hist_osi_across_days_{metric}.png'))
#
#     plt.show()

def hist_osi_angle_all_animals(obj, screen='big', metric='OSI'):
    """
    Plot histogram distribution of OSI/DSI/preferred angle across all animals.

    For OSI/DSI:
        - Mann-Whitney U test
        - Kolmogorov-Smirnov test

    For preferred_orientation/preferred_direction:
        - Circular mean is printed
        - KS test is run on wrapped values as a rough descriptive comparison
    """

    from scipy.stats import mannwhitneyu, ks_2samp, circmean

    list_animals = obj.dat.keys()
    d = {k: [] for k in np.unique([a.split('_')[1] for a in list_animals])}

    for animal in list_animals:
        group = animal.split('_')[1]

        for day in obj.dat[animal].keys():
            recordings = [
                r for r in obj.dat[animal][day].keys()
                if (screen in r) and ('chirps' not in r)
            ]

            for recording in recordings:
                responsive_indices = obj.dat[animal][day][recording]['thresholded_cells']
                metric_dat = obj.dat[animal][day][recording][metric][responsive_indices]
                d[group].append(metric_dat)

    for group in d.keys():
        d[group] = np.concatenate(d[group])
        d[group] = d[group][~np.isnan(d[group])]

    groups = list(d.keys())

    stats_text = ""

    if len(groups) == 2:
        g1, g2 = groups
        x = d[g1]
        y = d[g2]

        print(f"\n{g1} vs {g2}")
        print(f"n {g1}: {len(x)} cells")
        print(f"n {g2}: {len(y)} cells")

        if metric in ['OSI', 'DSI']:
            mw_stat, mw_p = mannwhitneyu(x, y, alternative='two-sided')
            ks_stat, ks_p = ks_2samp(x, y)

            print(f"Mann-Whitney U: statistic = {mw_stat:.3f}, p = {mw_p:.3e}")
            print(f"Kolmogorov-Smirnov: statistic = {ks_stat:.3f}, p = {ks_p:.3e}")

            stats_text = (
                f"{g1} vs {g2}\n"
                f"MW p = {mw_p:.2e}\n"
                f"KS p = {ks_p:.2e}"
            )

        elif metric in ['preferred_orientation', 'preferred_direction']:
            x_wrapped = ((x + 180) % 360) - 180
            y_wrapped = ((y + 180) % 360) - 180

            ks_stat, ks_p = ks_2samp(x_wrapped, y_wrapped)

            x_circ_mean = circmean(x_wrapped, high=180, low=-180)
            y_circ_mean = circmean(y_wrapped, high=180, low=-180)

            print(f"Circular mean {g1}: {x_circ_mean:.2f} degrees")
            print(f"Circular mean {g2}: {y_circ_mean:.2f} degrees")
            print(f"Wrapped KS test: statistic = {ks_stat:.3f}, p = {ks_p:.3e}")
            print("Note: KS on angles is approximate because angles are circular.")

            stats_text = (
                f"{g1} vs {g2}\n"
                f"circ mean {g1} = {x_circ_mean:.1f}°\n"
                f"circ mean {g2} = {y_circ_mean:.1f}°\n"
                f"KS p = {ks_p:.2e}"
            )

    else:
        print("Stats currently only implemented for exactly 2 groups.")

    colours = {'phpeb': 'black', 'GNAT': 'red'}
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))

    if metric in ['preferred_orientation', 'preferred_direction']:
        bins = np.linspace(-180, 180, 9)
    elif metric in ['OSI', 'DSI']:
        bins = np.linspace(0, 1, 20)
    else:
        bins = 20

    for group in groups:
        weights = np.ones_like(d[group]) * (100.0 / d[group].size)

        ax.hist(
            d[group],
            bins=bins,
            weights=weights,
            density=False,
            color=colours.get(group, 'gray'),
            label=f"{group} n={len(d[group])}",
            histtype='step',
            alpha=0.8,
            linewidth=3
        )

    ax.set_ylabel('% cells', fontsize=14)
    ax.set_xlabel(metric, fontsize=14)
    ax.legend(frameon=False)

    if stats_text:
        ax.text(
            0.05, 0.95,
            stats_text,
            transform=ax.transAxes,
            va='top',
            ha='left',
            fontsize=10
        )

    plt.suptitle(metric)
    plt.tight_layout()
    plt.show()

    return d

def corr_vector (obj, screen, null_distribution = False, n = 1000):

    '''
    :param obj:
    :param animal (str):
    :param thresholded_cells: either 0 (take all cells for the analysis), 1 (cells that pass treshold at least once), 2 (cells that pass threshold on all days)
    :param null_distribution (bool):
    :param n (int):
    :param across_days: True if we want to get 1 correlation value across all days(average the matrix). False if we want to compare separate days
    :return: correlation matrix, shape (n_days, n_days, n_cells)

    NB: deconvolved must be true
    '''

    list_animals = obj.dat.keys()

    groups = np.unique([a.split('_')[1] for a in list_animals])

    # store recordings in lists, temporarily
    d = {k: [] for k in groups}

    # this block turns d into a dictionary with keys 'groups' (GNAT or phpeb), where each value is a list with n_animals entries, each with shape (n_repeats, n_theta, n_cells, n_timepoints)
    for animal in list_animals:
        group = animal.split('_')[1]
        for day in obj.dat[animal].keys():

            for recording in [r for r in obj.dat[animal][day].keys() if (screen in r) and ('chirps' not in r)]:

                responsive_indices = obj.dat[animal][day][recording]['thresholded_cells']

                if responsive_indices.sum() > 0: # if there is at least 1 cell that passes threshold > store it
                    # shape: (n_repeats, n_theta, n_cells, n_timepoints)
                    metric_dat = obj.dat[animal][day][recording][
                                     'zscored_matrix_baseline'
                                 ][:,:, responsive_indices, :]
                    d[group].append(metric_dat)

    # concatenate all data in d > so that d dictionary stores, for each group, the response matrix of shape (n_repeats, n_theta, n_cells, n_timepoints)
    for group in d.keys():

        if len(d[group]) == 0:
            d[group] = np.array([])
            continue

        # find minimum number of timepoints
        min_t = min(arr.shape[-1] for arr in d[group])

        # crop all recordings to min time
        cropped = [
            arr[..., :min_t]
            for arr in d[group]
        ]

        # concatenate across cell axis
        d[group] = np.concatenate(cropped, axis=2)

        print(group, d[group].shape)

    groups = list(set([ani.split('_')[1] for ani in obj.dat.keys()]))
    corr_values = {k: [] for k in groups}

    for group in groups:

        n_cells = d[group].shape[2]
        corr_values[group] = np.zeros(n_cells) # 1 correlation value per cell

        for i_cell, cell_num in enumerate(tqdm(np.arange(n_cells), desc = 'Calculating correlation distribution for each cell')):
            correlation_matrix = np.zeros(n_cells)

            param_data = d[group]  # (n_repeats, n_orientations, n_cells, n_timepoints)

            # comparing the same day, want to do a split-half reliability metrix
            if null_distribution:

                # Split data into first half and second half along the first axis (repeats/trials)
                half = param_data.shape[0] // 2
                vector_first_half = param_data[:half, :, cell_num, :].mean(axis=(0, -1))  # (n_orientations,)

                null_dist = np.zeros(n)
                # for the null distribution: shuffle cells across days (or just pick random cells)
                for i_n in range(n):
                    vector_second_half = param_data[half:, :, np.random.choice(np.arange(n_cells)), :].mean(axis=(0, -1))  # (n_orientations,)

                    # Compute Pearson correlation between halves
                    corr_coef, _ = pearsonr(vector_first_half, vector_second_half)
                    null_dist[i_n] = corr_coef

                corr_values[group][i_cell] = null_dist.mean()

            else:
                # Split data into first half and second half along the first axis (repeats/trials)
                half = param_data.shape[0] // 2
                vector_first_half = param_data[:half, :, cell_num, :].mean(axis=(0, -1))  # (n_orientations,)
                vector_second_half = param_data[half:, :, cell_num, :].mean(axis=(0, -1))  # (n_orientations,)

                # Compute Pearson correlation between halves
                corr_coef, _ = pearsonr(vector_first_half, vector_second_half)

                corr_values[group][i_cell] = corr_coef


    return corr_values

def plot_corr_vector (obj, screen):

    groups = list(set([ani.split('_')[1] for ani in obj.dat.keys()]))

    corr_vec_aligned = corr_vector(obj, screen, null_distribution=False)
    corr_vec_shuffled = corr_vector(obj, screen, null_distribution=True, n=100)

    fig, ax = plt.subplots (1, ncols = len(groups), figsize = (8,4), sharey = True)

    for i_group, group in enumerate(groups):
        vec = corr_vec_aligned[group]
        vec_null = corr_vec_shuffled[group]

        bin_edges = np.linspace(min(vec.min(), vec_null.min()), max(vec.max(), vec_null.max()), 10)

        ax[i_group].hist(vec, color='lightsalmon', alpha=0.8, bins=bin_edges, label='Aligned')
        ax[i_group].hist(vec_null, color='gray', alpha=0.8, bins=bin_edges, label='Shuffled')
        ax[i_group].set_title (group)
        ax[i_group].set_xlabel ('Pearson (r) correlation', fontsize = 14)
        ax[i_group].set_ylabel('Cell count', fontsize=14)
        plt.ylim([0, 100])

    plt.suptitle(f'Correlation distribution', fontsize = 15)
    # plt.tick_params(axis='both', labelsize=12)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()


# def response_amplitude_fff (object, animal, day, session):
#     # dont' use z score responses becuase responses need to be positive
#     # grab portion of the response during the moving grating (skip first second) (shape n_repeats, n_ori, n_cells, ntimepoints), mean across repeats > (shape n_ori, n_cells, ntimepoints)
#     arr = object.dat[animal][day][session]['dfof_matrix_baseline'][:,:, object.fps*1:object.fps*4].mean(axis = 0)
#
#     # this is the amplitude of the response (max peak - min peak)
#     amplitude = arr.max(axis = (0,-1)) # max across all orientations & timepoints shape n_cells
#
#     return amplitude

def max_response_amplitude (object, animal, day, session):
    # dont' use z score responses becuase responses need to be positive
    # grab portion of the response during the moving grating (skip first second) (shape n_repeats, n_ori, n_cells, ntimepoints), mean across repeats > (shape n_ori, n_cells, ntimepoints)
    arr = object.dat[animal][day][session]['dfof_matrix_baseline'][:,:, :,object.fps*1:object.fps*4].mean(axis = 0)

    # this is the amplitude of the response (max peak - min peak)
    amplitude = arr.max(axis = (0,-1)) # max across all orientations & timepoints shape n_cells

    return amplitude

def mean_response_amplitude (object, animal, day, session):
    # dont' use z score responses becuase responses need to be positive
    # grab portion of the response during the moving grating (skip first second) (shape n_repeats, n_ori, n_cells, ntimepoints), mean across repeats > (shape n_ori, n_cells, ntimepoints)
    arr = object.dat[animal][day][session]['dfof_matrix_baseline'][:,:, :, object.fps*1:object.fps*4].mean(axis = 0)

    # this is the amplitude of the response (max peak - min peak)
    amplitude = arr.max(axis = -1).mean(axis = 0) # max across all timepoints, averaged across orientations, shape n_cells

    return amplitude

from scipy.signal import find_peaks
import numpy as np

def mean_response_events(object, animal, day, session, height=None, prominence=None):
    """
    Count the number of response events (peaks) for each cell.

    Parameters
    ----------
    height : float, optional
        Minimum peak height.
    prominence : float, optional
        Minimum peak prominence.

    Returns
    -------
    events : ndarray, shape (n_cells,)
        Mean number of events across orientations.
    """

    # Average across repeats
    arr = object.dat[animal][day][session]['dfof_matrix_baseline'][:, :, :, object.fps*1:object.fps*4].mean(axis=0)   # (n_ori, n_cells, n_time)

    n_ori, n_cells, _ = arr.shape
    event_counts = np.zeros((n_ori, n_cells))

    for ori in range(n_ori):
        for cell in range(n_cells):
            peaks, _ = find_peaks(
                arr[ori, cell],
                height=height,
                prominence=prominence
            )
            event_counts[ori, cell] = len(peaks)

    # Average across orientations
    return event_counts.mean(axis=0)

#
# def hist_amplitude_all_animals (obj, screen = 'big', amp_type = 'mean'):
#     '''
#     plots histogram distribution of OSIs/preferred orientation for each day ACROSS ALL ANIMALS
#
#     :param obj:
#     :param metric: 'OSI' or 'DSI' or 'preferred_orientation' or 'preferred_direction'
#     :param amp_type: max or mean
#     '''
#     list_animals = obj.dat.keys()
#     d = {k: [] for k in np.unique([a.split ('_')[1] for a in list_animals])}
#
#     for animal in list_animals:
#         group = animal.split('_')[1]
#
#         for day in obj.dat[animal].keys():
#             for recording in [r for r in obj.dat[animal][day].keys() if (screen in r) and ('chirps' not in r)]:
#
#                 responsive_indices = obj.dat[animal][day][recording]['thresholded_cells']
#                 metric_dat = obj.dat[animal][day][recording][f'{amp_type}_amplitude'][responsive_indices]  # shape n_cells
#                 # metric_dat = obj.dat[animal][day][recording][metric]#[responsive_indices]  # shape n_cells
#
#                 d[group].append(metric_dat)
#
#     for group in d.keys():
#         d[group] = np.concatenate(d[group])
#
#
#     colours = {'phpeb': 'black', 'GNAT': 'red'}
#     fig, ax = plt.subplots (nrows = 1, ncols = 1, figsize = (5,4), sharex=True, sharey=True)
#
#     for i_group, group in enumerate(d.keys()):
#
#             bins = np.linspace(0, 1, 20)
#
#             weights = np.ones_like(d[group]) * (100.0 / d[group].size)
#             ax.hist(
#                 d[group], bins=bins, weights=weights,
#                 density=False, color=colours.get(group, 'gray'), histtype='step', alpha=0.8, linewidth=3, label = group
#             )
#
#             # ax.set_ylabel(
#             #     rf'$\bf{{{group}}}$' + '\n % cells',
#             #     fontsize=14)
#             ax.set_ylabel('\n % cells',
#                 fontsize=14)
#             ax.set_xlabel(f'{amp_type} amplitude')
#             #ax.set_title(f'{group}')
#             ax.legend()
#
#
#     # font_kwargs = dict(fontsize="large")
#     # add_headers(fig, col_headers=["Preferred Orientation", "Orientation Selectivity Index"], row_headers=['Cell count \n (' + day + ')' for day in list(obj.dat_subject.keys())], **font_kwargs)
#     plt.suptitle(f'{amp_type} amplitude')
#     # plt.tight_layout()
#     # folder_path = os.path.join(os.path.dirname(obj.path), 'figures', 'population_all_animals','histograms')
#     # if not os.path.exists(folder_path):
#     #     os.makedirs(folder_path)
#     # plt.savefig(os.path.join(folder_path, f'hist_osi_across_days_{metric}.svg'))
#     # plt.savefig(os.path.join(folder_path, f'hist_osi_across_days_{metric}.png'))
#
#     plt.show()



def hist_amplitude_all_animals(obj, screen='big', amp_type='max_amplitude'):
    """
    Plot amplitude distributions across all animals, grouped by animal group,
    and compare distributions statistically.

    amp_type: max_amplitude, mean_amplitude

    Stats:
    - Mann-Whitney U: tests whether one group tends to have larger values
    - Kolmogorov-Smirnov: tests whether the overall distributions differ
    """
    import numpy as np
    from scipy.stats import mannwhitneyu, ks_2samp

    list_animals = obj.dat.keys()
    d = {k: [] for k in np.unique([a.split('_')[1] for a in list_animals])}

    for animal in list_animals:
        group = animal.split('_')[1]

        for day in obj.dat[animal].keys():
            recordings = [
                r for r in obj.dat[animal][day].keys()
                if (screen in r) and ('chirps' not in r)
            ]

            for recording in recordings:
                responsive_indices = obj.dat[animal][day][recording]['thresholded_cells']

                metric_dat = obj.dat[animal][day][recording][amp_type][responsive_indices]


                d[group].append(metric_dat)

    # concatenate all cells per group
    for group in d.keys():
        d[group] = np.concatenate(d[group])
        d[group] = d[group][~np.isnan(d[group])]

    groups = list(d.keys())

    # stats
    stats_text = ""

    # if len(groups) == 2:
    #     g1, g2 = groups
    #     x = d[g1]
    #     y = d[g2]
    #
    #     mw_stat, mw_p = mannwhitneyu(x, y, alternative='two-sided')
    #     ks_stat, ks_p = ks_2samp(x, y)
    #
    #     print(f"\n{g1} vs {g2}")
    #     print(f"n {g1}: {len(x)} cells")
    #     print(f"n {g2}: {len(y)} cells")
    #     print(f"Mann-Whitney U: statistic = {mw_stat:.3f}, p = {mw_p:.3e}")
    #     print(f"Kolmogorov-Smirnov: statistic = {ks_stat:.3f}, p = {ks_p:.3e}")
    #
    #     stats_text = (
    #         f"{g1} vs {g2}\n"
    #         f"MW p = {mw_p:.2e}\n"
    #         f"KS p = {ks_p:.2e}"
    #     )
    #
    # else:
    #     print("Stats currently only implemented for exactly 2 groups.")

    ##############################
    import numpy as np
    from scipy.stats import permutation_test, ks_2samp

    if len(groups) == 2:
        g1, g2 = groups
        # each group is a 1d array of mean amplitude, 53 cells from 4 GNATs, and 268 cells from 7 controls
        x = d[g1]
        y = d[g2]

        # Test statistic: two-sample KS D = largest vertical gap between the
        # two empirical CDFs. Sensitive to any difference in distribution
        # (location, spread, shape), not just a shift in the mean.
        def ks_stat(x, y):
            return ks_2samp(x, y).statistic

        B = 100_000  # Monte Carlo permutations; exact enumeration is impossible
        # here (C(321,53) is about 1e60)

        res = permutation_test(
            (x, y),
            ks_stat,
            permutation_type='independent',  # shuffle the group labels across all 321 cells
            n_resamples=B,
            batch=1000,  # memory-friendly chunking
            alternative='greater',  # D is a distance: only large values are
            # evidence against H0
            random_state=0,  # fixes the p-value so it reproduces
        )

        p = res.pvalue  # scipy uses (#extreme + 1)/(B + 1)
        print(f"KS D = {res.statistic:.3f}")
        print(f"p = {p:.2e}   (permutation, B = {B:,})")
        if p <= 2 / (B + 1):
            print(f"-> at the Monte Carlo floor; report as p < {1 / B:.0e}")

    ###############################

    # plot
    colours = {'phpeb': 'black', 'GNAT': 'red'}
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))

    bins = np.linspace(0, 1, 20)

    for group in groups:
        weights = np.ones_like(d[group]) * (100.0 / d[group].size)

        ax.hist(
            d[group],
            bins=bins,
            weights=weights,
            density=False,
            color=colours.get(group, 'gray'),
            histtype='step',
            alpha=0.8,
            linewidth=3,
            label=f"{group} n={len(d[group])}"
        )

    ax.set_ylabel('% cells', fontsize=14)
    ax.set_xlabel(amp_type, fontsize=14)
    ax.legend(frameon=False)

    if stats_text:
        ax.text(
            0.05, 0.95,
            stats_text,
            transform=ax.transAxes,
            va='top',
            ha='left',
            fontsize=10
        )

    plt.suptitle(amp_type)
    plt.tight_layout()
    plt.show()

    return d

def hist_events_all_animals(obj, screen='big', amp_type='max_amplitude'):
    """
    Plot amplitude distributions across all animals, grouped by animal group,
    and compare distributions statistically.

    amp_type: max_amplitude, mean_amplitude, mean_events

    Stats:
    - Mann-Whitney U: tests whether one group tends to have larger values
    - Kolmogorov-Smirnov: tests whether the overall distributions differ
    """

    from scipy.stats import mannwhitneyu, ks_2samp

    list_animals = obj.dat.keys()
    d = {k: [] for k in np.unique([a.split('_')[1] for a in list_animals])}

    for animal in list_animals:
        group = animal.split('_')[1]

        for day in obj.dat[animal].keys():
            recordings = [
                r for r in obj.dat[animal][day].keys()
                if (screen in r) and ('chirps' not in r)
            ]

            for recording in recordings:
                responsive_indices = obj.dat[animal][day][recording]['thresholded_cells']

                metric_dat = obj.dat[animal][day][recording][amp_type][responsive_indices]


                d[group].append(metric_dat)

    # concatenate all cells per group
    for group in d.keys():
        d[group] = np.concatenate(d[group])
        d[group] = d[group][~np.isnan(d[group])]

    groups = list(d.keys())

    # stats
    stats_text = ""

    if len(groups) == 2:
        g1, g2 = groups
        x = d[g1]
        y = d[g2]

        mw_stat, mw_p = mannwhitneyu(x, y, alternative='two-sided')
        ks_stat, ks_p = ks_2samp(x, y)

        print(f"\n{g1} vs {g2}")
        print(f"n {g1}: {len(x)} cells")
        print(f"n {g2}: {len(y)} cells")
        print(f"Mann-Whitney U: statistic = {mw_stat:.3f}, p = {mw_p:.3e}")
        print(f"Kolmogorov-Smirnov: statistic = {ks_stat:.3f}, p = {ks_p:.3e}")

        stats_text = (
            f"{g1} vs {g2}\n"
            f"MW p = {mw_p:.2e}\n"
            f"KS p = {ks_p:.2e}"
        )

    else:
        print("Stats currently only implemented for exactly 2 groups.")

    # plot
    colours = {'phpeb': 'black', 'GNAT': 'red'}
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))

    bins = np.linspace(0, 30, 20)

    for group in groups:
        weights = np.ones_like(d[group]) * (100.0 / d[group].size)

        ax.hist(
            d[group],
            bins=bins,
            weights=weights,
            density=False,
            color=colours.get(group, 'gray'),
            histtype='step',
            alpha=0.8,
            linewidth=3,
            label=f"{group} n={len(d[group])}"
        )

    ax.set_ylabel('% cells', fontsize=14)
    ax.set_xlabel(amp_type, fontsize=14)
    ax.legend(frameon=False)

    if stats_text:
        ax.text(
            0.05, 0.95,
            stats_text,
            transform=ax.transAxes,
            va='top',
            ha='left',
            fontsize=10
        )

    plt.suptitle(amp_type)
    plt.tight_layout()
    plt.show()

    return d
