from helpers import *
from plotting import *

class DataAnalysis:

    def __init__(self, path_list, dict_animals_days,  response_type = 'deconvolved', fps = 20, zscore_threshold = 0.8, std_threshold = 0.8, show_plots = False):

        self.dict_animals_days = dict_animals_days
        self.path_list = path_list
        self.save_path = os.path.join(self.path_list[0], 'figures')
        self.show_plots = show_plots
        self.response_type = response_type
        self.fps = fps              # sampling rate of the data acquisition
        self.zscore_threshold = zscore_threshold
        self.std_threshold = std_threshold
        self.preprocessing()

    def preprocessing (self):

        self.dat = {}

        for animal in self.dict_animals_days.keys():
            self.dat[animal] = {}
            for day in tqdm(self.dict_animals_days[animal], desc = f'Processing: {animal}'):

                # if the animal / day isn't stored on the first drive, check the second
                if os.path.exists(os.path.join(self.path_list[0], 'data', animal, day)):
                    self.data_path = os.path.join(self.path_list[0], 'data')
                else:
                    self.data_path = os.path.join(self.path_list[1], 'data')

                self.dat[animal][day] = {}

                recordings = [file for file in os.listdir(os.path.join(self.data_path, animal, day)) if ((not file.endswith('.mat')) and (not file.endswith('.png')) and (not file.endswith('.tif')))]
                for recording in recordings:
                    sub_files = [file for file in os.listdir(os.path.join(self.data_path, animal, day, recording)) if ((not file.endswith('.mat')) and (not file.endswith('.png')) and (not file.endswith('.tiff')))]
                    for sub_file in sub_files: #sub_files:

                        self.dat[animal][day][sub_file] = {}

                        # load log file information
                        log_file = [file for file in os.listdir(os.path.join(self.data_path, animal, day, recording, sub_file, 'logfiles')) if file.endswith('_log.txt')][0]
                        log_path = os.path.join(self.data_path, animal, day, recording, sub_file, 'logfiles', log_file)

                        self.dat[animal][day][sub_file]['log'] = {}
                        self.dat[animal][day][sub_file]['log']['stim_dict'], self.dat[animal][day][sub_file]['log']['list_stim_types'], self.dat[animal][day][sub_file]['log']['list_stim_names'], self.dat[animal][day][sub_file]['ntheta'], self.dat[animal][day][sub_file]['nrepeats'] = load_stims (log_path)

                        # load suite2p information
                        suite2p_path = os.path.join(self.data_path, animal, day, recording, sub_file, 'experiments', 'suite2p', 'plane0')
                        responses, iscell, ops, stat = suite2p_files (suite2p_path, response_type ='fluorescence')
                        spikes, _, _, _ = suite2p_files(suite2p_path, response_type= 'deconvolved')
                        self.dat[animal][day][sub_file]['meanImg'] = ops['meanImg']

                        # load ttl / neural information
                        ttl_file = [file for file in os.listdir(os.path.join(self.data_path, animal, day, recording, sub_file, 'experiments')) if (file.endswith('.mat') and (not file.endswith('realtime.mat')))][0]
                        ttl_path = os.path.join(self.data_path, animal, day, recording, sub_file, 'experiments', ttl_file)
                        self.ttl_file = loadmat(ttl_path)
                        event_id = np.squeeze(loadmat(ttl_path)['info']['event_id'][0][0])
                        ttls = np.squeeze(loadmat(ttl_path)['info']['frame'][0][0])  # [event_id == 1]

                        # if we're showing chirps dont need to add extra missing 0 for wait period.
                        if 'chirps' not in sub_file:
                            self.dat[animal][day][sub_file]['ttl_data'] = check_ttls (self, ttls, self.dat[animal][day][sub_file]['log']['list_stim_names'])
                        else:
                            self.dat[animal][day][sub_file]['ttl_data'] = ttls

                        # we calculate this so that we can shift the behavioural data / neural data over so everything starts as of the first TTL (start of wait period)
                        start_recording, end_recording = self.dat[animal][day][sub_file]['ttl_data'][0], self.dat[animal][day][sub_file]['ttl_data'][-1]

                        # only keep ROIs that qualify as cells
                        # z-score responses > z = (x-mu)/sigma) ; where mu = mean of the responses, sigma = standard deviation of the responses
                        responses = responses[iscell == 1]                          # only take ROIs that suite2p has identified as a proper cell
                        spikes = spikes[iscell == 1]  # only take ROIs that suite2p has identified as a proper cell
                        #responses -= responses.min(axis=1, keepdims=True)           # normalize - ensure all responses are non-negative

                        self.dat[animal][day][sub_file]['responses'] = responses
                        self.dat[animal][day][sub_file]['deconvolved_responses'] = spikes
                        self.dat[animal][day][sub_file]['zscored_responses'] = zscore(responses, axis=1)

                        if 'chirp' not in sub_file:
                            parsed = np.array([parse_grating_array(
                                self.dat[animal][day][sub_file]['log']['stim_dict'][grating_block]) for grating_block in
                                      self.dat[animal][day][sub_file]['log']['stim_dict'].keys() if 'Grating' in grating_block])

                            self.dat[animal][day][sub_file]['SFs'] = parsed[:,:, 1]
                            self.dat[animal][day][sub_file]['TFs'] = parsed[:, :, 2]
                            self.dat[animal][day][sub_file]['thetas'] = parsed[:, :, 0]

                            self.dat[animal][day][sub_file]['n_SF'] = len(np.unique(self.dat[animal][day][sub_file]['SFs']))
                            self.dat[animal][day][sub_file]['n_TF'] = len(np.unique(self.dat[animal][day][sub_file]['TFs']))
                            self.dat[animal][day][sub_file]['n_theta'] = len(np.unique(self.dat[animal][day][sub_file]['thetas']))

                            # now need to fix & parse ttl data
                            get_neuronal_responses_ttls(self, self.dat[animal][day][sub_file])

                            # reshape into shape n_repeats x n_theta x n_SF x n_TF (+ squeeze gets rid of singleton dimension) x n_cells x n_timepoints
                            self.dat[animal][day][sub_file]['n_cells'] = self.dat[animal][day][sub_file]['responses'].shape[0]

                            # 8 orientations x 6 repeats x 1 SF x 1 TF
                            if (self.dat[animal][day][sub_file]['n_SF'] == 1) and (self.dat[animal][day][sub_file]['n_TF'] == 1):
                                print('8 ori x 6 repeats')

                                store_metrics(self, animal, day, sub_file)
                                self.dat[animal][day][sub_file]['max_amplitude'] = max_response_amplitude(self, animal, day, sub_file)
                                self.dat[animal][day][sub_file]['mean_amplitude'] = mean_response_amplitude(self, animal, day, sub_file)

                                self.dat[animal][day][sub_file]['mean_events'] = mean_response_events(self, animal, day, sub_file, height=None, prominence=None)
