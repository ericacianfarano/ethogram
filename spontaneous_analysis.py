
from imports import *
from classes import *
from helpers import *
from full_field_plots import *

animals_days = {'EC_GCaMP6s_06': ['20240925', '20241121'],
                'EC_GCaMP6s_09': ['20241009', '20241122'],
                'EC_RD1_08': ['20241003','20250108'],
                'EC_RD1_09': ['20241007','20250110'],
                'EC_RD1_06': ['20241002','20250108']
                }


#animals_days = {'EC_GCaMP6s_06': ['20241121']}
data_object = DataAnalysis (['E:\\vision_restored', 'I:\\vision_restored'], dict_animals_days = animals_days, response_type = 'fluorescence', dlc = False, show_plots = False)


spon = np.squeeze(data_object.dat['EC_GCaMP6s_09']['20241122']['big100_SFxTF_000_013']['zscored_responses_ttls']['Wait_0']) # shape n_repeats, n_cells, n_timepoints
#spon = np.squeeze(np.concatenate((data_object.dat['EC_GCaMP6s_09']['20241122']['big100_SFxO_000_010']['zscored_responses_ttls']['Wait_0'], data_object.dat['EC_GCaMP6s_09']['20241122']['big100_SFxO_000_010']['zscored_responses_ttls']['Wait_1']),axis = 2)) # shape n_repeats, n_cells, n_timepoints
os_cells = data_object.dat['EC_GCaMP6s_09']['20241122']['big100_SFxO_000_010']['thresholded_cells'] & (data_object.dat['EC_GCaMP6s_09']['20241122']['big100_SFxO_000_010']['OSI'] > 0.4)
pref_orientations = data_object.dat['EC_GCaMP6s_09']['20241122']['big100_SFxO_000_010']['preferred_orientation']
spon_arr = spon [os_cells] # filtered according to active cells that are OS

spon_arr = np.squeeze(data_object.dat['EC_GCaMP6s_09']['20241122']['big100_SFxTF_000_011']['zscored_responses_ttls']['Wait_1']) # shape n_repeats, n_cells, n_timepoints

def compute_coactivation_matrix(data_trials_all, valid_epochs, response_threshold):
    '''
    :param data_trials_all:
    :param valid_epochs:
    :param response_threshold:
    :return: symmetric coactivation_matrix, where high values mean two neurons often fire together across epochs.
    '''
    n_cells = data_trials_all.shape[1]
    coactivation_matrix = np.zeros((n_cells, n_cells), dtype=int)

    for epoch in data_trials_all[valid_epochs]:  # shape: (n_cells, n_timepoints)
        active_cells = np.where(np.any(epoch > response_threshold, axis=1))[0]

        # Count self-activation
        for i in active_cells:
            coactivation_matrix[i, i] += 1

        # Count co-activation
        for i, j in combinations(active_cells, 2):
            coactivation_matrix[i, j] += 1
            coactivation_matrix[j, i] += 1

    return coactivation_matrix

# evoked = [data_object.dat['EC_GCaMP6s_09']['20241122']['big100_SFxO_000_010']['zscored_responses'],
#             data_object.dat['EC_GCaMP6s_06']['20240925']['big100_SFxO_000_001']['zscored_responses'],
#             data_object.dat['EC_RD1_10']['20250110']['big100_SFxO_000_007']['zscored_responses'],
#             data_object.dat['EC_RD1_09']['20250110']['big100_SFxO_000_005']['zscored_responses']
#           ]
#
# spon = [np.squeeze(data_object.dat['EC_GCaMP6s_09']['20241122']['big100_SFxTF_000_011']['zscored_responses_ttls']['Wait_1']),
#         np.squeeze(data_object.dat['EC_GCaMP6s_06']['20240925']['big100_SFxTF_000_002']['zscored_responses_ttls']['Wait_1']),
#         np.squeeze(data_object.dat['EC_RD1_10']['20250110']['big100_SFxTF_000_008']['zscored_responses_ttls']['Wait_1']),
#         np.squeeze(data_object.dat['EC_RD1_09']['20250110']['big100_SFxTF_000_006']['zscored_responses'])]

evoked = [data_object.dat['EC_GCaMP6s_09']['20241009']['big100_SFxO_000_000']['zscored_responses'],
            data_object.dat['EC_GCaMP6s_09']['20241122']['big100_SFxO_000_010']['zscored_responses'],
            data_object.dat['EC_RD1_06']['20241002']['big100_SFxO_000_000']['zscored_responses'],
            data_object.dat['EC_RD1_06']['20250108']['big100_SFxO_000_004']['zscored_responses']
          ]

spon = [np.squeeze(data_object.dat['EC_GCaMP6s_09']['20241009']['big100_SFxTF_000_001']['zscored_responses_ttls']['Wait_1']),
        np.squeeze(data_object.dat['EC_GCaMP6s_09']['20241122']['big100_SFxTF_000_011']['zscored_responses_ttls']['Wait_1']),
        np.squeeze(data_object.dat['EC_RD1_06']['20241002']['big100_SFxTF_000_000']['zscored_responses_ttls']['Wait_1']),
        np.squeeze(data_object.dat['EC_RD1_06']['20250108']['big100_SFxTF_000_005']['zscored_responses'])]


# spon_arr = np.squeeze(data_object.dat['EC_GCaMP6s_09']['20241122']['big100_SFxTF_000_011']['zscored_responses_ttls']['Wait_1']) # shape n_repeats, n_cells, n_timepoints
# # spon_arr  = np.squeeze(data_object.dat['EC_GCaMP6s_06']['20240925']['big100_SFxTF_000_002']['zscored_responses_ttls']['Wait_1'])
# spon_arr  = np.squeeze(data_object.dat['EC_RD1_10']['20250110']['big100_SFxTF_000_008']['zscored_responses_ttls']['Wait_1'])
# # spon_arr  = data_object.dat['EC_RD1_09']['20250110']['big100_SFxTF_000_006']['zscored_responses']

spon = True
for animal in [a for a in data_object.dat if (('RD1_06' in a) or ('RD1_08' in a) or ('RD1_09' in a) or ('GCaMP6s_09' in a))]: # and ('06' in a) or ('09' in a)]:
    for day in data_object.dat[animal]:
        for subfile in [s for s in data_object.dat[animal][day] if 'big100_SFxTF' in s]:
            print(animal, day, subfile)

            if spon:
                spon_arr = np.squeeze(data_object.dat[animal][day][subfile]['zscored_responses_ttls']['Wait_1'])
            else:
                spon_arr = data_object.dat[animal][day][subfile]['zscored_responses']

            chunk_size = 5
            n_full_chunks = spon_arr.shape[1] // chunk_size
            # shape > (n_trials, n_features, n_timepoints_per_epoch) (n_epochs, n_cells, n_timepoints)

            data_trials_all = np.array([spon_arr[:, i:i + chunk_size] for i in range(0, n_full_chunks * chunk_size, chunk_size)])
            #dominant_epochs = get_dominant_orientation_epochs(data_trials_all, pref_orientations)

            cell_threshold = 5 #5
            response_threshold = 3 #3
            # only take epochs that have at least 'cell_threshold' co-active cells that each go above 'response_threshold' > (n_epochs, n_cells, n_timepoints)
            valid_epochs = filter_active_epochs(data_trials_all, response_threshold=response_threshold, cell_threshold=cell_threshold)

            # cell_activity_counts = np.sum([np.any(epoch > response_threshold, axis=1) for epoch in data_trials_all[valid_epochs]], axis=0)
            # active_cells_mask = cell_activity_counts > 5  # only keep neurons active in ≥ 5 epochs
            # data_trials_all = data_trials_all[:, active_cells_mask, :]

            coactivation_matrix = compute_coactivation_matrix(data_trials_all, valid_epochs, response_threshold)

            coactivation_norm = normalize(coactivation_matrix, axis=1)  # L2-normalized per row

            # sns.clustermap(coactivation_matrix, method='ward', cmap='viridis')
            # plt.title('Neuron Co-activation Matrix')
            # plt.show()

            coactivation_pca = PCA(n_components=50).fit_transform(coactivation_norm)

            k = 5
            #plt.scatter(X_umap[:, 0], X_umap[:, 1], c=cluster_labels, cmap='tab10', s=10)
            #plt.title("UMAP Clustering in UMAP Space")
            #reducer = umap.UMAP(metric='cosine',  random_state=42)
            #X_umap = reducer.fit_transform(coactivation_matrix)
            X_umap = umap.UMAP(metric='cosine', random_state=42).fit_transform(coactivation_pca)
            cluster_labels = KMeans(n_clusters=k).fit_predict(X_umap)
            score = silhouette_score(X_umap, cluster_labels)
            plt.figure(figsize=(6, 6))
            norm = Normalize(vmin=0, vmax=cluster_labels.max())
            cmap = cm.get_cmap('plasma', 256)
            new_cmap = cmap(np.linspace(0.0, 0.85, 256))  # Avoid bright yellow tail
            trimmed_cmap = ListedColormap(new_cmap)
            plt.scatter(X_umap[:, 0], X_umap[:, 1], c=cluster_labels, cmap=trimmed_cmap, s=10)
            plt.title(f" Co-activation UMAP, {animal} {day}, {spon*'spon'}")
            #plt.ylim([0,11])
            #plt.xlim([3, 20])
            plt.show()

            import seaborn as sns
            import scipy.cluster.hierarchy as sch
            from scipy.spatial.distance import pdist, squareform

            # # Step 1: Compute the distance matrix
            # dists = pdist(coactivation_matrix)  # or coactivation_norm
            # linkage_matrix = sch.linkage(dists, method='ward')
            # # Step 2: Get the dendrogram reordering
            # dendro = sch.dendrogram(linkage_matrix, no_plot=True)
            # idx_order = dendro['leaves']  # list of reordered indices
            # # Step 3: Apply that order to your matrix
            # sorted_matrix = coactivation_matrix[np.ix_(idx_order, idx_order)]
            # # Step 4 (optional): Plot it manually
            # import matplotlib.pyplot as plt
            # plt.imshow(sorted_matrix, cmap='viridis')
            # plt.title("Coactivation Matrix (Cluster-Ordered)")
            # plt.colorbar()
            # plt.show()
            # You can reorder cluster labels, coordinates, etc.
            # X_umap_ordered = X_umap[idx_order]
            # cluster_labels_ordered = cluster_labels[idx_order]

            # Z = linkage(coactivation_matrix, method='ward')
            # cluster_labels = fcluster(Z, t=5, criterion='maxclust')  # change t=5 as needed



plot_raster(data_trials_all, valid_epochs, n_trials_to_plot=60, n_cells_to_plot=400, threshold=20)

# then average over each epoch (time) to get average response> (n_epochs, n_cells)
data_trials = data_trials_all [valid_epochs].mean(axis=-1)

print(f'{animal}, {day}, {data_trials.shape[0] / data_trials_all.shape[0] * 100}% of trials have >4 coactive cells')

# Normalize each trial's activity pattern (vector) to unit length (L2 norm)
# each trial (each row) is normalized to unit length (its L2 norm is 1)
norms = np.linalg.norm(data_trials, axis=1, keepdims=True)
data_trials_normalized = data_trials / norms

pca, principal_components, explained_variance_ratio = perform_pca (data_trials_normalized.T, n_components = 2)

plot_pca_projection('EC_GCaMP6s_09', principal_components, 5, 8, trial_type = 'spontaneous')


#######################################


def plot_slope_eigenvals(power_law_slopes, timepoints=False):
    plt.figure(figsize=(5, 4))

    if timepoints:
        colors = {'Control_0': 'cornflowerblue', 'Control_1': 'darkblue', 'RD1_0': 'salmon', 'RD1_1': 'firebrick'}
    else:
        colors = {'Control': 'black', 'RD1': 'red', 'GNAT': 'green'}

    group_names = list(power_law_slopes.keys())
    x_positions = {group: i+1 for i, group in enumerate(group_names)}  # map group name to x
    group_data = {group: np.array(vals) for group, vals in power_law_slopes.items()}
    jittered_x = {}

    # Plot scatter points and means
    for group in group_names:
        x_pos = x_positions[group]
        slopes = group_data[group]
        x_jittered = np.full_like(slopes, x_pos, dtype=float) + np.random.normal(0, 0.05, size=slopes.shape)
        jittered_x[group] = x_jittered

        plt.scatter(x_jittered, slopes, color=colors.get(group, 'gray'), alpha=0.4)
        plt.plot([x_pos], [np.mean(slopes)], marker='d', markersize=10, color=colors.get(group, 'gray'), label=group)

    # Connect paired animals across timepoints
    if timepoints:
        for base_group in ['Control', 'RD1']:
            g0, g1 = f'{base_group}_0', f'{base_group}_1'
            if g0 in group_data and g1 in group_data and len(group_data[g0]) == len(group_data[g1]):
                for i in range(len(group_data[g0])):
                    plt.plot([jittered_x[g0][i], jittered_x[g1][i]],
                             [group_data[g0][i], group_data[g1][i]],
                             color=colors.get(g0, 'gray'), alpha=0.3, linewidth=1)

    # Mann–Whitney U tests with Bonferroni correction
    comparisons = list(combinations(group_names, 2))
    p_values = []
    for g1, g2 in comparisons:
        stat, p = mannwhitneyu(group_data[g1], group_data[g2], alternative='two-sided')
        p_values.append(p)

    corrected_pvals = [min(p * len(p_values), 1.0) for p in p_values]
    print("Corrected p-values:", corrected_pvals)

    for idx, ((g1, g2), p_corr) in enumerate(zip(comparisons, corrected_pvals)):
        if p_corr < 0.05:
            i1, i2 = x_positions[g1], x_positions[g2]
            y_max = max(group_data[g1].max(), group_data[g2].max())
            y = y_max + 0.05 + 0.05 * idx
            plt.plot([i1, i1, i2, i2], [y - 0.01, y, y, y - 0.01], color='black')
            plt.text((i1 + i2) / 2, y + 0.005, '*', ha='center', fontsize=14)

    plt.xticks(list(x_positions.values()), list(x_positions.keys()))
    plt.ylabel('Power-law slope (eigenspectrum)')
    plt.title('Slope of the Eigenspectrum')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.legend()
    plt.show()



def variance_explained(var_explained_dict, log=False, timepoints = False):
    """
    Plot variance explained of different groups in 'var_explained_dict'.
    If log=True, plot log10(PC index) vs log10(variance explained).
    """

    plt.figure(figsize=(8, 5))
    if timepoints:
        colors = {'Control_0': 'cornflowerblue', 'Control_1': 'darkblue', 'RD1_0': 'salmon', 'RD1_1': 'firebrick'}
        x = np.arange(1, var_explained_dict['Control_0'].shape[1] + 1)
    else:
        colors = {'Control': 'black', 'RD1': 'red', 'GNAT': 'green'}
        x = np.arange(1, var_explained_dict['Control'].shape[1] + 1)
    x_plot = np.log10(x) if log else x

    for group, data in var_explained_dict.items():
        data = np.array(data)
        if log:
            data = np.log10(data + 1e-10)
        mean = data.mean(axis=0)
        error = sem(data, axis=0)

        c = colors.get(group, 'gray')
        plt.plot(x_plot, data.T, alpha=0.3, c=c)
        plt.plot(x_plot, mean, alpha=0.8, c=c, linewidth=2.5, label=group)
        plt.fill_between(x_plot, mean - error, mean + error, color=c, alpha=0.2)

    plt.legend(fontsize=14)
    plt.xlabel('log10(PC index)' if log else 'PC index')
    plt.ylabel('log10(variance explained)' if log else 'Variance explained')
    plt.title('Log-Log Variance Decay' if log else 'Variance explained by PCs')
    #plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def get_dominant_orientation_epochs(data_trials_all, pref_orientations, response_threshold=3, cell_threshold=4, dominance_fraction=0.5, band_deg=30):
    '''
    Returns a boolean array indicating whether each epoch has >X% of active cells with similar orientation preferences.

    Parameters:
    - data_trials_all: shape (n_epochs, n_cells, n_timepoints)
    - pref_orientations: shape (n_cells,) in degrees
    - response_threshold: z-score threshold to consider a cell active
    - cell_threshold: min number of active cells in an epoch to be considered
    - dominance_fraction: fraction of active cells that must cluster near the circular mean
    - band_deg: angular window (in degrees) for defining orientation similarity

    Returns:
    - dominant_mask: shape (n_epochs,), dtype=bool
    '''

    n_epochs = data_trials_all.shape[0]
    dominant_mask = np.zeros(n_epochs, dtype=bool)
    band_rad = np.deg2rad(band_deg)

    for epoch_i in range(n_epochs):
        epoch = data_trials_all[epoch_i]  # shape: (n_cells, n_timepoints)

        # Determine which cells are active in this epoch
        active_mask = np.any(epoch > response_threshold, axis=-1)
        active_indices = np.where(active_mask)[0]

        if len(active_indices) < cell_threshold:
            continue  # skip this epoch

        # Get preferred orientations of active cells
        active_oris = np.deg2rad(pref_orientations[active_indices])
        mean_ori = circmean(active_oris, high=2*np.pi)

        # Angular difference from circular mean
        angular_diff = np.angle(np.exp(1j * (active_oris - mean_ori)))
        within_band = np.abs(angular_diff) < band_rad

        # Check if enough cells fall within the band
        if np.sum(within_band) / len(active_oris) >= dominance_fraction:
            dominant_mask[epoch_i] = True

    return dominant_mask



animals_days = {'EC_GCaMP6s_05': ['20240925'],
                'EC_GCaMP6s_06': ['20241121'],
                'EC_GCaMP6s_08': ['20240927'], #0927
                'EC_GCaMP6s_09': ['20241122'],
                'EC_RD1_05': ['20241122'],  # '20240927', '20241116',
                'EC_RD1_06': ['20250108'], #['20241002', '20241116']
                #'EC_RD1_07': ['20241002','20241003','20241116'],
                'EC_RD1_08': ['20250108'], #['20241003','20241116'],
                'EC_RD1_09': ['20250110'], #['20241007','20241113'],
                'EC_RD1_10': ['20250110'], #['20241007','20241113'],
                'EC_GNAT_03': ['20240924'],
                'EC_GNAT_05': ['20240924'],
                'EC_GNAT_06': ['20240924'],
                'EC_GNAT_04': ['20241004'],
                }

#animals_days = {'EC_GCaMP6s_06': ['20241121']}
data_object = DataAnalysis (['E:\\vision_restored', 'I:\\vision_restored'], dict_animals_days = animals_days, response_type = 'fluorescence', dlc = False, show_plots = False)

coactive_epochs = True
chunk_size = 3 # n frames
var_explained = {'Control': [], 'RD1':[]}
eigenvals_slope = {'Control': [], 'RD1':[]}
corr_values = {'Control': [], 'RD1':[]}
# var_explained = {'Control': [], 'RD1':[], 'GNAT':[]}
# eigenvals_slope = {'Control': [], 'RD1':[], 'GNAT':[]}
for i_animal, animal in enumerate([animal for animal in animals_days.keys() if 'GNAT' not in animal]):

    group = 'Control' if 'GCaMP6s' in animal else animal.split('_')[1]

    # list of tuples, with each entry being (day, subfile)
    if group != 'GNAT':
        days_recordings = [(day, subfile) for day in animals_days[animal] for subfile in data_object.dat[animal][day].keys() if ('SFxTF' in subfile) and ('big' in subfile)][:2]
    else:
        days_recordings = [(day, subfile) for day in animals_days[animal] for subfile in
                           data_object.dat[animal][day].keys() if ('big' in subfile)][:2]

    for i, (day, subfile) in enumerate(days_recordings):

        # shape n_Cells, n_timepoints
        spon_arr = np.squeeze(data_object.dat[animal][day][subfile]['zscored_responses_ttls']['Wait_1'])

        if coactive_epochs:
            n_full_chunks = spon_arr.shape[1] // chunk_size

            # shape > (n_trials, n_features, n_timepoints_per_epoch) (n_epochs, n_cells, n_timepoints)
            #data_trials = np.array([spon_arr[:,i:i + chunk_size] for i in range(0, n_full_chunks * chunk_size, chunk_size)]).reshape (n_full_chunks, -1)
            data_trials_all = np.array([spon_arr[:, i:i + chunk_size] for i in range(0, n_full_chunks * chunk_size, chunk_size)])

            # only take epochs that have at least 'cell_threshold' co-active cells that each go above 'response_threshold' > (n_epochs, n_cells, n_timepoints)
            valid_epochs = filter_active_epochs(data_trials_all, response_threshold=3, cell_threshold = 5)
            #plot_raster(data_trials_all, valid_epochs, n_trials_to_plot=60, n_cells_to_plot=400, threshold=4)

            # then average over each epoch (time) to get average response> (n_epochs, n_cells)
            data_trials = data_trials_all[valid_epochs].mean (axis = -1)

        else:
            data_trials = spon_arr.T
        # Normalize each trial's activity pattern (vector) to unit length (L2 norm)
        # each trial (each row) is normalized to unit length (its L2 norm is 1)
        norms = np.linalg.norm(data_trials, axis=1, keepdims=True)
        data_trials_normalized = data_trials / norms

        # performing PCA & projection
        # explained variance ratio > eigenvalues
        pca, principal_components, explained_variance_ratio = perform_pca (data_trials_normalized, n_components = 50)

        var_explained[group].append(explained_variance_ratio)
        eigenvals_slope[group].append(decay_eigenspectra(explained_variance_ratio))

        corr_matrix = np.corrcoef(data_trials.T)
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(corr_matrix)
        plt.figure(figsize=(5, 5))
        colors = {'Control': 'black', 'RD1': 'red', 'GNAT': 'green'}
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha = 0.6, s = 15,color = colors[group])
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.title(f'PCA of pearson correlation {group}')
        plt.show()

for group in var_explained:
    var_explained[group] = np.array(var_explained[group])
    eigenvals_slope[group] = np.array(eigenvals_slope[group])
    #corr_matrix[group] = np.array(corr_matrix[group])

variance_explained (var_explained, log = True)
plot_slope_eigenvals(eigenvals_slope)


####################### extra analysis: looking at event rate
# def compute_event_rates_per_cell(data, z_threshold=3, fps=data_object.fps):
#     """
#     Compute spontaneous event rates (Hz) for each cell from z-scored traces.
#
#     Parameters:
#     - data: array of shape (n_cells, n_timepoints)
#     - z_threshold: z-score threshold for peak detection
#     - fps: imaging frame rate (frames per second)
#
#     Returns:
#     - rates: array of shape (n_cells,) with event rates in Hz
#     """
#     n_cells, n_timepoints = data.shape
#     duration = n_timepoints / fps  # in seconds
#     rates = np.zeros(n_cells)
#
#     for cell_idx in range(n_cells):
#         trace = data[cell_idx, :]
#         peaks, _ = find_peaks(trace, height=z_threshold)
#         rates[cell_idx] = len(peaks) / duration
#
#     return rates

def compute_event_features_per_cell(data, z_threshold=3, fps=10):
    """
    Compute spontaneous event rates, amplitudes, and durations from z-scored traces.

    Parameters:
    - data: array of shape (n_cells, n_timepoints)
    - z_threshold: threshold to detect peaks
    - fps: imaging frame rate (frames per second)

    Returns:
    - rates: array of shape (n_cells,) with event rates in Hz
    - all_amplitudes: list of all peak amplitudes across cells
    - all_durations: list of all event durations (in seconds) across cells
    """
    n_cells, n_timepoints = data.shape
    duration = n_timepoints / fps  # in seconds
    rates = np.zeros(n_cells)
    all_amplitudes = []
    all_durations = []

    for cell_idx in range(n_cells):
        trace = data[cell_idx, :]
        peaks, properties = find_peaks(trace, height=z_threshold)
        rates[cell_idx] = len(peaks) / duration

        # Collect amplitudes (z-scored peak heights)
        all_amplitudes.extend(properties["peak_heights"])

        # Collect durations (half-max width in seconds)
        widths = peak_widths(trace, peaks, rel_height=0.5)[0] / fps
        all_durations.extend(widths)

    return rates, all_amplitudes, all_durations


event_rates = {'Control': [], 'RD1':[]}
event_amplitudes = {'Control': [], 'RD1':[]}
event_durations = {'Control': [], 'RD1':[]}
dominant_frequencies_per_cell = {'Control': [], 'RD1':[]}
power_spectrum = {'Control': [], 'RD1':[]}

coactive_counts = {'Control': [], 'RD1':[]}
z_thresh = 3
for i_animal, animal in enumerate([animal for animal in animals_days.keys() if 'GNAT' not in animal]):
    group = 'Control' if 'GCaMP6s' in animal else animal.split('_')[1]

    # list of tuples, with each entry being (day, subfile)
    if group != 'GNAT':
        days_recordings = [(day, subfile) for day in animals_days[animal] for subfile in data_object.dat[animal][day].keys() if ('SFxTF' in subfile) and ('big' in subfile)][:2]
    else:
        days_recordings = [(day, subfile) for day in animals_days[animal] for subfile in
                           data_object.dat[animal][day].keys() if ('big' in subfile)][:2]

    for i, (day, subfile) in enumerate(days_recordings):
        spon_arr = np.squeeze(data_object.dat[animal][day][subfile]['zscored_responses_ttls']['Wait_1']) # shape n_Cells, n_timepoints
        #event_rates[group].append(compute_event_rates_per_cell(spon_arr, z_threshold=z_thresh))

        rates, all_amplitudes, all_durations = compute_event_features_per_cell(spon_arr, z_threshold=z_thresh)
        event_rates[group].append(rates)
        event_amplitudes[group].append(all_amplitudes)
        event_durations[group].append(all_durations)

        active = spon_arr > z_thresh  # shape (n_cells, n_timepoints) > binarized
        coactive_counts[group].append(active.sum(axis=0))  # number of active cells at each timepoint

        # fourier
        n_cells, n_timepoints = spon_arr.shape
        sampling_rate = data_object.fps
        # Compute FFT
        fft_vals = np.fft.rfft(spon_arr, axis=1)  # Only positive frequencies
        fft_freqs = np.fft.rfftfreq(n_timepoints, d=1 / sampling_rate)
        # Compute power spectrum
        power = np.abs(fft_vals) ** 2  # shape: (n_cells, n_freqs)
        # Get dominant frequency per cell
        dominant_freqs = fft_freqs[np.argmax(power[:, 1:], axis=1) + 1]  # skip DC (index 0)
        dominant_frequencies_per_cell[group].append(dominant_freqs)
        power_spectrum[group].append(power.mean(axis = 0))

for group in event_rates:
    event_rates[group] = np.array([item for sublist in event_rates[group] for item in sublist])
    coactive_counts[group] = np.array([item for sublist in coactive_counts[group] for item in sublist])
    event_amplitudes[group] = np.array([item for sublist in event_amplitudes[group] for item in sublist])
    event_durations[group] = np.array([item for sublist in event_durations[group] for item in sublist])
    dominant_frequencies_per_cell[group] = np.array([item for sublist in dominant_frequencies_per_cell[group] for item in sublist])
    power_spectrum[group] = np.array([item for sublist in power_spectrum[group] for item in sublist])
    #eigenvals_slope[group] = np.array(eigenvals_slope[group])

plot_spon_event_rate(event_rates, timepoints=False)
coactive_cells_per_frame(coactive_counts, timepoints=False)
plot_spon_event_properties(event_amplitudes, event_durations, timepoints=False)
plot_fourrier(dominant_frequencies_per_cell,power_spectrum, timepoints=False)

def plot_spon_event_rate(event_rates_groups, timepoints=False):
    '''
    Plotting spontaneous activity rates
    :param event_rates_groups: dictionary with group data
    :param timepoints:
    :return:
    '''
    plt.figure(figsize=(5, 4))

    if timepoints:
        colors = {'Control_0': 'cornflowerblue', 'Control_1': 'darkblue', 'RD1_0': 'salmon', 'RD1_1': 'firebrick'}
    else:
        colors = {'Control': 'black', 'RD1': 'red', 'GNAT': 'green'}

    if timepoints:
        event_rates_groups = {k: v for k, v in event_rates_groups.items() if 'RD1' in k}

    all_counts = np.concatenate([event_rates_groups[g] for g in event_rates_groups])
    bins = np.linspace(all_counts.min(), all_counts.max(), 30)  # 45 bins

    for g in event_rates_groups.keys():
        plt.hist(event_rates_groups[g], bins = bins,histtype ='step', linewidth = 2.5, density = True, label = g, color = colors[g], alpha = 1)
    plt.legend()
    plt.title('Spontaneous activity rates')
    plt.xlabel('Spontaneous Event rates (Hz)')
    plt.ylabel('Probability density')
    plt.show()

    # Optional: KS test between first two groups
    group1, group2 = list(event_rates_groups.keys())[:2]
    stat, pval = ks_2samp(event_rates_groups[group1], event_rates_groups[group2])
    print(f"KS test between {group1} and {group2}: D = {stat:.3f}, p = {pval:.3e}")

    stat, pval = mannwhitneyu(event_rates_groups[group1], event_rates_groups[group2],alternative='two-sided')
    print(f"Mann-Whitney U test: U = {stat:.3f}, p = {pval:.3e}")

def plot_spon_event_properties(amplitudes, durations, timepoints=False):
    '''
    Plotting spontaneous activity rates
    :param event_rates_groups: dictionary with group data
    :param timepoints:
    :return:
    '''
    fig, ax = plt.subplots (1,2,figsize=(10, 5))

    if timepoints:
        colors = {'Control_0': 'cornflowerblue', 'Control_1': 'darkblue', 'RD1_0': 'salmon', 'RD1_1': 'firebrick'}
    else:
        colors = {'Control': 'black', 'RD1': 'red', 'GNAT': 'green'}

    if timepoints: # only plot the rd1 data
        amplitudes = {k: v for k, v in amplitudes.items() if 'RD1' in k}
        durations = {k: v for k, v in durations.items() if 'RD1' in k}

    # amplitudes:
    all_counts = np.concatenate([amplitudes[g] for g in amplitudes])
    bins = np.linspace(all_counts.min(), 15, 30)  #all_counts.max()
    for g in amplitudes.keys():
        ax[0].hist(amplitudes[g], bins = bins,histtype ='step', linewidth = 2.5, density = True, label = g, color = colors[g], alpha = 0.7)
    ax[0].legend()
    ax[0].set_title('Spontaneous event amplitudes')
    ax[0].set_xlabel('Spontaneous Event amplitudes (z-score)')
    ax[0].set_ylabel('Probability density')
    #plt.show()
    # stats between first two groups
    group1, group2 = list(amplitudes.keys())[:2]
    stat, pval = ks_2samp(amplitudes[group1], amplitudes[group2])
    print(f"AMPLITUDE: KS test between {group1} and {group2}: D = {stat:.3f}, p = {pval:.3e}")
    stat, pval = mannwhitneyu(amplitudes[group1], amplitudes[group2],alternative='two-sided')
    print(f"AMPLITUDE: Mann-Whitney U test: U = {stat:.3f}, p = {pval:.3e}")

    # event durations:
    all_counts = np.concatenate([durations[g] for g in durations])
    bins = np.linspace(all_counts.min(), 7, 30)  # all_counts.max()
    for g in durations.keys():
        ax[1].hist(durations[g], bins = bins,histtype ='step', linewidth = 2.5, density = True, label = g, color = colors[g], alpha = 0.7)
    ax[1].legend()
    ax[1].set_title('Spontaneous event duration')
    ax[1].set_xlabel('Spontaneous Event duration')
    ax[1].set_ylabel('Probability density')

    # stats between first two groups
    group1, group2 = list(durations.keys())[:2]
    stat, pval = ks_2samp(durations[group1], durations[group2])
    print(f"DURATION: KS test between {group1} and {group2}: D = {stat:.3f}, p = {pval:.3e}")
    stat, pval = mannwhitneyu(durations[group1], durations[group2],alternative='two-sided')
    print(f"DURATION: Mann-Whitney U test: U = {stat:.3f}, p = {pval:.3e}")

    plt.show()

def plot_fourrier(dom_frequs_dict, power_spec, timepoints=False):
    '''
    Plotting spontaneous activity rates
    :param dom_frequs_dict: dictionary with group data
    :param timepoints:
    :return:
    '''
    fig, ax = plt.subplots (1,2,figsize=(10, 5))

    if timepoints:
        colors = {'Control_0': 'cornflowerblue', 'Control_1': 'darkblue', 'RD1_0': 'salmon', 'RD1_1': 'firebrick'}
    else:
        colors = {'Control': 'black', 'RD1': 'red', 'GNAT': 'green'}

    if timepoints: # only plot the rd1 data
        dom_frequs_dict = {k: v for k, v in dom_frequs_dict.items() if 'RD1' in k}
        power_spec = {k: v for k, v in power_spec.items() if 'RD1' in k}

    # dom freq:
    all_counts = np.concatenate([dom_frequs_dict[g] for g in dom_frequs_dict])
    #bins = np.linspace(all_counts.min(), 15, 30)  #all_counts.max()
    bins = np.linspace(all_counts.min(), all_counts.max(), 30)  # all_counts.max()
    for g in dom_frequs_dict.keys():
        ax[0].hist(dom_frequs_dict[g], bins = bins,histtype ='step', linewidth = 2.5, density = True, label = g, color = colors[g], alpha = 0.7)
    ax[0].legend()
    ax[0].set_title('Fourier')
    ax[0].set_xlabel('Dominant frequency per cell')
    ax[0].set_ylabel('Probability density')
    #plt.show()
    # stats between first two groups
    group1, group2 = list(dom_frequs_dict.keys())[:2]
    stat, pval = ks_2samp(dom_frequs_dict[group1], dom_frequs_dict[group2])
    print(f"dom_frequs_dict: KS test between {group1} and {group2}: D = {stat:.3f}, p = {pval:.3e}")
    stat, pval = mannwhitneyu(dom_frequs_dict[group1], dom_frequs_dict[group2],alternative='two-sided')
    print(f"dom_frequs_dict: Mann-Whitney U test: U = {stat:.3f}, p = {pval:.3e}")

    # power spectrum
    all_counts = np.concatenate([power_spec[g] for g in power_spec])
    #bins = np.linspace(all_counts.min(), 15, 30)  #all_counts.max()
    bins = np.linspace(all_counts.min(), data_object.fps/2, 30)  # all_counts.max()
    for g in power_spec.keys():
        plt.plot(fft_freqs, power.mean(axis=0))

        ax[1].hist(power_spec[g], bins = bins,histtype ='step', linewidth = 2.5, density = True, label = g, color = colors[g], alpha = 0.7)
    ax[1].legend()
    ax[1].set_title('Fourier')
    ax[1].set_xlabel('Power spectrum (av. across cells)')
    ax[1].set_ylabel('Probability density')
    #plt.show()
    # stats between first two groups
    group1, group2 = list(power_spec.keys())[:2]
    stat, pval = ks_2samp(power_spec[group1], power_spec[group2])
    print(f"dom_frequs_dict: KS test between {group1} and {group2}: D = {stat:.3f}, p = {pval:.3e}")
    stat, pval = mannwhitneyu(power_spec[group1], power_spec[group2],alternative='two-sided')
    print(f"dom_frequs_dict: Mann-Whitney U test: U = {stat:.3f}, p = {pval:.3e}")
    plt.show()

def coactive_cells_per_frame(coactive_counts_groups, timepoints=False):
    '''
    Plotting spontaneous activity rates
    :param event_rates_groups: dictionary with group data
    :param timepoints:
    :return:
    '''
    plt.figure(figsize=(5, 4))

    if timepoints:
        colors = {'Control_0': 'cornflowerblue', 'Control_1': 'darkblue', 'RD1_0': 'salmon', 'RD1_1': 'firebrick'}
    else:
        colors = {'Control': 'black', 'RD1': 'red', 'GNAT': 'green'}

    if timepoints:
        coactive_counts_groups = {k: v for k, v in coactive_counts_groups.items() if 'RD1' in k}

        all_counts = np.concatenate([coactive_counts_groups[g] for g in coactive_counts_groups])
        #all_counts = np.concatenate([coactive_counts_groups[g] for g in [g for g in coactive_counts.keys() if 'Control' not in g]])
        bins = np.linspace(all_counts.min(), all_counts.max(), 50)  # 45 bins

        for g in coactive_counts_groups.keys(): #[g for g in coactive_counts.keys() if 'Control' not in g]:
            plt.hist(coactive_counts_groups[g], bins=bins,histtype ='step', linewidth = 2.5, density=True, label=g, color=colors[g], alpha=1)
    else:
        all_counts = np.concatenate([coactive_counts_groups[g] for g in coactive_counts_groups])
        bins = np.linspace(all_counts.min(), all_counts.max(), 110)  # 45 bins

        for g in coactive_counts_groups.keys():
            plt.hist(coactive_counts_groups[g], bins = bins, histtype ='step', linewidth = 2.5,density = True, label = g, color = colors[g], alpha =1)
    plt.legend()
    plt.title('Co-active cells per frame')
    plt.xlim([0,30])
    plt.xlabel('Number of co-active cells')
    plt.ylabel('Probability density')
    plt.show()

    # Optional: KS test between first two groups
    group1, group2 = list(coactive_counts_groups.keys())[:2]
    stat, pval = ks_2samp(coactive_counts_groups[group1], coactive_counts_groups[group2])
    print(f"KS test between {group1} and {group2}: D = {stat:.3f}, p = {pval:.3e}")

    stat, pval = mannwhitneyu(coactive_counts_groups[group1], coactive_counts_groups[group2],alternative='two-sided')
    print(f"Mann-Whitney U test: U = {stat:.3f}, p = {pval:.3e}")

### SAME ANALYSIS AS ABOVE BUT LOOKING AT DIFFERENT TIME POINTS

coactive_epochs = True
chunk_size = 3 # n frames
var_explained = {'Control_0': [],'Control_1': [], 'RD1_0':[], 'RD1_1':[]}
eigenvals_slope =  {'Control_0': [],'Control_1': [], 'RD1_0':[], 'RD1_1':[]}

for i_animal, animal in enumerate(animals_days.keys()):#[animal for animal in animals_days.keys() if 'GNAT' not in animal]):
    g = 'Control' if 'GCaMP6s' in animal else animal.split('_')[1]
    days_recordings = [(day, subfile) for day in animals_days[animal] for subfile in data_object.dat[animal][day].keys() if ('SFxTF' in subfile) and ('big' in subfile)][:2]

    for i, (day, subfile) in enumerate(days_recordings):
        group = g + '_' +str(i)

        # shape n_Cells, n_timepoints
        spon_arr = np.squeeze(data_object.dat[animal][day][subfile]['zscored_responses_ttls']['Wait_1'])

        if coactive_epochs:
            n_full_chunks = spon_arr.shape[1] // chunk_size

            # shape > (n_trials, n_features, n_timepoints_per_epoch) (n_epochs, n_cells, n_timepoints)
            #data_trials = np.array([spon_arr[:,i:i + chunk_size] for i in range(0, n_full_chunks * chunk_size, chunk_size)]).reshape (n_full_chunks, -1)
            data_trials_all = np.array([spon_arr[:, i:i + chunk_size] for i in range(0, n_full_chunks * chunk_size, chunk_size)])

            # only take epochs that have at least 'cell_threshold' co-active cells that each go above 'response_threshold' > (n_epochs, n_cells, n_timepoints)
            valid_epochs = filter_active_epochs(data_trials_all, response_threshold=3, cell_threshold = 5)
            #plot_raster(data_trials_all, valid_epochs, n_trials_to_plot=60, n_cells_to_plot=400, threshold=4)

            # then average over each epoch (time) to get average response> (n_epochs, n_cells)
            data_trials = data_trials_all[valid_epochs].mean (axis = -1)

        else:
            data_trials = spon_arr.T
        # Normalize each trial's activity pattern (vector) to unit length (L2 norm)
        # each trial (each row) is normalized to unit length (its L2 norm is 1)
        norms = np.linalg.norm(data_trials, axis=1, keepdims=True)
        data_trials_normalized = data_trials / norms

        # performing PCA & projection
        # explained variance ratio > eigenvalues
        pca, principal_components, explained_variance_ratio = perform_pca (data_trials_normalized, n_components = 50)

        var_explained[group].append(explained_variance_ratio)
        eigenvals_slope[group].append(decay_eigenspectra(explained_variance_ratio))

for group in var_explained:
    var_explained[group] = np.array(var_explained[group])
    eigenvals_slope[group] = np.array(eigenvals_slope[group])

variance_explained (var_explained, log = True, timepoints = True)
plot_slope_eigenvals(eigenvals_slope, timepoints = True)

##### stponaneous activity rates at different time points

event_rates = {'Control_0': [],'Control_1': [], 'RD1_0':[], 'RD1_1':[]}
coactive_counts = {'Control_0': [],'Control_1': [], 'RD1_0':[], 'RD1_1':[]}
event_amplitudes= {'Control_0': [],'Control_1': [], 'RD1_0':[], 'RD1_1':[]}
event_durations= {'Control_0': [],'Control_1': [], 'RD1_0':[], 'RD1_1':[]}
dominant_frequencies_per_cell= {'Control_0': [],'Control_1': [], 'RD1_0':[], 'RD1_1':[]}
z_thresh = 3
for i_animal, animal in enumerate(animals_days.keys()):#[animal for animal in animals_days.keys() if 'GNAT' not in animal]):
    g = 'Control' if 'GCaMP6s' in animal else animal.split('_')[1]
    days_recordings = [(day, subfile) for day in animals_days[animal] for subfile in data_object.dat[animal][day].keys() if ('SFxTF' in subfile) and ('big' in subfile)][:2]

    for i, (day, subfile) in enumerate(days_recordings):
        group = g + '_' +str(i)
        spon_arr = np.squeeze(data_object.dat[animal][day][subfile]['zscored_responses_ttls']['Wait_1']) # shape n_Cells, n_timepoints
        #event_rates[group].append(compute_event_rates_per_cell(spon_arr, z_threshold=z_thresh))

        rates, all_amplitudes, all_durations = compute_event_features_per_cell(spon_arr, z_threshold=z_thresh)
        event_rates[group].append(rates)
        event_amplitudes[group].append(all_amplitudes)
        event_durations[group].append(all_durations)

        active = spon_arr > z_thresh  # shape (n_cells, n_timepoints) > binarized
        coactive_counts[group].append(active.sum(axis=0))  # number of active cells at each timepoint

        # fourier
        n_cells, n_timepoints = spon_arr.shape
        sampling_rate = data_object.fps
        # Compute FFT
        fft_vals = np.fft.rfft(spon_arr, axis=1)  # Only positive frequencies
        fft_freqs = np.fft.rfftfreq(n_timepoints, d=1 / sampling_rate)
        # Compute power spectrum
        power = np.abs(fft_vals) ** 2  # shape: (n_cells, n_freqs)
        # Get dominant frequency per cell
        dominant_freqs = fft_freqs[np.argmax(power[:, 1:], axis=1) + 1]  # skip DC (index 0)
        dominant_frequencies_per_cell[group].append(dominant_freqs)

for group in event_rates:
    event_rates[group] = np.array([item for sublist in event_rates[group] for item in sublist])
    coactive_counts[group] = np.array([item for sublist in coactive_counts[group] for item in sublist])
    event_amplitudes[group] = np.array([item for sublist in event_amplitudes[group] for item in sublist])
    event_durations[group] = np.array([item for sublist in event_durations[group] for item in sublist])
    dominant_frequencies_per_cell[group] = np.array([item for sublist in dominant_frequencies_per_cell[group] for item in sublist])
    #eigenvals_slope[group] = np.array(eigenvals_slope[group])

plot_spon_event_rate(event_rates, timepoints=True)
coactive_cells_per_frame(coactive_counts, timepoints=True)
plot_spon_event_properties(event_amplitudes, event_durations, timepoints=True)
plot_dominant_frequencies(dominant_frequencies_per_cell, timepoints=True)

n_trials_plot = 15
n_cells_plot = 100
# take the first n_trials_plot trials
data_trials = data_trials_all[20:20+n_trials_plot, :n_cells_plot]
valid_epochs_short = valid_epochs[20:20+n_trials_plot]



stacked_trials = data_trials.reshape((data_trials.shape[0]*data_trials.shape[-1], -1)).shape

def binarize_array(data, threshold=4):
    return (data >= threshold).astype(int)

def plot_raster(data_trials_array, valid_epochs_arr, n_trials_to_plot = 20, n_cells_to_plot = 100, threshold=4):

    # take the first n_trials_plot trials > shape n_epochs, n_cells, n_timepoints
    dat = data_trials_array[:n_trials_to_plot, :n_cells_to_plot]
    valid_epochs_short = valid_epochs_arr[:n_trials_to_plot]

    # shape n_cells, n_epochs x n_timepoints
    stacked_trials = np.hstack([dat[epoch] for epoch in range(dat.shape[0])])#dat.reshape((dat.shape[0]*dat.shape[-1], -1)).T
    bin_stacked = binarize_array(stacked_trials, threshold=threshold)

    plt.figure(figsize = (10,5))
    plt.imshow(bin_stacked, cmap = 'Greys')
    for epoch in range(dat.shape[0]):
        if valid_epochs_short[epoch]:
            plt.axvspan(epoch * dat.shape[-1], (epoch + 1) * dat.shape[-1], color='grey', alpha=0.2)
            #print(epoch * dat.shape[-1], (epoch + 1) * dat.shape[-1])
    plt.xticks(np.arange(0,stacked_trials.shape[-1],data_object.fps*5), np.arange(0,stacked_trials.shape[-1],data_object.fps*5)/data_object.fps)
    plt.xlabel('Time (s)')
    plt.ylabel('Cell Number')
    plt.show()

plot_raster(data_trials_all, valid_epochs, n_trials_to_plot = 20, n_cells_to_plot = 100, threshold=4)

def plot_raster_with_valid_times(data_trials, valid_epochs, threshold=4):
    """
    Plots a raster with spikes where the response exceeds the threshold.
    Grey vertical bars highlight timepoints where the epoch is valid.

    :param data_trials: numpy array (n_epochs, n_cells, n_timepoints)
    :param valid_epochs: boolean array (n_epochs,) indicating valid epochs
    :param threshold: response threshold for detecting activity
    """
    n_epochs, n_cells, n_timepoints = data_trials.shape

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    for epoch in range(n_epochs):
        # Identify active cells and their time points
        active_cells, active_times = np.where(data_trials[epoch] > threshold)

        # Offset y-values by epoch to separate trials
        ax.scatter(active_times, active_cells + epoch * n_cells, s=1, color='black')

        # Highlight valid time points with grey bars
        if valid_epochs[epoch]:
            ax.axvspan(epoch * n_timepoints, (epoch + 1) * n_timepoints, color='grey', alpha=0.2)
        print(epoch * n_timepoints, (epoch + 1) * n_timepoints)

    ax.set_xlabel("Time (frames)")
    ax.set_ylabel("Cells (grouped by epoch)")
    ax.set_title("Raster Plot with Valid Epochs Highlighted")
    plt.show()


plot_raster_with_valid_times(data_trials, valid_epochs_short, threshold=4)


plt.figure()
plt.imshow(arr[:100], cmap = 'Greys')
[plt.axvline(line, c = 'red', alpha = 0.5) for line in np.arange(0,arr.shape[-1], chunk_size)]
plt.show()


plt.figure(figsize=(12, 8))
for epoch_idx, (epoch, is_valid) in enumerate(zip(data_trials, valid_epochs)):
    color = 'red' if is_valid else 'black'

    cells, timepoints = np.where(epoch > 4)         # Find spike locations (nonzero values)
    timepoints += epoch_idx * epoch.shape[1] # Adjust timepoints to stack epochs horizontally
    plt.scatter(timepoints, cells, color=color, s=1) # plot spikes

plt.xlabel('Time (stacked epochs)', fontsize=12)
plt.ylabel('Cells', fontsize=12)
plt.title('Raster Plot: Valid (Red) vs Invalid (Black)', fontsize=14)
plt.show()

# Create raster plot
fig, ax = plt.subplots(figsize=(15, 8))

# Get spike times (list of time indices for each cell)
spike_times = [np.where(data_trials[cell])[0] for cell in range(n_cells)]

# Plot raster (each row represents a different cell)
ax.eventplot(spike_times, linelengths=0.8, linewidths=0.5, color="black")

# Add vertical lines at epoch boundaries
for i in range(1, n_epochs):
    ax.axvline(i * epoch_length, color="red", linestyle="--", linewidth=1.5)

# Labels and formatting
ax.set_xlabel("Time (bins)")
ax.set_ylabel("Cell index")
ax.set_title("Raster Plot with 5 Epochs Stacked Horizontally")
ax.invert_yaxis()  # Ensure cell 0 is at the top

plt.tight_layout()
plt.show()





def plot_pca_projection(animal, principal_components, n_repeats, n_orientations, trial_type = 'evoked'):
    """
    Plots the projection of trials onto the first two principal components,
    with color indicating stimulus orientation.

    Args:
        animal (str): Animal identifier.
        principal_components (np.ndarray): Principal components of shape (n_trials, n_components).
        n_repeats (int): Number of stimulus repeats.
        n_orientations (int): Number of stimulus orientations.
        trial_type (str): either 'evoked' or 'spontaneous'
    """

    if trial_type == 'evoked':
        # Assign colors according to orientation
        orientations = np.linspace(0, 315, n_orientations)  # 0, 45, 90, ..., 315
        orientation_ids = np.repeat(np.arange(n_orientations), n_repeats)
        colors = orientations[orientation_ids]

        fig = plt.figure(figsize=(5, 4))
        #ax = fig.add_subplot(111, projection = '3d')
        ax = fig.add_subplot(111)

        scatter = ax.scatter(
            principal_components[:, 0],
            principal_components[:, 1],
            #principal_components[:, 2],
            c=colors,
            cmap='plasma',
            alpha=0.8
        )

        # color bar showing orientation
        cbar = plt.colorbar(scatter, ax=ax, ticks=orientations)
        cbar.set_label('Orientation (degrees)')
        cbar.set_ticks(orientations)
        cbar.set_ticklabels([f'{int(o)}°' for o in orientations])


    elif trial_type == 'spontaneous':

        fig = plt.figure(figsize=(5, 4))
        #ax = fig.add_subplot(111, projection='3d')
        ax = fig.add_subplot(111)

        scatter = ax.scatter(
            principal_components[:, 0],
            principal_components[:, 1],
            #principal_components[:, 2],
            c=np.arange(principal_components.shape[0]),
            cmap='plasma',
            alpha=0.8
        )

    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_title(f'{animal} - PCA Projection \n of single-trial grating-evoked activity')

    plt.show()
    return fig

# ------------------------
# Variance Explained Curve (for plot b)
# ------------------------
def plot_variance_explained(animal, explained_variance_ratio):
    plt.figure(figsize=(5, 3))
    plt.plot(np.arange(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o')
    plt.xlabel('PC index')
    plt.ylabel('Variance explained')
    plt.title(f'Variance explained by PCs ({animal})')
    return fig

