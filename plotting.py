import matplotlib.pyplot as plt

from helpers import *

def plot_raw_responses (object, animal):
    '''
    Plot the trace of 20 random cells + the start of each TTL (grey dotted line)
    :param object:
    :param animal:
    :return:
    '''

    n_cells_to_plot = 20
    first_minutes = 8

    for day in object.dat[animal].keys():
        for session in object.dat[animal][day].keys():

            # there is only one SF and TF
            if (object.dat[animal][day][session]['n_SF'] > 0) and (object.dat[animal][day][session]['n_TF'] > 0) and (object.dat[animal][day][session]['n_theta'] > 0):

                data_array = object.dat[animal][day][session]['responses'][:, :first_minutes*60*object.fps]
                ttl = object.dat[animal][day][session]['ttl_data']
                cells = [int(num) for num in np.linspace(0, data_array.shape[0]-1, n_cells_to_plot)]

                fig, ax = plt.subplots(figsize=(16, 9))
                colors = plt.cm.turbo(np.linspace(0, 1, data_array.shape[0]))
                fig.tight_layout()
                global_counter = 0
                for i_cell, cell in enumerate(cells):
                    plt.plot( global_counter+data_array[cell,:]/data_array[cell,:].max(), color=colors[cell])
                    global_counter += 1
                    [ax.axvline(x, c= 'grey', ls = '--', linewidth = 0.8, alpha = 0.4) for x in np.unique(ttl) if x < (first_minutes*60*object.fps)]
            else:
                data_array = object.dat[animal][day][session]['responses']
                ttl = np.array([l[2][0] for l in object.dat[animal][day][session]['dict_stim_ttls'].values()])
                cells = [int(num) for num in np.linspace(0, data_array.shape[0] - 1, n_cells_to_plot)]
                #print(ttl)

                fig, ax = plt.subplots(figsize=(16, 9))
                colors = plt.cm.turbo(np.linspace(0, 1, data_array.shape[0]))
                fig.tight_layout()
                global_counter = 0
                for i_cell, cell in enumerate(cells):
                    plt.plot(global_counter + data_array[cell, :] / data_array[cell, :].max(), color=colors[cell])
                    global_counter += 1
                    [ax.axvline(x, c='grey', ls='--', linewidth=0.8, alpha=0.4) for x in ttl]

            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_title(f'{animal} ({day}, {session})', fontsize=15)
            ax.set_xlabel('Time', fontsize=14)
            ax.set_ylabel('Cell #', fontsize=14)
            plt.subplots_adjust(top=0.95)

            if not os.path.exists(os.path.join(object.save_path, 'responses', animal)):
                os.makedirs(os.path.join(object.save_path, 'responses', animal))

            plt.savefig(os.path.join(object.save_path,'responses',animal, f'responses-{animal}_{session}_{day}.png' ))

            if not object.show_plots:
                plt.close('all')

def spatial_footprint (object, animal, day, recording, sub_file):
    suite2p_path = os.path.join(object.data_path, animal, day, recording, sub_file, 'experiments', 'suite2p', 'plane0')

    responses, iscell, ops, stat = suite2p_files(suite2p_path, response_type='deconvolved')
    stat_iscell = stat[iscell == 1]
    plt.figure()
    im = np.zeros((ops['Ly'], ops['Lx']))
    for n in range(0, stat_iscell.shape[0]):
        ypix = stat_iscell[n]['ypix'][~stat_iscell[n]['overlap']]
        xpix = stat_iscell[n]['xpix'][~stat_iscell[n]['overlap']]
        im[ypix, xpix] = n + 1
    plt.axis('off')
    plt.imshow(im, cmap='magma')
    plt.title(suite2p_path.split('\\')[3])
    animal, day, session = suite2p_path.split('\\')[3], suite2p_path.split('\\')[4], suite2p_path.split('\\')[6]

    if not os.path.exists(os.path.join(object.save_path, 'spatial footprints')):
        os.mkdir(os.path.join(object.save_path, 'spatial footprints'))

    plt.savefig(os.path.join(object.save_path, 'spatial footprints', f'sf-{animal}-{day}-{session}.png' ))
    if not object.show_plots:
        plt.close('all')


def plot_tuning_curves (object, animal, day, session):

    if (object.dat[animal][day][session]['n_SF'] == 1) and (object.dat[animal][day][session]['n_TF'] == 1):
        orientations = np.squeeze(object.dat[animal][day][session]['theta'])
        #tuning_curves_mean = np.squeeze(((object.dat[animal][day][session]['mean_ordered_grat_responses'].mean(axis=0)).mean(axis=-1)).T)
        tuning_curves_mean = object.dat[animal][day][session]['tuning_curves'].mean(axis=0).T  # mean across repeats
        tuning_curves = np.squeeze(object.dat[animal][day][session]['mean_ordered_grat_responses'].mean(axis=-1)) # shape n_repeats, n_ori, n_cells

        #plt.savefig(os.path.join(object.save_path, 'tuning curves', animal, f'tc#{n_page}-{animal}-{day}-{session}.png'))
    elif (object.dat[animal][day][session]['n_theta'] > 1) and (object.dat[animal][day][session]['n_SF'] > 1):  # there is more than one orientaiton or SF
        orientations = object.dat[animal][day][session]['theta']
        tuning_curves_mean = object.dat[animal][day][session]['tuning_curves'].mean(axis = 0).T # mean across repeats
        tuning_curves = object.dat[animal][day][session]['tuning_curves'] # shape n_repeats, n_ori, n_cells

        # orientations = object.dat[animal][day][session]['theta']
        # tuning_curves_mean = object.dat[animal][day][session]['tuning_curves']  # mean across repeats
        #print(tuning_curves_mean.shape)
        #tuning_curves = object.dat[animal][day][session]['tuning_curves']  # shape n_repeats, n_ori, n_cells

    else:
        return

    plasma = plt.get_cmap('plasma')
    cNorm = colors.Normalize(vmin=0, vmax=tuning_curves.shape[0] + 1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plasma)
    scalarMap.set_array([])

    n_cells = tuning_curves_mean.shape[0]
    nrows = 7
    ncols = 5
    n_pages = int(np.ceil(n_cells / (nrows * ncols)))

    if not os.path.exists(os.path.join(object.save_path, 'tuning curves', animal)):
        os.makedirs(os.path.join(object.save_path, 'tuning curves', animal))

    with PdfPages(os.path.join(object.save_path, 'tuning curves', animal,
                               f'{animal} tuning curves ({day}, {session}).pdf')) as pdf:

        for n_page in range(n_pages):
            cells = np.arange((n_page + 0) * nrows * ncols, (n_page + 1) * nrows * ncols)
            fig, ax = plt.subplots(nrows, ncols, figsize=(18, 9))
            ax = ax.ravel()

            for i, i_cell in enumerate(cells):
                if i_cell < n_cells:
                    # print(orientations.shape, tuning_curves_mean[i_cell].shape)
                    ax[i].plot(orientations, tuning_curves_mean[i_cell], c='black', linewidth=1.5)
                    [ax[i].plot(orientations, tuning_curves[repeat, :, i_cell], c=scalarMap.to_rgba(repeat), alpha=0.5,
                                linewidth=0.8) for repeat in range(tuning_curves.shape[0])]

                    # # mean response across frames for each block / orientation / cell
                    ax[i].set_xlabel('Orientation (deg)', fontsize=8)
                    # ax[i_cell].set_ylabel(f'Response')
                    # ax[i_cell].set_title(f'ROI {i_cell})')
                    ax[i].set_xticks(orientations[::2])
                    ax[i].set_title(f'OSI: {np.round(object.dat[animal][day][session]["OSI"][i_cell], 3)}')
                if i_cell >= n_cells:
                    ax[i].axis('off')
            plt.tight_layout()
            pdf.savefig()
            if not object.show_plots:
                plt.close()


def plot_rasters(object, animal, day, session):

    if (object.dat[animal][day][session]['n_SF'] == 1) and (object.dat[animal][day][session]['n_TF'] == 1):
        orientations = np.squeeze(object.dat[animal][day][session]['theta'])

        # shape n_timepoints, n_cells, n_orientations
        tuning_curves_mean = np.squeeze(((object.dat[animal][day][session]['mean_ordered_grat_responses_whole'].mean(axis=0))).T)

        n_cells = tuning_curves_mean.shape[1]
        nrows = 8
        ncols = 5
        n_pages = int(np.ceil(n_cells / (nrows*ncols)))

        if not os.path.exists(os.path.join(object.save_path, 'rasters', animal)):
            os.makedirs(os.path.join(object.save_path, 'rasters', animal))

        with PdfPages(os.path.join(object.save_path, 'rasters', animal, f'{animal} rasters ({day}, {session}).pdf')) as pdf:

            for n_page in range(n_pages):
                cells = np.arange((n_page + 0) * nrows * ncols, (n_page + 1) * nrows * ncols)

                fig, ax = plt.subplots (nrows, ncols, figsize = (18,9))
                ax = ax.ravel()

                for i, i_cell in enumerate(cells):
                    if i_cell < n_cells:
                        ax[i].imshow(tuning_curves_mean[:, i_cell, :].T, vmin = tuning_curves_mean.min(), vmax = tuning_curves_mean.max())
                        ax[i].set_xlabel('Time (s)', fontsize = 8)
                        ax[i].set_ylabel(f'Ori (deg)', fontsize = 8)
                        #ax[i_cell].set_title(f'ROI {i_cell})')
                        ax[i].set_xticks(np.arange(tuning_curves_mean.shape[0])[::int(object.fps)])
                        ax[i].set_xticklabels([int(np.round(x)) for x in (np.arange(-object.fps, tuning_curves_mean.shape[0]-object.fps)[::int(object.fps)])/int(object.fps)])
                        ax[i].set_yticks(np.arange(tuning_curves_mean.shape[-1])[::4])
                        ax[i].set_yticklabels ([int(x) for x in orientations[::4]])
                        ax[i].axvline (int(object.fps), c = 'white')
                        ax[i].axvline(2*int(object.fps), c='white')
                    if i_cell >= n_cells:
                        ax[i].axis('off')

                plt.tight_layout()
                pdf.savefig()
                if not object.show_plots:
                    plt.close()

                #plt.savefig(os.path.join(object.save_path, 'rasters', f'rasters#{n_page}-{animal}-{day}-{session}.png'))
    # else:

def plot_activity_rasters (object, animal, day, session):

    ttls_whole = object.dat[animal][day][session]['responses_ttls_whole']

    # shape repeats x n_orientations x n_cells x timepoints (1s before + 1s static + 3s moving)
    stacked = np.array([ttls_whole[key] for key in [t for t in ttls_whole.keys() if 'Grating' in t]])

    # reshape to stack repeats and orientations
    stacked = stacked.reshape(stacked.shape[0]*stacked.shape[1], stacked.shape[2], -1)

    n_cells = stacked.shape[1]
    nrows = 7
    ncols = 7
    n_pages = int(np.ceil(n_cells / (nrows * ncols)))

    if not os.path.exists(os.path.join(object.save_path, 'activity rasters', animal)):
        os.makedirs(os.path.join(object.save_path, 'activity rasters', animal))

    with PdfPages(os.path.join(object.save_path, 'activity rasters', animal, f'{animal} activity rasters ({day}, {session}).pdf')) as pdf:

        for n_page in range(n_pages):
            cells = np.arange((n_page + 0) * nrows * ncols, (n_page + 1) * nrows * ncols)
            fig, ax = plt.subplots(nrows, ncols, figsize=(18, 9))
            ax = ax.ravel()

            for i, i_cell in enumerate(cells):
                if i_cell < n_cells:

                    ax[i].imshow(stacked[:,i_cell,:], vmin=0, vmax=1.5, cmap="gray_r", aspect="auto")
                    ax[i].axvline(object.fps, c = 'blue', alpha = 0.5)
                    ax[i].axvline(object.fps*2, c='red', alpha = 0.5)
                    ax[i].set_xticks([])
                    ax[i].set_yticks([])
                    ax[i].set_aspect('auto')
                    ax[i].set_title(f'ROI {i_cell}')

                if i_cell >= n_cells:
                    ax[i].axis('off')

            plt.tight_layout()
            pdf.savefig()
            plt.show()
            plt.close()


def on_off_responses(object, animal, day, session):

    # excluding first response
    response = object.dat[animal][day][session]['zscored_responses_ttls'][1:]

    # variables and dependencies for colour mapping
    plasma = plt.get_cmap('plasma')
    cNorm = colors.Normalize(vmin=0, vmax=response.shape[0] + 1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plasma)
    scalarMap.set_array([])

    n_cells = response.shape[1]
    nrows = 7
    ncols = 5
    n_pages = int(np.ceil(n_cells / (nrows * ncols)))

    if not os.path.exists(os.path.join(object.save_path, 'on-off responses', animal)):
        os.makedirs(os.path.join(object.save_path, 'on-off responses', animal))

    with PdfPages(os.path.join(object.save_path, 'on-off responses', animal,
                               f'{animal} on-off responses ({day}, {session}).pdf')) as pdf:

        for n_page in range(n_pages):
            cells = np.arange((n_page + 0) * nrows * ncols, (n_page + 1) * nrows * ncols)
            fig, ax = plt.subplots(nrows, ncols, figsize=(18, 9))
            ax = ax.ravel()

            for i, i_cell in enumerate(cells):
                if i_cell < n_cells:

                    x = np.arange(response.shape[-1])
                    # print(orientations.shape, tuning_curves_mean[i_cell].shape)

                    [ax[i].plot(x, response[repeat, i_cell, :], c=scalarMap.to_rgba(repeat), alpha=0.4,
                                linewidth=0.8) for repeat in range(response.shape[0])]
                    ax[i].plot(x, response[:, i_cell, :].mean(axis=0), c='black', linewidth=1.5)

                    # # mean response across frames for each block / orientation / cell
                    ax[i].set_xlabel('Orientation (deg)', fontsize=8)
                    # ax[i_cell].set_ylabel(f'Response')
                    # ax[i_cell].set_title(f'ROI {i_cell})')
                    ax[i].axvline(object.fps * 0, color='grey', linewidth=1, alpha=0.5)
                    ax[i].axvline(object.fps * 4, color='black', linewidth = 1, alpha=0.5)
                    ax[i].axvline(object.fps * 8, color='red', linewidth = 1, alpha=0.5)
                    ax[i].axvline(object.fps * 12, color='black', linewidth = 1, alpha=0.5)
                    ax[i].axvline(object.fps * 16, color='grey', linewidth=1, alpha=0.5)
                    ax[i].set_xticks(x[::object.fps*4]) #int((x[::object.fps])// object.fps))


                if i_cell >= n_cells:
                    ax[i].axis('off')
            plt.tight_layout()
            pdf.savefig()
            if not object.show_plots:
                plt.close()


#data_object.dat['EC_RD1opto_04']['20241112']['small_chirp_000_006']['responses_ttls']
def osi_pd_hist (object, responsive=True):
    '''
    :param object:
    :param responsive: whether or not to only consider responive cells that pass the z score threshold
    :return:
    '''

    screens = ['big', 'small']
    fig, ax = plt.subplots(2, 2, sharey = True, figsize=(9, 8))

    # Extract data
    def get_group_data(group_name, responsive):
        if responsive:
            return list(chain.from_iterable(
                [object.dat[animal][day][sub_file][metric][object.dat[animal][day][sub_file]['thresholded_cells'] == 1]
                 for sub_file in object.dat[animal][day].keys()
                 if (('SFxO' in sub_file) and (screen in sub_file) and ('l4' not in sub_file))]
                for animal in object.dat.keys() if group_name in animal
                for day in object.dat[animal].keys() if (day in object.dict_animals_days[animal])
            ))
        else:
            return list(chain.from_iterable(
                [object.dat[animal][day][sub_file][metric]
                 for sub_file in object.dat[animal][day].keys()
                 if (('SFxO' in sub_file) and (screen in sub_file) and ('l4' not in sub_file))]
                for animal in object.dat.keys() if group_name in animal
                for day in object.dat[animal].keys() if (day in object.dict_animals_days[animal])
            ))

    for k, metric in enumerate(['OSI', 'DSI']):#enumerate(['OSI', 'DSI']):
        for i, screen in enumerate(screens):

            #np.nonzero(object.dat[animal][day][sub_file][metric] * object.dat[animal][day][sub_file]['thresholded_cells'])
            # average response across recordings, repeats, across cells
            opto = get_group_data('MWopto', responsive)
            rd1 = get_group_data('RD1', responsive)
            control = get_group_data('GCaMP6s', responsive)

            # Flatten lists
            opto = list(chain.from_iterable(opto))
            rd1 = list(chain.from_iterable(rd1))
            control = list(chain.from_iterable(control))

            bins = np.linspace(0, 1, 30)
            ax[k, i].hist(opto, edgecolor='blue', linewidth = 2, alpha=0.7,
                       label=f'opto ({len(opto)} cells)', histtype='step',density=True,
                       bins=bins)
            ax[k, i].hist(rd1, edgecolor='red',linewidth = 2, alpha=0.7,
                       label=f'rd1 ({len(rd1)} cells)', histtype='step',density=True,
                       bins=bins)
            ax[k, i].hist(control, edgecolor='black',linewidth = 2, alpha=0.7,
                       label=f'control ({len(control)} cells)', histtype='step',density=True,
                       bins=bins)
            if k == 0:
                #ax[k, i].set_title(f'{screen} monitor \n \n {metric} distribution')
                ax[k, i].set_title(f'Ultrabright Monitor \n \n {metric} distribution' if screen == 'small' else f'Regular Monitor \n \n {metric} distribution')
            else:
                ax[k, i].set_title(f'{metric} distribution')
            ax[k, i].set_xlabel(metric)
            ax[k, i].set_ylabel('proportion of cells')

            # --------------- Statistical Tests --------------- #
            # Kruskal-Wallis test (global test for all groups)
            # this checks that at least one group differs significantly from the others
            # p < 0.05 if at least one group differs
            kw_stat, kw_pval = kruskal(control, rd1, opto)

            # Pairwise KS tests
            comparisons = [('Control', control), ('RD1', rd1), ('Opto', opto)]
            ks_results = {}

            for (name1, data1), (name2, data2) in itertools.combinations(comparisons, 2):
                ks_stat, ks_pval = ks_2samp(data1, data2)
                ks_results[f'{name1} vs {name2}'] = ks_pval

            # # Pairwise Mann-Whitney U tests (alternative to KS)
            # mw_results = {}
            # for (name1, data1), (name2, data2) in itertools.combinations(comparisons, 2):
            #     mw_stat, mw_pval = mannwhitneyu(data1, data2, alternative='two-sided')
            #     mw_results[f'{name1} vs {name2}'] = mw_pval

            # Print results
            print(f'\n{metric} ({screen} monitor)')
            print(f'Kruskal-Wallis test: p = {kw_pval:.4f}')
            for comp, pval in ks_results.items():
                print(f'KS test, {comp}: p = {pval:.4f}')
            # for comp, pval in mw_results.items():
            #     print(f'Mann-Whitney {comp}: p = {pval:.4f}')

    plt.legend()
    plt.savefig(os.path.join(object.save_path, 'OSI-DSI histogram'))
    plt.show()

def split (arr, n_shuffles = 50):
    '''
    :param arr: shape n_repeats, n_cells, n_timepoints
    :return:
    '''
    n_repeats = arr.shape[0]
    n_cells, n_timepoints = arr.shape[-2], arr.shape[-1]
    half = n_repeats //2

    # true split
    half1, half2 = np.reshape(arr[:half], (-1, n_cells, n_timepoints)),  np.reshape(arr[half:], (-1, n_cells, n_timepoints))
    print(half1.shape, half2.shape)
    #half1, half2 = arr[:half].mean(axis=0), arr[half:].mean(axis=0)

    # true_corrs = np.array([
    #     pearsonr(half1[c], half2[c])[0]
    #     for c in range(n_cells)
    # ])

# split (data_object.dat['EC_GCaMP6s_06']['20241113']['big100_chirp_000_001']['zscored_matrix_baseline'])
# split (data_object.dat['EC_GCaMP6s_06']['20240925']['big100_SFxO_000_001']['zscored_matrix_baseline'])


def responsive_cells_amplitude (object):

    stims = ['chirps', 'SFxO']
    screens = ['big', 'small']
    fig, ax = plt.subplots(2, 2, sharey = True, figsize=(5, 8))
    def pop_data (object, group, stim, screen):
        if stim == 'SFxO':
            g = np.array(list(chain.from_iterable(
                [object.dat[animal][day][sub_file]['zscored_matrix_baseline'].mean(axis=(0, 1, 2))[
                     object.dat[animal][day][sub_file]['thresholded_cells'] == 1].max(axis=-1).mean()
                 for sub_file in object.dat[animal][day].keys() if
                 ((stim in sub_file) and (screen in sub_file) and ('l4' not in sub_file))]
                for animal in object.dat.keys() if group in animal
                for day in object.dat[animal].keys() if (day in object.dict_animals_days[animal]))))
        elif stim == 'chirps':
            g = np.array(list(chain.from_iterable(
                [object.dat[animal][day][sub_file]['zscored_matrix_baseline'].mean(axis=0)[
                     object.dat[animal][day][sub_file]['thresholded_cells'] == 1].max(axis=-1).mean()
                 for sub_file in object.dat[animal][day].keys() if
                 ((stim in sub_file) and (screen in sub_file) and ('l4' not in sub_file))]
                for animal in object.dat.keys() if group in animal
                for day in object.dat[animal].keys() if (day in object.dict_animals_days[animal]))))
        return g

    for i_stim, stim in enumerate(stims):
        for i, screen in enumerate(screens):
            # average response across recordings, repeats, across cells
            opto = pop_data(object, 'MWopto', stim, screen)
            rd1 = pop_data(object, 'RD1', stim, screen)
            control = pop_data(object, 'GCaMP6s', stim, screen)

            print(stim, screen, opto, rd1, control)

            # if stim == 'SFxO':
            #     gnat = np.array(list(chain.from_iterable(
            #         [100*object.dat[animal][day][sub_file]['thresholded_cells'].sum()/len(object.dat[animal][day][sub_file]['thresholded_cells'])
            #          for sub_file in object.dat[animal][day].keys() if ((screen in sub_file) and ('l4' not in sub_file))]
            #         for animal in object.dat.keys() if ('GNAT' in animal)
            #         for day in object.dat[animal].keys() if (day in object.dict_animals_days[animal]))))
            #
            #     print(gnat)
            # is there a statistically sig diff in the distributions of the three groups?
            stat, p = kruskal(control, rd1, opto)
            print(f'KW H-statistic: {stat:.3f}, p-value: {p:.3f}')

            if p < 0.05: # follow up with testing pairwise comparisons (with correction for multiple comparisons)
                # Do pairwise comparisons manually:
                print('mannwhitney two-sided test, with bonferroni multiple comparison correction')
                pvals = [
                    mannwhitneyu(control, rd1, alternative='two-sided').pvalue,
                    mannwhitneyu(control, opto, alternative='two-sided').pvalue,
                    mannwhitneyu(rd1, opto, alternative='two-sided').pvalue
                ]

                # Apply Bonferroni correction manually (3 comparisons):
                _, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='bonferroni')

                print(pvals_corrected)

                # Define comparisons in same order as pvals
                comparisons = [
                    ('GCaMP6s', 'rd1'),
                    ('GCaMP6s', 'opto'),
                    ('rd1', 'opto')
                ]

                # Coordinates for group positions on x-axis
                group_coords = {'GCaMP6s': 0, 'rd1': 1, 'opto': 2}
                max_val = max(np.max(control), np.max(rd1), np.max(opto)) * 1.05
                step = max_val * 0.05  # vertical spacing between significance bars
                h = max_val

                # Loop through each comparison and plot if significant
                for (group1, group2), p_val in zip(comparisons, pvals_corrected):
                    if p_val < 0.05:
                        x1, x2 = group_coords[group1], group_coords[group2]
                        y = h
                        h += step  # update height for next line if needed

                        # Decide number of stars
                        if p_val < 0.001:
                            stars = '***'
                        elif p_val < 0.01:
                            stars = '**'
                        else:
                            stars = '*'

                        # Draw the line and stars
                        ax[i_stim, i].plot([x1, x1, x2, x2], [y, y + step / 2, y + step / 2, y], lw=1.5, c='k')
                        ax[i_stim, i].text((x1 + x2) * 0.5, y + step * 0.6, stars,
                                           ha='center', va='bottom', fontsize=14)

            ax[i_stim, i].scatter(np.array([0] * len(control)), control, s = 35, color = 'grey', alpha = 0.45, label = 'control')
            ax[i_stim, i].scatter(np.array([1] * len(rd1)), rd1, s = 35, color = 'firebrick', alpha = 0.45, label = 'RD1')
            ax[i_stim, i].scatter(np.array([2] * len(opto)), opto,s = 35, color = 'blue', alpha = 0.45, label = 'opto')
            # if stim == 'SFxO':
            #     ax[i_stim, i].scatter(np.array([3] * len(gnat)), gnat, s=65, color='pink', alpha=0.6, label='GNAT')

            ax[i_stim, i].bar(np.array([0]), np.nanmean(control), color = 'grey', alpha = 0.3, label = 'control')
            ax[i_stim, i].bar(np.array([1]), np.nanmean(rd1), color = 'firebrick', alpha = 0.3, label = 'RD1')
            ax[i_stim, i].bar(np.array([2]), np.nanmean(opto), color = 'blue', alpha = 0.3, label = 'opto')
            # if stim == 'SFxO':
            #     ax[i_stim, i].bar(np.array([3]), gnat.mean(), color='pink', alpha=0.3, label='GNAT')

            #screen_label = 'regular monitor' if 'big' in screen else 'ultrabright monitor'
            ax[i_stim, i].set_title(f'{"FFF" if "chirps" in stim else "SFxO"} ({"regular monitor" if "big" in screen else "ultrabright monitor"})')

            # if i == 0:
            #     ax[i_stim, i].set_ylabel('Proportion of responsive cells (%)')
            # if stim == 'SFxO':
            #     ax[i_stim, i].set_xticks(np.array([0,1,2,3]))
            #     ax[i_stim, i].set_xticklabels(['Control', 'RD1', 'Opto', 'GNAT'])
            # else:
            ax[i_stim, i].set_xticks(np.array([0, 1, 2]))
            ax[i_stim, i].set_xticklabels(['Control', 'RD1', 'Opto'])

    fig.text(0.04, 0.5, 'Response amplitude', va='center', rotation='vertical', fontsize=12)
    plt.savefig(os.path.join(object.save_path, 'response amplitude'))
    plt.show()

def p_to_stars(p):
    if np.isnan(p):
        return 'n.s.'
    elif p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'n.s.'


def responsive_cells_hist(object):

    stims = ['chirps', 'O']
    screens = ['big', 'small']

    fig, ax = plt.subplots(2,2,
        #sharey=True,
        figsize=(8, 8))

    colors = {'phpeb': '#EE2926','GNAT': '#477BBF'}

    display_names = ['WT',r'GNAT1/2$^{\mathit{mut}}$']

    positions = np.array([0.0, 0.65])

    # Reproducible scatter-point jitter
    rng = np.random.default_rng(42)

    for i_stim, stim in enumerate(stims):
        for i_screen, screen in enumerate(screens):

            current_ax = ax[i_stim, i_screen]

            if stim == 'chirps':

                control = np.array(list(chain.from_iterable(
                            [100* object.dat[animal][day][sub_file]['thresholded_cells'].sum()/ len(object.dat[animal][day][sub_file]['thresholded_cells'])
                                for sub_file in object.dat[animal][day].keys()
                                if (
                                    stim in sub_file
                                    and screen in sub_file
                                    and 'l4' not in sub_file)]
                            for animal in object.dat.keys()
                            if 'phpeb' in animal
                            for day in object.dat[animal].keys()
                            if day in object.dict_animals_days[animal])),dtype=float)

                gnat = np.array(
                    list(chain.from_iterable([100* object.dat[animal][day][sub_file]['thresholded_cells'].sum()/ len(object.dat[animal][day][sub_file][
                                        'thresholded_cells'])
                                for sub_file in object.dat[animal][day].keys()
                                if (stim in sub_file and screen in sub_file and 'l4' not in sub_file)]
                            for animal in object.dat.keys()
                            if 'GNAT' in animal
                            for day in object.dat[animal].keys()
                            if day in object.dict_animals_days[animal])),dtype=float)

            else:

                control = np.array(list(chain.from_iterable([
                                100* object.dat[animal][day][sub_file]['thresholded_cells'].sum()/ len(object.dat[animal][day][sub_file]['thresholded_cells'])
                                for sub_file in object.dat[animal][day].keys()
                                if (
                                    'chirps' not in sub_file
                                    and screen in sub_file
                                    and 'l4' not in sub_file)]
                            for animal in object.dat.keys()
                            if 'phpeb' in animal
                            for day in object.dat[animal].keys()
                            if day in object.dict_animals_days[animal])),dtype=float)

                gnat = np.array(
                    list(chain.from_iterable(
                            [100* object.dat[animal][day][sub_file]['thresholded_cells'].sum()/ len(
                                    object.dat[animal][day][sub_file]['thresholded_cells'])
                                for sub_file in object.dat[animal][day].keys()
                                if (
                                    'chirps' not in sub_file
                                    and screen in sub_file
                                    and 'l4' not in sub_file)]
                            for animal in object.dat.keys()
                            if 'GNAT' in animal
                            for day in object.dat[animal].keys()
                            if day in object.dict_animals_days[animal])),dtype=float)

            # Remove NaNs and infinities
            control = control[np.isfinite(control)]
            gnat = gnat[np.isfinite(gnat)]

            group_data = [control,gnat]

            # make boxplots
            bp = current_ax.boxplot(
                group_data, positions=positions,
                widths=0.30, patch_artist=True,
                showfliers=False, whis=1.5,
                medianprops=dict(color='black',linewidth=2.0),
                boxprops=dict(color='black',linewidth=1.5),
                whiskerprops=dict(color='black',linewidth=1.5),
                capprops=dict(color='black',linewidth=1.5))

            box_colors = [colors['phpeb'],colors['GNAT']]

            for patch, colour in zip(bp['boxes'],box_colors):
                patch.set_facecolor(colour)
                patch.set_alpha(1)
                patch.set_edgecolor('black')
                patch.set_linewidth(1.5)

            # add scatter points
            for i_group, (values, colour) in enumerate(zip(group_data, box_colors)):
                jitter = rng.uniform(-0.04,  0.04,size=len(values))

                # shift points slightly left of their box
                point_offset = -0.22

                x_values = (np.full(len(values),positions[i_group])+ point_offset+ jitter)

                current_ax.scatter(x_values,values,
                    s=42, color=colour, alpha=1, edgecolors='none',zorder=3)

            # Fisher–Pitman permutation test (only for regular monitor & SFxori recording)
            if (screen == 'big' and stim != 'chirps' and len(control) > 0 and len(gnat) > 0):

                def diff_means(x, y):
                    return np.mean(x) - np.mean(y)

                n_control, n_gnat = len(control), len(gnat)
                n_perms = comb(n_control + n_gnat, n_control)

                print(
                    f"\n{stim}, {screen}: "
                    f"n = {n_control} vs {n_gnat} -> "
                    f"{n_perms} distinct group assignments"
                )

                res = permutation_test((control, gnat),diff_means,
                    permutation_type='independent', n_resamples=np.inf, alternative='two-sided')

                statistic = res.statistic
                p = res.pvalue

                # Fisher–Pitman standardized Z statistic
                all_values = np.concatenate([control,gnat])

                n_total = n_control + n_gnat

                T = control.sum()
                E = n_control * all_values.mean()

                V = ((n_control * n_gnat)/ (n_total * (n_total - 1))* ((all_values- all_values.mean()) ** 2).sum())

                if V > 0:
                    z = (T - E) / np.sqrt(V)
                else:
                    z = np.nan

                print(
                    "Exact Fisher-Pitman permutation test: "
                    f"mean difference = {statistic:.4f}, "
                    f"Z = {z:.3f}, "
                    f"p = {p:.4f}"
                )

                # Add significance bracket only when significant
                if p < 0.05:

                    if p < 0.001:
                        stars = '***'
                    elif p < 0.01:
                        stars = '**'
                    else:
                        stars = '*'

                    all_data = np.concatenate([control, gnat])

                    data_min = np.min(all_data)
                    data_max = np.max(all_data)
                    data_range = data_max - data_min

                    if data_range == 0:
                        data_range = 1

                    bracket_bottom = (data_max+ 0.07 * data_range)

                    bracket_top = ( data_max + 0.13 * data_range)

                    current_ax.plot(
                        [positions[0],positions[0],positions[1],positions[1]],
                        [bracket_bottom, bracket_top, bracket_top, bracket_bottom],
                        color='black', linewidth=1.5, clip_on=False)

                    current_ax.text(
                        np.mean(positions),
                        bracket_top + 0.02 * data_range,
                        stars,
                        ha='center', va='bottom',
                        fontsize=18, fontfamily='Arial', color='black')

            # labels and formatting
            stimulus_label = ('FFF' if stim == 'chirps' else 'SF × O')

            screen_label = ( 'Regular monitor' if screen == 'big' else 'Ultrabright monitor')

            current_ax.set_title(
                f'{stimulus_label}\n{screen_label}',
                fontsize=15, fontfamily='Arial', color='black', pad=12)

            current_ax.set_xticks(positions)
            current_ax.set_xticklabels(
                display_names,
                fontsize=13, fontfamily='Arial', color='black')

            current_ax.set_facecolor('white')

            # Only show left spine
            current_ax.spines['left'].set_visible(True)
            current_ax.spines['left'].set_color('black')
            current_ax.spines['left'].set_linewidth(1.5)
            current_ax.spines['right'].set_visible(False)
            current_ax.spines['top'].set_visible(False)
            current_ax.spines['bottom'].set_visible(False)

            current_ax.grid(False)
            current_ax.tick_params(axis='y',direction='out', length=5, width=1.5, colors='black', labelsize=12)
            current_ax.tick_params(axis='x', length=0, pad=9)

    fig.text( 0.02,0.5,'Proportion of responsive cells (%)',
        va='center', rotation='vertical', fontsize=15, fontfamily='Arial')

    fig.patch.set_facecolor('white')

    plt.subplots_adjust(
        left=0.13, right=0.98,
        bottom=0.10, top=0.92,
        wspace=0.35, hspace=0.42)

    plt.savefig(
        os.path.join(
            object.save_path,
            'percent_responsive_cells_gnat.pdf'),
        format='pdf',
        bbox_inches='tight')

    plt.savefig(
        os.path.join(
            object.save_path,
            'percent_responsive_cells_gnat.png'),
        dpi=300, bbox_inches='tight')

    plt.show()

def hist_osi_angle (obj, animal, day, session):
    '''
    plots histogram distribution of OSIs/preferred orientation for each day
    :param obj:
    :return:
    '''

    if not os.path.exists(os.path.join(obj.save_path, 'histogram', animal)):
        os.makedirs(os.path.join(obj.save_path, 'histogram', animal))

    with PdfPages(os.path.join(obj.save_path, 'histogram', animal,f'{animal} OSI_PD histogram ({day}, {session}).pdf')) as pdf:
        fig, ax = plt.subplots(ncols = 2, figsize = (6,5))

        ax[0].hist(obj.dat[animal][day][session]['preferred_orientation'], color =  'r', alpha = 0.5, label = f'Cell count, {day}', bins = np.linspace(-180,180,25))
        ax[1].hist(obj.dat[animal][day][session]['OSI'], color =  'b', alpha = 0.5, label = f'Cell count, {day}', bins = np.linspace(0,1,25))

        # ax[i,0].set_ylim([0, 6])
        # ax[i, 0].set_xlim([-100, 100])
        # ax[i, 1].set_ylim([0, 18])
        ax[0].set_xlabel('Orientation (deg)')
        ax[0].set_ylabel('Cell count')
        ax[0].set_title("Preferred Orientation")
        ax[1].set_title("Orientation Selectivity Index")
        ax[1].set_xlabel('OSI')
        ax[1].set_ylabel('Cell count')

        #font_kwargs = dict(fontsize="large")
        #add_headers(fig, col_headers=["Preferred Orientation", "Orientation Selectivity Index"], row_headers=['Cell count \n (' + day + ')' for day in list(obj.dat_subject.keys())], **font_kwargs)
        plt.suptitle(f'{animal}, {day}, {session}')

        plt.tight_layout()
        plt.show()
        pdf.savefig()
        # if not obj.show_plots:
        #     plt.close()


def sf_tuning_curves(obj, animal, day, session, amplitude_curve_SFs, sorted_sfs_arr):

    if not os.path.exists(os.path.join(obj.save_path, 'sf_tuning_curve', animal)):
        os.makedirs(os.path.join(obj.save_path, 'sf_tuning_curve', animal))

    n_cells = amplitude_curve_SFs.shape[-1]
    nrows = 6
    ncols = 8
    n_pages = int(np.ceil(n_cells / (nrows*ncols)))

    with PdfPages(os.path.join(obj.save_path, 'sf_tuning_curve', animal, f'{animal} sf tuning ({day}, {session}).pdf')) as pdf:

        for n_page in range(n_pages):
            cells = np.arange((n_page + 0) * nrows * ncols, (n_page + 1) * nrows * ncols)

            fig, ax = plt.subplots(nrows, ncols, figsize=(23, 12), sharex=True, sharey=False)
            ax = ax.ravel()

            for i, i_cell in enumerate(cells):
                if i_cell < n_cells:

                    ax[i].plot(amplitude_curve_SFs[:, :, i_cell].mean(axis = 0), c='black', alpha=1, linewidth = 3)
                    [ax[i].plot(amplitude_curve_SFs[i_repeat, :, i_cell], c = 'blue', alpha = 0.4) for i_repeat in range(amplitude_curve_SFs.shape[0])]
                    #print(np.arange(amplitude_curve_SFs[:, :, i_cell].shape[0]).shape)
                    ax[i].set_xticks(np.arange(sorted_sfs_arr.shape[1]))
                    #print(np.squeeze(sorted_sfs_arr.mean(axis=0)).shape)
                    ax[i].set_xticklabels(np.squeeze(sorted_sfs_arr.mean(axis=0)))
                    ax[i].set_title(f'ROI {i_cell}')
                    ax[i].set_xlabel('SF')
                    ax[i].set_ylabel('Amplitude')

                if i_cell >= n_cells:
                    ax[i].axis('off')

            plt.tight_layout()
            pdf.savefig()
            if not obj.show_plots:
                plt.close()

def sf_tf_tuning_curves(obj, animal, day, session, sorted_sfs_arr, sorted_tfs_arr):

    sf_tf_tuning = obj.dat[animal][day][session]['SF_TF_tuning']

    if not os.path.exists(os.path.join(obj.save_path, 'sf_tf_tuning_curve', animal)):
        os.makedirs(os.path.join(obj.save_path, 'sf_tf_tuning_curve', animal))

    n_cells = sf_tf_tuning.shape[-1]
    nrows = 6
    ncols = 9
    n_pages = int(np.ceil(n_cells / (nrows*ncols)))

    with PdfPages(os.path.join(obj.save_path, 'sf_tf_tuning_curve', animal, f'{animal} sf tf tuning ({day}, {session}).pdf')) as pdf:

        for n_page in range(n_pages):
            cells = np.arange((n_page + 0) * nrows * ncols, (n_page + 1) * nrows * ncols)

            fig, ax = plt.subplots(nrows, ncols, figsize=(23, 12), sharex=True, sharey=True)
            ax = ax.ravel()

            for i, i_cell in enumerate(cells):
                if i_cell < n_cells:

                    ax[i].imshow(sf_tf_tuning[:, :, i_cell].T, cmap = 'plasma', aspect='auto')

                    ax[i].set_xticks(np.arange(len(sorted_sfs_arr)))
                    ax[i].set_xticklabels(np.squeeze(sorted_sfs_arr))
                    ax[i].set_yticks(np.arange(len(sorted_tfs_arr)))
                    ax[i].set_yticklabels(np.squeeze(sorted_tfs_arr))
                    ax[i].set_title(f'ROI {i_cell}')
                    ax[i].set_xlabel('SF')
                    ax[i].set_ylabel('TF')

                if i_cell >= n_cells:
                    ax[i].axis('off')

            plt.tight_layout()
            pdf.savefig()
            if not obj.show_plots:
                plt.close()

#parameter_matrix_plot(data_object, 'EC_GCaMP6s_09', '20241122', 'small_SFxO_000_012')

def polar_plots(object, animal, day, subfile, n_cells_plot = 32, zscore_threshold = 3):

    if ((object.dat[animal][day][subfile]['n_theta'] != 1) and (object.dat[animal][day][subfile]['n_TF'] == 1)):
        cells_to_plot = np.arange(n_cells_plot)

        # param_matrix: shape ((n_repeats, n_orientation, n_sf, cells, timepoints))
        param_matrix, thetas, sfs, tfs = build_parameter_matrix(object, animal, day, subfile, response_window='whole', zscore=True)

        # shape ((n_cells)) > index of preferred spatial frequency
        preferred_sf = pref_sf(object, param_matrix)

        colors = plt.cm.plasma(np.linspace(0, 0.9, len(sfs) * len(thetas)))
        colors_cells = plt.cm.plasma(np.linspace(0, 0.9, n_cells_plot))

        #zscore_threshold_cells = object.dat[animal][day][subfile]['thresholded_cells'] #zscore_thresholding(object, animal, day, subfile, zscore_threshold = zscore_threshold)

        _, tuning_curves_sf_pref, (complex_sf_pref, osi_sf_pref, pref_orientation_sf_pref), (_,_,_), _, = complex_phase(object, animal, day, subfile, sf_i=preferred_sf)

        if (object.dat[animal][day][subfile]['n_SF'] > 1) and (object.dat[animal][day][subfile]['n_theta'] > 1):
            if not os.path.exists(os.path.join(object.save_path, 'polar tuning curve', animal)):
                os.makedirs(os.path.join(object.save_path, 'polar tuning curve', animal))

            n_rows = 4
            n_cols = int(np.ceil(n_cells_plot / n_rows))
            fig, ax = plt.subplots(nrows= n_rows, ncols=n_cols, subplot_kw={'projection': 'polar'}, figsize=(18, 9))
            ax = ax.ravel()
            for cell_i in cells_to_plot:

                # average across repeats
                rmax = np.array((build_tuning_curves(object, animal, day, subfile, avg_over_param=None, sf_i=preferred_sf)[1][:, :, cell_i])).max()

                # we want to look at tuning of a specific sf
                # shape ((n_repeats, n_orientation, cells))
                _, response_sf, _ = build_tuning_curves(object, animal, day, subfile, avg_over_param=None, sf_i=preferred_sf)
                (_, osi, _), (_, dsi, _) = complex_phase_from_tuning(response_sf, thetas)

                # r = vector of responses for each direction(response vector)
                r = response_sf[:, :, cell_i].mean(axis=0)
                r_repeats = response_sf[:, :, cell_i]
                r /= rmax
                r_repeats /= rmax
                #r /= r.sum()  # normalizing responses so they're between 0 and 1
                theta = np.deg2rad(thetas)

                # to join the last point and first point
                idx = np.arange(r.shape[0] + 1)
                idx[-1] = 0
                if 'GCaMP6s' in animal:
                    colour = 'black'
                elif 'RD1' in animal:
                    colour = 'red'
                elif 'MWopto' in animal:
                    colour = 'blue'
                idx = np.arange(r.shape[0] + 1)
                idx[-1] = 0
                ax[cell_i].plot(theta, r, linewidth=2, color=colour,
                              alpha=0.8, zorder =10)  # , color=scalarMap.to_rgba(cell), alpha=0.6)
                ax[cell_i].plot(theta[idx], r[idx], linewidth=2, color=colour,
                              alpha=0.8, zorder =10)  # , color=scalarMap.to_rgba(cell), alpha=0.6)

                for i in range(r_repeats.shape[0]):
                    idx = np.arange(r_repeats[i].shape[0] + 1)
                    idx[-1] = 0
                    ax[cell_i].plot(theta, r_repeats[i], linewidth=1, color='grey',
                                  alpha=0.4)  # , color=scalarMap.to_rgba(cell), alpha=0.6)
                    ax[cell_i].plot(theta[idx], r_repeats[i][idx], linewidth=1, color='grey',
                                  alpha=0.4)  # , color=scalarMap.to_rgba(cell), alpha=0.6)
                ax[cell_i].set_thetagrids([0, 90, 180, 270], y=0.15,
                                        labels=['0', '\u03c0' + '/2', '\u03c0', '3' + '\u03c0' + '/2'],
                                        fontsize=8)  # labels = ['0', '','\u03c0','']
                ax[cell_i].set_rlabel_position(45)  # r is normalized response
                #polar_ax.set_rticks([np.round(r.max(), 1)])
                ax[cell_i].set_rticks([])
                ax[cell_i].set_rmax(1)
                ax[cell_i].grid(True)

            if object.dat[animal][day][subfile]['thresholded_cells'][cell_i]:
                ax[cell_i].set_title(f'Cell {cell_i} \nOSI: {np.round(osi[cell_i], 2)}, DSI: {np.round(dsi[cell_i], 2)}', fontsize = 10, color = 'green', fontweight = 'bold')
            else:
                ax[cell_i].set_title(f'Cell {cell_i} \nOSI: {np.round(osi[cell_i], 2)}, DSI: {np.round(dsi[cell_i], 2)}', fontsize=10, color='r')
        plt.tight_layout()
        plt.savefig(os.path.join(object.save_path, 'polar tuning curve', animal,
                                 f'{animal} polar tuning curve ({day}, {subfile}).pdf'))
        plt.savefig(os.path.join(object.save_path, 'polar tuning curve', animal,
                                 f'{animal} polar tuning curve ({day}, {subfile}).svg'))
        plt.show()


def tuning_curves(object, animal, day, subfile, n_cells_plot = 36, zscore_threshold = 3):

    if ((object.dat[animal][day][subfile]['n_theta'] != 1) and (object.dat[animal][day][subfile]['n_TF'] == 1)):
        cells_to_plot = np.arange(n_cells_plot)

        # param_matrix: shape ((n_repeats, n_orientation, n_sf, cells, timepoints))
        param_matrix, thetas, sfs, tfs = build_parameter_matrix(object, animal, day, subfile, response_window='whole', zscore=True)

        # shape ((n_cells)) > index of preferred spatial frequency
        preferred_sf = pref_sf(object, param_matrix)

        colors = plt.cm.plasma(np.linspace(0, 0.9, len(sfs) * len(thetas)))
        colors_cells = plt.cm.plasma(np.linspace(0, 0.9, n_cells_plot))

        #zscore_threshold_cells = object.dat[animal][day][subfile]['thresholded_cells'] #zscore_thresholding(object, animal, day, subfile, zscore_threshold = zscore_threshold)

        _, tuning_curves_sf_pref, (complex_sf_pref, osi_sf_pref, pref_orientation_sf_pref), (_,_,_), _, = complex_phase(object, animal, day, subfile, sf_i=preferred_sf)

        if (object.dat[animal][day][subfile]['n_SF'] > 1) and (object.dat[animal][day][subfile]['n_theta'] > 1):
            if not os.path.exists(os.path.join(object.save_path, 'tuning curve', animal)):
                os.makedirs(os.path.join(object.save_path, 'tuning curve', animal))

            n_rows = 6
            n_cols = int(np.ceil(n_cells_plot / n_rows))
            fig, ax = plt.subplots(nrows= n_rows, ncols=n_cols, figsize=(18, 9))
            ax = ax.ravel()
            #for i, cell_i in enumerate(np.argwhere(object.dat[animal][day][subfile]['thresholded_cells']==1)[:,0][:n_cells_plot]):
            for i, cell_i in enumerate(np.arange(n_cells_plot)):

                # average across repeats
                rmax = np.array((build_tuning_curves(object, animal, day, subfile, avg_over_param=None, sf_i=preferred_sf)[1][:, :, cell_i])).max()

                # we want to look at tuning of a specific sf
                # shape ((n_repeats, n_orientation, cells))
                _, response_sf, _ = build_tuning_curves(object, animal, day, subfile, avg_over_param=None, sf_i=preferred_sf)
                (_, osi, _), (_, dsi, _) = complex_phase_from_tuning(response_sf, thetas)

                # r = vector of responses for each direction(response vector)
                r = response_sf[:, :, cell_i].mean(axis=0)
                r_repeats = response_sf[:, :, cell_i]
                r /= rmax
                r_repeats /= rmax
                #r /= r.sum()  # normalizing responses so they're between 0 and 1
                theta = thetas #np.deg2rad(thetas)

                # to join the last point and first point
                if 'GCaMP6s' in animal:
                    colour = 'black'
                elif 'RD1' in animal:
                    colour = 'firebrick'
                elif 'MWopto' in animal:
                    colour = 'blue'

                ax[i].plot(theta, r, linewidth=2, color=colour,
                              alpha=0.8, zorder =10)  # , color=scalarMap.to_rgba(cell), alpha=0.6)

                for i_repeat in range(r_repeats.shape[0]):
                    ax[i].plot(theta, r_repeats[i_repeat], linewidth=1, color='grey',
                                  alpha=0.4)  # , color=scalarMap.to_rgba(cell), alpha=0.6)

                # ax[cell_i].set_thetagrids([0, 90, 180, 270], y=0.15,
                #                         labels=['0', '\u03c0' + '/2', '\u03c0', '3' + '\u03c0' + '/2'],
                #                         fontsize=8)  # labels = ['0', '','\u03c0','']
                #ax[cell_i].set_rlabel_position(45)  # r is normalized response
                #polar_ax.set_rticks([np.round(r.max(), 1)])
                # ax[cell_i].set_rticks([])
                # ax[cell_i].set_rmax(1)
                # ax[cell_i].grid(True)

                ax[i].set_xticks([0, 90, 180, 270])

                if object.dat[animal][day][subfile]['thresholded_cells'][cell_i]:
                    ax[i].set_title(f'Cell {cell_i} \nOSI: {np.round(osi[cell_i], 2)}, DSI: {np.round(dsi[cell_i], 2)}', fontsize = 10, color = 'green', fontweight = 'bold')
                else:
                    ax[i].set_title(f'Cell {cell_i} \nOSI: {np.round(osi[cell_i], 2)}, DSI: {np.round(dsi[cell_i], 2)}', fontsize=10, color='r')

        plt.tight_layout()
        plt.savefig(os.path.join(object.save_path, 'tuning curve', animal,
                                 f'{animal} tuning curve ({day}, {subfile}).pdf'))
        plt.savefig(os.path.join(object.save_path, 'tuning curve', animal,
                                 f'{animal} tuning curve ({day}, {subfile}).svg'))
        plt.show()



def tuning_curves_aligned(object, animal, day, subfile):

    if ((object.dat[animal][day][subfile]['n_theta'] != 1) and (object.dat[animal][day][subfile]['n_TF'] == 1)):
        # param_matrix: shape ((n_repeats, n_orientation, n_sf, cells, timepoints))
        param_matrix, thetas, sfs, tfs = build_parameter_matrix(object, animal, day, subfile, response_window='whole', zscore=True)

        # shape ((n_cells)) > index of preferred spatial frequency
        preferred_sf = pref_sf(object, param_matrix)

        _, tuning_curves_sf_pref, _ = build_tuning_curves(object, animal, day, subfile, zscore = False, avg_over_param = None, sf_i=preferred_sf)

        tuning_curves_sf_pref_mean = tuning_curves_sf_pref.mean(axis = 0)

        # for each cell, take the angle which yields the max mean response as the best angle index
        max_theta_idx = np.argmax(tuning_curves_sf_pref.mean(axis=0), axis=0) # shape n_ cells

        aligned_tuning_curves = np.stack([np.roll(tuning_curves_sf_pref_mean[:, i], (len(thetas) // 2) - max_theta_idx[i]) for i in range(tuning_curves_sf_pref_mean.shape[1])], axis=1)

        # normalize to max response
        #aligned_tuning_curves /= aligned_tuning_curves.max(axis = 0)

        # only return thresholded cells
        return aligned_tuning_curves[:, object.dat[animal][day][subfile]['thresholded_cells']]

def tuning_curves_aligned_groups(object):
    def get_group_dat (group):
        group_dat = np.concatenate(
            list(chain.from_iterable(
                [tuning_curves_aligned(object, animal, day, sub_file)
                 for sub_file in object.dat[animal][day].keys()
                 if 'SFxO' in sub_file and 'big100' in sub_file]
                for animal in object.dat.keys() if group in animal
                for day in object.dat[animal].keys()
            )), axis=1)  # concatenate across cells
        return group_dat

    # tuning curve for all cells > shape n_ori, n_cells
    control = get_group_dat('GCaMP6s')
    rd1 = get_group_dat('RD1')
    opto = get_group_dat('MWopto')

    # After loading your tuning curves...
    global_max = max(control.max(), rd1.max(), opto.max())

    # Normalize all curves using the same reference max
    control /= global_max
    rd1 /= global_max
    opto /= global_max

    control = 0.5 * (control + np.roll(control, shift=control.shape[0] // 2, axis=0))
    rd1 = 0.5 * (rd1 + np.roll(rd1, shift=rd1.shape[0] // 2, axis=0))
    opto = 0.5 * (opto + np.roll(opto, shift=opto.shape[0] // 2, axis=0))

    control_mean, control_sem = control.mean(axis=1), sem(control, axis=1)
    rd1_mean, rd1_sem = rd1.mean(axis=1), sem(rd1, axis=1)
    opto_mean, opto_sem = opto.mean(axis=1), sem(opto, axis=1)

    plt.figure()
    plt.plot(control_mean, c = 'black')
    plt.plot(rd1_mean, c='red')
    plt.plot(opto_mean, c='blue')

    plt.fill_between(np.arange(len(control_mean)), control_mean - control_sem, control_mean + control_sem, color='black', alpha=0.3, label='Sighted')
    plt.fill_between(np.arange(len(rd1_mean)), rd1_mean - rd1_sem, rd1_mean + rd1_sem, color='red', alpha=0.3, label='rd1')
    plt.fill_between(np.arange(len(opto_mean)), opto_mean - opto_sem, opto_mean + opto_sem, color='blue', alpha=0.3, label='rd1:MWopto')

    #plt.xticks(np.arange(8), ['-135°', '-90°', '-45°', '0°', '45°', '90°', '135°', '180°'])
    plt.xticks(np.arange(8), ['-180°', '-135°', '-90°', '-45°', '0°', '45°', '90°', '135°'])

    plt.xlabel('Orientation offset (aligned to preferred)')
    plt.legend()
    plt.ylabel('Normalized response')
    plt.tight_layout()
    plt.show()

