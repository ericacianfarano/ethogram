import matplotlib.pyplot as plt

from retmap import *

screen = 'big100'
project = 'ethogram'  # restored or ethogram
over_time = False
path = r'I:\retmap'

if project == 'restored':

    batch_retmap(animals_days, screen, path, draw_mask=False)

    # total visual area across different groups
    total_visual_area_g, largest_neg_patch_area_g, mean_amplitude_g = {}, {}, {}
    for animal in animals_days:
        g = animal[3:-3]

        if (g not in total_visual_area_g):
            total_visual_area_g[g] = [total_visual_area[animal]]
            largest_neg_patch_area_g[g] = [largest_neg_patch_area[animal]]
            mean_amplitude_g[g] = [mean_amplitude[animal]]
        else:
            total_visual_area_g[g].append(total_visual_area[animal])
            largest_neg_patch_area_g[g].append(largest_neg_patch_area[animal])
            mean_amplitude_g[g].append(mean_amplitude[animal])

    fig, ax = plt.subplots (1,3, figsize = (12,4))
    group_names = ['GCaMP6s', 'GNAT', 'RD1', 'RD1opto']
    for i_subplot, metric, in enumerate([total_visual_area_g, largest_neg_patch_area_g, mean_amplitude_g]):

        for i, group in enumerate(group_names):
            arr = np.array(metric[group])

            if group == 'GCaMP6s':
                colour = 'black'
            if group == 'GNAT':
                colour = 'tomato'
            if group == 'RD1':
                colour = 'firebrick'
            if group == 'RD1opto':
                colour = 'blue'

            ax[i_subplot].scatter ([i]*len(arr) + np.random.uniform(-0.2, 0.2, len(arr)), arr, c = colour, alpha = 0.5, s = 30)
            ax[i_subplot].bar ([i], arr.mean(), color = colour, alpha = 0.6)

            if i_subplot == 0:
                ax[i_subplot].set_title ('total visual area')
            elif i_subplot == 1:
                ax[i_subplot].set_title ('largest negative patch')
            elif i_subplot == 2:
                ax[i_subplot].set_title ('mean pixel amplitude')

        # significance tests
        control, gnat, rd1, opto = metric['GCaMP6s'], metric['GNAT'], metric['RD1'], metric['RD1opto']
        stat, p = kruskal(control, gnat, rd1, opto)
        print(f'KW H-statistic: {stat:.3f}, p-value: {p:.3f}')

        if p < 0.05:  # follow up with testing pairwise comparisons (with correction for multiple comparisons)
            # Do pairwise comparisons manually:
            print('mannwhitney two-sided test, with bonferroni multiple comparison correction')
            pvals = [
                mannwhitneyu(control, gnat, alternative='two-sided').pvalue,
                mannwhitneyu(control, rd1, alternative='two-sided').pvalue,
                mannwhitneyu(control, opto, alternative='two-sided').pvalue,
                mannwhitneyu(rd1, opto, alternative='two-sided').pvalue
            ]

            # Apply Bonferroni correction manually (3 comparisons):
            _, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='bonferroni')

            print(pvals_corrected)

            # Define comparisons in same order as pvals
            comparisons = [
                ('GCaMP6s', 'GNAT'),
                ('GCaMP6s', 'RD1'),
                ('GCaMP6s', 'RD1opto'),
                ('RD1', 'RD1opto')
            ]

            # Coordinates for group positions on x-axis
            group_coords = {'GCaMP6s': 0, 'GNAT':1, 'RD1': 2, 'RD1opto': 3}

            max_val = max(np.max(control), np.max(gnat), np.max(rd1), np.max(opto)) * 1.05
            print(max_val)
            step = max_val * 0.05  # vertical spacing between significance bars
            h = max_val

            # Loop through each comparison and plot if significant
            for (group1, group2), p_val in zip(comparisons, pvals_corrected):
                if p_val < 0.05:
                    x1, x2 = group_coords[group1], group_coords[group2]
                    y = h
                    h += step  # update height for next line if needed

                    print(x1, x2, y)

                    # Decide number of stars
                    if p_val < 0.001:
                        stars = '***'
                    elif p_val < 0.01:
                        stars = '**'
                    else:
                        stars = '*'

                    # Draw the line and stars
                    ax[i_subplot].plot([x1, x1, x2, x2], [y, y + step / 2, y + step / 2, y], lw=1.5, c='k')
                    ax[i_subplot].text((x1 + x2) * 0.5, y + step * 0.6, stars,
                                       ha='center', va='bottom', fontsize=14)

        ax[i_subplot].set_xticks(range(len(np.unique([animal[3:-3] for animal in animals_days.keys()]))))
        ax[i_subplot].set_xticklabels(np.unique([animal[3:-3] for animal in animals_days.keys()]))
        ax[i_subplot].set_ylabel('# pixels')
    plt.show()

if project == 'restored' and over_time:

    batch_retmap(animals_days, screen, path, animal_dobs=animals_birthdate, draw_mask=False)

    # Define colors for each group
    group_colors = {
        'GCaMP6s': 'black',
        'GNAT': 'tomato',
        'RD1': 'firebrick'
    }

    fig, ax = plt.subplots (1,2, figsize = (8,4))
    group_names = ['GCaMP6s', 'GNAT', 'RD1']
    for i_subplot, metric, in enumerate([total_visual_area_days, largest_neg_patch_area_days]):

        for subject, data in metric.items():
            # Assume subject is in the format "EC_GROUP_xx" (e.g., "EC_GCaMP6s_06")
            parts = subject.split('_')
            group = parts[1]  # e.g., "GCaMP6s", "GNAT", "RD1", etc.

            # Extract age and value pairs; convert age from "P86" to 86.
            ages = []
            values = []
            for age_str, val in data.items():
                ages.append(int(age_str.lstrip('P')))
                values.append(val)

            # Sort the pairs by age
            ages = np.array(ages)
            values = np.array(values)
            order = np.argsort(ages)
            ages_sorted = ages[order]
            values_sorted = values[order]

            # Plot the trajectory for this animal
            ax[i_subplot].scatter(ages_sorted, values_sorted, linestyle='-',
                     color=group_colors.get(group, 'gray'), label=subject.split('_')[1], s = 20, alpha = 0.6)
            ax[i_subplot].plot(ages_sorted, values_sorted, linestyle='-',
                     color=group_colors.get(group, 'gray'), alpha = 0.7)

        ax[i_subplot].set_xlabel("Age (Postnatal days)")
        ax[i_subplot].set_ylabel("# pixels")
        if i_subplot==0:
            ax[i_subplot].set_title("Visually responsive area")
        elif i_subplot ==1:
            ax[i_subplot].set_title("Largest negative spot")
        #ax[i_subplot].legend(fontsize=8, loc='best', ncol=2)
        legend_handles = [mpatches.Patch(color=group_colors[group], label=group) for group in group_names]
        ax[i_subplot].legend(handles=legend_handles, fontsize=8, loc='best')
    plt.show()


# for animal in [item for item in os.listdir(os.path.join(path))]:
#     for day in [item for item in os.listdir(os.path.join(path, animal)) if
#                 os.path.isdir(os.path.join(path, animal, item))]:
#         for subfile in [item for item in os.listdir(os.path.join(path, animal, day)) if
#                         os.path.isdir(os.path.join(path, animal, day, item))]:
#             print(animal, day, subfile)
#             config_path = fr'I:\retmap\{animal}\{day}\{subfile}\config.txt'
#             mode = "2"
#             run_retmap(config_path, mode)

# parser = argparse.ArgumentParser()
# parser.add_argument("config", help="File path to config file")
# parser.add_argument("-m", "--mode", help="Choose analysis mode: 1 - Create complex fields\t 2 - Create and plot sign map")
# args = parser.parse_args()
#
# run_retmap(args.config, args.mode)


if __name__ == '__main__':

    from retmap import *

    screen = 'big100'
    project = 'ethogram'  # restored or ethogram
    over_time = False
    path = r'J:\retmap'


    if project == 'ethogram':
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
            'EC_GNAT_06': ['20240923']
        }

        # animals_days = {'EC_phpeb_11': ['20241113'],
        #                 # 'EC_GCaMP6s_12': ['20250123'],
        #                 'EC_phpeb_13': ['20250123'],
        #                 'EC_GNAT_03': ['20240924'],
        #                 'EC_GNAT_05': ['20240924'],
        #                 'EC_GNAT_06': ['20240924'],
        #                 'EC_GNAT_04': ['20241004'],
        #                 }

    elif project == 'restored' and over_time:  # looking across rd1s and gnats, over time
        animals_days = {
            'EC_GCaMP6s_06': ['20241121', '20250408'],  # maybe remove day 1
            'EC_GCaMP6s_09': ['20241009', '20241121', '20250408'],
            'EC_GNAT_06': ['20240923', '20250404'],
            'EC_GNAT_03': ['20240923', '20250408'],
            # 'EC_GNAT_05': ['20240923', '20250408'],
            'EC_RD1_05': ['20240903', '20240918', '20241121'],
            'EC_RD1_06': ['20241001', '20241204', '20241218', '20250107', '20250408'],
            'EC_RD1_08': ['20241001', '20241204', '20241218', '20250107', '20250408'],
            'EC_RD1_09': ['20241006', '20241219', '20250109'],
            'EC_RD1_10': ['20241006', '20241219', '20250109', '20250408'],
        }
        animals_birthdate = {'EC_GCaMP6s_06': '20240624',
                             'EC_GCaMP6s_09': '20240414',
                             'EC_GNAT_06': '20240621',
                             'EC_GNAT_03': '20240621',
                             # 'EC_GNAT_05': '20240621',
                             'EC_RD1_05': '20240606',
                             'EC_RD1_06': '20240704',
                             # 'EC_RD1_07': '20240704',
                             'EC_RD1_08': '20240704',
                             'EC_RD1_09': '20240704',
                             'EC_RD1_10': '20240704'}
    elif project == 'restored':
        animals_days = {'EC_GCaMP6s_05': ['20240917'],
                        'EC_GCaMP6s_06': ['20241121'],
                        'EC_GCaMP6s_08': ['20240918'],
                        'EC_GCaMP6s_09': ['20241121'],
                        'EC_RD1_05': ['20241121'],
                        'EC_RD1_06': ['20250107'],
                        'EC_RD1_07': ['20241204'],
                        'EC_RD1_08': ['20250107'],
                        'EC_RD1_09': ['20250109'],
                        'EC_RD1_10': ['20250109'],
                        'EC_RD1opto_04': ['20241111'],
                        'EC_RD1opto_02': ['20241111'],
                        'EC_RD1opto_05': ['20241118'],
                        'EC_RD1opto_03': ['20241119'],
                        'EC_RD1opto_08': ['20250305'],
                        'EC_RD1opto_10': ['20250305'],
                        'EC_GNAT_03': ['20240923'],
                        'EC_GNAT_04': ['20241003'],
                        'EC_GNAT_05': ['20240923'],
                        'EC_GNAT_06': ['20250404']
                        }
        animals_birthdate = {'EC_GCaMP6s_05': '20240624',
                             'EC_GCaMP6s_06': '20240624',
                             'EC_GCaMP6s_08': '20240414',
                             'EC_GCaMP6s_09': '20240414',
                             'EC_GNAT_03': '20240621',
                             'EC_GNAT_04': '20240621',
                             'EC_GNAT_05': '20240621',
                             'EC_GNAT_06': '20240621',
                             'EC_RD1_05': '20240606',
                             'EC_RD1_06': '20240704',
                             'EC_RD1_07': '20240704',
                             'EC_RD1_08': '20240704',
                             'EC_RD1_09': '20240704',
                             'EC_RD1_10': '20240704',
                             'EC_RD1opto_04': '20240606',
                             'EC_RD1opto_02': '20240517',
                             'EC_RD1opto_05': '20240619',
                             'EC_RD1opto_03': '20240606',
                             'EC_RD1opto_08': '20240725',
                             'EC_RD1opto_10': '20240731',
                             }

    batch_retmap(animals_days, screen, path, draw_mask=False, draw_lines = False)

    # plt.close('all')

    # quantifying total visual area, largest negative patch, mean pixel amplitude > across diff groups
    total_visual_area_g, largest_neg_patch_area_g, mean_amplitude_g, largest_pos_patch_area_g, amp_threshold_g = {}, {}, {}, {}, {}
    for animal in animals_days:
        g = animal.split('_')[1]

        if (g not in total_visual_area_g):
            total_visual_area_g[g] = [total_visual_area[animal]]
            largest_neg_patch_area_g[g] = [largest_neg_patch_area[animal]]
            largest_pos_patch_area_g[g] = [largest_pos_patch_area[animal]]
            mean_amplitude_g[g] = [mean_amplitude[animal]]
            amp_threshold_g[g] = [amplitude_threshold[animal]]
        else:
            total_visual_area_g[g].append(total_visual_area[animal])
            largest_neg_patch_area_g[g].append(largest_neg_patch_area[animal])
            largest_pos_patch_area_g[g].append(largest_pos_patch_area[animal])
            mean_amplitude_g[g].append(mean_amplitude[animal])
            amp_threshold_g[g].append(amplitude_threshold[animal])

    fig, ax = plt.subplots (1,5, figsize = (14,4))
    group_names = ['phpeb', 'GNAT']
    for i_subplot, metric, in enumerate([total_visual_area_g, largest_neg_patch_area_g, mean_amplitude_g, largest_pos_patch_area_g, amp_threshold_g]):

        for i, group in enumerate(group_names):
            arr = np.array(metric[group])

            if group == 'phpeb':
                colour = '#EE2926'
            if group == 'GNAT':
                colour = '#477BBF'

            ax[i_subplot].scatter ([i]*len(arr) + np.random.uniform(-0.2, 0.2, len(arr)), arr, c = colour, alpha = 0.5, s = 30)
            ax[i_subplot].bar ([i], arr.mean(), color = colour, alpha = 0.6)

            if i_subplot == 0:
                ax[i_subplot].set_title ('total visual area')
                print('total visual area')
            elif i_subplot == 1:
                ax[i_subplot].set_title ('largest negative patch')
                print('larg neg patch')
            elif i_subplot == 2:
                ax[i_subplot].set_title ('mean pixel amplitude')
                print('mean pix amp')
            elif i_subplot == 3:
                ax[i_subplot].set_title ('largest positive patch')
                print('larg pos patch')
            elif i_subplot == 4:
                ax[i_subplot].set_title ('amplitude threshold')
                print('amp thresh')

        # significance tests

        # Exact two-sample Fisher-Pitman permutation test
        from scipy.stats import permutation_test
        from math import comb

        control = np.asarray(metric['phpeb'], dtype=float)
        gnat = np.asarray(metric['GNAT'], dtype=float)

        # Remove NaNs/infinite values
        control = control[np.isfinite(control)]
        gnat = gnat[np.isfinite(gnat)]

        def diff_means(x, y):
            return np.mean(x) - np.mean(y)

        n_control = len(control)
        n_gnat = len(gnat)
        n_perms = comb(n_control + n_gnat, n_control)

        print(
            f"\nMetric {i_subplot}: "
            f"n = {n_control} vs {n_gnat}, "
            f"{n_perms} possible assignments"
        )

        res = permutation_test(
            (control, gnat),
            diff_means,
            permutation_type='independent',
            n_resamples=np.inf,
            alternative='two-sided',
        )

        stat = res.statistic
        p = res.pvalue

        # Fisher-Pitman standardized Z statistic
        all_values = np.concatenate([control, gnat])
        n_total = n_control + n_gnat

        T = control.sum()
        E = n_control * all_values.mean()

        V = (
                (n_control * n_gnat)
                / (n_total * (n_total - 1))
                * ((all_values - all_values.mean()) ** 2).sum()
        )

        z = (T - E) / np.sqrt(V)

        print(
            f"Exact Fisher-Pitman permutation test: "
            f"mean difference = {stat:.4f}, "
            f"Z = {z:.3f}, "
            f"p = {p:.4f}"
        )

        ############################################
        # stat, p = kruskal(control, gnat)
        # print(f'KW H-statistic: {stat:.3f}, p-value: {p:.3f}')
        #
        # mw_pval = mannwhitneyu(control, gnat, alternative='two-sided').pvalue
        # print(f'Mann-Whitney p value {mw_pval}')

        # if p < 0.05:  # follow up with testing pairwise comparisons
        #     # Do pairwise comparisons manually:
        #     print('mannwhitney two-sided test')
        #     pvals = [mannwhitneyu(control, gnat, alternative='two-sided').pvalue]
        #
        #     print(pvals)
        #
        #     comparisons = [('GCaMP6s', 'GNAT')] # Define comparisons in same order as pvals
        #     group_coords = {'GCaMP6s': 0, 'GNAT':1}  # Coordinates for group positions on x-axis
        #
        #     max_val = max(np.max(control), np.max(gnat)) * 1.05
        #     step = max_val * 0.05  # vertical spacing between significance bars
        #     h = max_val
        #
        #     # Loop through each comparison and plot if significant
        #     for (group1, group2), p_val in zip(comparisons, pvals):
        #         if p_val < 0.05:
        #             x1, x2 = group_coords[group1], group_coords[group2]
        #             y = h
        #             h += step  # update height for next line if needed
        #
        #             print(x1, x2, y)
        #
        #             # plot the number of stars according to significance value
        #             if p_val < 0.001:
        #                 stars = '***'
        #             elif p_val < 0.01:
        #                 stars = '**'
        #             else:
        #                 stars = '*'
        #
        #             ax[i_subplot].plot([x1, x1, x2, x2], [y, y + step / 2, y + step / 2, y], lw=1.5, c='k')
        #             ax[i_subplot].text((x1 + x2) * 0.5, y + step * 0.6, stars,
        #                                ha='center', va='bottom', fontsize=14)
        # Add significance annotation
        if p < 0.05:
            if p < 0.001:
                stars = '***'
            elif p < 0.01:
                stars = '**'
            else:
                stars = '*'

            max_val = max(np.max(control), np.max(gnat))
            data_range = max_val - min(np.min(control), np.min(gnat))

            # Avoid zero spacing if all values are identical
            step = 0.08 * data_range if data_range > 0 else 1
            y = max_val + step

            ax[i_subplot].plot(
                [0, 0, 1, 1],
                [y, y + step / 2, y + step / 2, y],
                lw=1.5,
                c='k'
            )

            ax[i_subplot].text(
                0.5,
                y + step * 0.6,
                f'{stars}\np = {p:.3g}',
                ha='center',
                va='bottom',
                fontsize=12
            )

        ax[i_subplot].set_xticks(range(len(np.unique([animal[3:-3] for animal in animals_days.keys()]))))
        ax[i_subplot].set_xticklabels(['phpeb','gnat'])
        ax[i_subplot].set_ylabel('# pixels')
    plt.tight_layout()
    plt.show()

fig, ax = plt.subplots(1, 5, figsize=(14, 4))

group_names = ['phpeb', 'GNAT']
colors = {'phpeb': '#EE2926', 'GNAT': '#477BBF'}

metrics = [
    total_visual_area_g,
    largest_neg_patch_area_g,
    mean_amplitude_g,
    largest_pos_patch_area_g,
    amp_threshold_g
]

titles = [
    'total visual area',
    'largest negative patch',
    'mean pixel amplitude',
    'largest positive patch',
    'amplitude threshold'
]

for i_subplot, metric in enumerate(metrics):

    group_data = [np.array(metric[group]) for group in group_names]

    bp = ax[i_subplot].boxplot(
        group_data,
        positions=np.arange(len(group_names)),
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color='black', linewidth=2),
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5)
    )

    for i, group in enumerate(group_names):
        colour = colors[group]
        arr = np.array(metric[group])

        bp['boxes'][i].set(facecolor=colour, alpha=0.35, edgecolor=colour)

        bp['whiskers'][2*i].set(color=colour)
        bp['whiskers'][2*i + 1].set(color=colour)
        bp['caps'][2*i].set(color=colour)
        bp['caps'][2*i + 1].set(color=colour)

        ax[i_subplot].scatter(
            np.full(len(arr), i) + np.random.uniform(-0.15, 0.15, len(arr)),
            arr,
            c=colour,
            alpha=0.7,
            s=30,
            zorder=3
        )

    ax[i_subplot].set_title(titles[i_subplot])
    ax[i_subplot].set_xticks(np.arange(len(group_names)))
    ax[i_subplot].set_xticklabels(['phpeb', 'gnat'])
    ax[i_subplot].set_ylabel('# pixels')

    # Exact two-sample Fisher-Pitman permutation test
    from scipy.stats import permutation_test
    from math import comb

    control = np.asarray(metric['phpeb'], dtype=float)
    gnat = np.asarray(metric['GNAT'], dtype=float)

    # Remove NaNs/infinite values
    control = control[np.isfinite(control)]
    gnat = gnat[np.isfinite(gnat)]

    def diff_means(x, y):
        return np.mean(x) - np.mean(y)

    n_control = len(control)
    n_gnat = len(gnat)
    n_perms = comb(n_control + n_gnat, n_control)

    print(
        f"\nMetric {i_subplot}: "
        f"n = {n_control} vs {n_gnat}, "
        f"{n_perms} possible assignments"
    )

    res = permutation_test(
        (control, gnat),
        diff_means,
        permutation_type='independent',
        n_resamples=np.inf,
        alternative='two-sided',
    )

    stat = res.statistic
    p = res.pvalue

    # Fisher-Pitman standardized Z statistic
    all_values = np.concatenate([control, gnat])
    n_total = n_control + n_gnat

    T = control.sum()
    E = n_control * all_values.mean()

    V = (
            (n_control * n_gnat)
            / (n_total * (n_total - 1))
            * ((all_values - all_values.mean()) ** 2).sum()
    )

    z = (T - E) / np.sqrt(V)

    print(
        f"Exact Fisher-Pitman permutation test: "
        f"mean difference = {stat:.4f}, "
        f"Z = {z:.3f}, "
        f"p = {p:.4f}"
    )

    # stat, p = kruskal(control, gnat)
    # print(f'{titles[i_subplot]}')
    # print(f'KW H-statistic: {stat:.3f}, p-value: {p:.3f}')
    #
    # mw_pval = mannwhitneyu(control, gnat, alternative='two-sided').pvalue
    # print(f'Mann-Whitney p value {mw_pval}')

    if p < 0.05:
        y = max(np.max(control), np.max(gnat)) * 1.05
        step = y * 0.05

        if p < 0.001:
            stars = '***'
        elif p < 0.01:
            stars = '**'
        else:
            stars = '*'

        ax[i_subplot].plot([0, 0, 1, 1], [y, y + step, y + step, y], lw=1.5, c='k')
        ax[i_subplot].text(0.5, y + step * 1.1, stars,
                           ha='center', va='bottom', fontsize=14)

plt.tight_layout()
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu

group_names = ["phpeb", "GNAT"]

display_names = [
    "WT",
    r"GNAT1/2$^{mut}$",
]

colors = {
    "phpeb": "#ee2424",
    "GNAT": "#467bbe",
}


metrics = [
    total_visual_area_g,
    largest_neg_patch_area_g,
    mean_amplitude_g,
    largest_pos_patch_area_g,
    amp_threshold_g,
]

titles = [
    "Total visual area",
    "Largest negative patch",
    "Mean pixel amplitude",
    "Largest positive patch",
    "Amplitude threshold",
]

ylabels = [
    "Area (mm²)",
    "Area (mm²)",
    "Amplitude",
    "Area (mm²)",
    "Amplitude",
]

fig, axes = plt.subplots(
    1,
    5,
    figsize=(15, 4),
)

# Reproducible point jitter
rng = np.random.default_rng(42)


for i_subplot, (ax, metric, title, ylabel) in enumerate(
    zip(axes, metrics, titles, ylabels)
):

    control = np.asarray(metric["phpeb"], dtype=float)
    blind = np.asarray(metric["GNAT"], dtype=float)

    control = control[np.isfinite(control)]
    blind = blind[np.isfinite(blind)]

    group_data = [control, blind]

    # Bring the two groups closer together
    positions = np.array([0.0, 0.65])

    bp = ax.boxplot(
        group_data,
        positions=positions,

        # Slightly thicker boxes
        widths=0.30,

        patch_artist=True,
        showfliers=False,

        # Standard Tukey whiskers
        whis=1.8,

        medianprops=dict(
            color="black",
            linewidth=1.8,
        ),

        # Slightly thinner black box outline
        boxprops=dict(
            color="black",
            linewidth=1.8,
        ),

        whiskerprops=dict(
            color="black",
            linewidth=1.8,
        ),

        capprops=dict(
            color="black",
            linewidth=1.8,
        ),
    )

    for i_group, (group, arr) in enumerate(
        zip(group_names, group_data)
    ):

        colour = colors[group]

        # Solid box fill
        bp["boxes"][i_group].set_facecolor(colour)
        bp["boxes"][i_group].set_alpha(1)

        # Small horizontal jitter
        jitter = rng.uniform(
            -0.04,
            0.04,
            size=len(arr),
        )

        # Move points slightly to the left of each box
        point_offset = -0.22

        x_points = (
            np.full(len(arr), positions[i_group])
            + point_offset
            + jitter
        )

        ax.scatter(
            x_points,
            arr,

            color=colour,
            s=42,
            alpha=1,

            # No black outline
            edgecolors="none",

            zorder=3,
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------
    # Exact two-sample Fisher-Pitman permutation test
    from scipy.stats import permutation_test
    from math import comb

    gnat = blind

    def diff_means(x, y):
        return np.mean(x) - np.mean(y)

    n_control = len(control)
    n_gnat = len(gnat)
    n_perms = comb(n_control + n_gnat, n_control)

    print(
        f"\nMetric {i_subplot}: "
        f"n = {n_control} vs {n_gnat}, "
        f"{n_perms} possible assignments"
    )

    res = permutation_test(
        (control, gnat),
        diff_means,
        permutation_type='independent',
        n_resamples=np.inf,
        alternative='two-sided',
    )

    stat = res.statistic
    p = res.pvalue

    # Fisher-Pitman standardized Z statistic
    all_values = np.concatenate([control, gnat])
    n_total = n_control + n_gnat

    T = control.sum()
    E = n_control * all_values.mean()

    V = (
            (n_control * n_gnat)
            / (n_total * (n_total - 1))
            * ((all_values - all_values.mean()) ** 2).sum()
    )

    z = (T - E) / np.sqrt(V)

    print(
        f"Exact Fisher-Pitman permutation test: "
        f"mean difference = {stat:.4f}, "
        f"Z = {z:.3f}, "
        f"p = {p:.4f}"
    )

    all_values = np.concatenate(group_data)

    if len(all_values) > 0:
        data_min = np.min(all_values)
        data_max = np.max(all_values)
        data_range = data_max - data_min

        if data_range == 0:
            data_range = abs(data_max) if data_max != 0 else 1

        if p < 0.05:

            if p < 0.001:
                stars = "***"
            elif p < 0.01:
                stars = "**"
            else:
                stars = "*"

            bracket_bottom = data_max + 0.08 * data_range
            bracket_top = data_max + 0.14 * data_range

            ax.plot(
                [
                    positions[0],
                    positions[0],
                    positions[1],
                    positions[1],
                ],
                [
                    bracket_bottom,
                    bracket_top,
                    bracket_top,
                    bracket_bottom,
                ],
                color="black",
                linewidth=2,
                clip_on=False,
            )

            ax.text(
                np.mean(positions),
                bracket_top + 0.02 * data_range,
                stars,
                ha="center",
                va="bottom",
                fontsize=20,
                fontfamily="Arial",
                color="black",
            )

            ax.set_ylim(
                data_min - 0.05 * data_range,
                data_max + 0.25 * data_range,
            )

    ax.set_title(
        title,
        fontsize=19,
        fontfamily="Arial",
        color="black",
        pad=16,
    )

    ax.set_ylabel(
        ylabel,
        fontsize=20,
        fontfamily="Arial",
        color="black",
    )

    ax.set_xticks(positions)

    ax.set_xticklabels(
        display_names,
        fontsize=13,
        fontfamily="Arial",
        color="black",
    )

    # Keep the tighter group centered within each subplot
    ax.set_xlim(-0.45, 1.10)

    ax.set_facecolor("white")

    # Only show left spine
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color("black")
    ax.spines["left"].set_linewidth(2)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # No grid
    ax.grid(False)

    # Y-axis ticks pointing outward
    ax.tick_params(
        axis="y",
        direction="out",
        length=6,
        width=2,
        colors="black",
        labelsize=18,
    )

    # Remove x tick marks but retain labels
    ax.tick_params(
        axis="x",
        length=0,
        pad=12,
    )


fig.patch.set_facecolor("white")

plt.subplots_adjust(
    left=0.06,
    right=0.99,
    bottom=0.20,
    top=0.85,
    wspace=0.50,
)

# plt.savefig(
#     "visual_area_metrics.pdf",
#     format="pdf",
#     bbox_inches="tight",
# )
#
# plt.savefig(
#     "visual_area_metrics.png",
#     dpi=300,
#     bbox_inches="tight",
# )

plt.show(block=True)