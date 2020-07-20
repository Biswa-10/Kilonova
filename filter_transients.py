import numpy as np
from astropy.table import Table
from dataframe import Data
import matplotlib.pyplot as plt


def plot_distribution(y1, y2, label1, label2, colors=['b', 'g'], cut=None):
    # sets up the axis and gets histogram data
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.hist([y1, y2], color=colors)
    n, bins, patches = ax1.hist([y1, y2], bins=50)
    ax1.cla()

    # plots the histogram data
    width = (bins[1] - bins[0])
    bins_shifted = bins
    ax1.bar(bins[:-1], n[0], width, color=colors[0], alpha=.6, label=label1)
    ax2.bar(bins[:-1], n[1], width, color=colors[1], alpha=.6, label=label2)

    # finishes the plot
    ax1.set_ylabel(label1 + " Count", color=colors[0], fontsize=15)
    ax2.set_ylabel(label2 + " Count", color=colors[1], fontsize=15)
    ax1.tick_params('y', colors=colors[0])
    ax2.tick_params('y', colors=colors[1])
    ax1.set_xlabel("$\ln{\;(\;PP+1\;)}$", fontsize=15)
    if cut is not None:
        ax1.vlines(8, 0, 10, linestyles='--', label='cut')
    fig.legend()
    # fig.savefig("important_plots/distribution_of_kn_metric")

    plt.tight_layout()
    return fig


def calc_priodic_penalty(data_ob, object_df):
    flux_and_error_diff = np.abs(object_df[data_ob.flux_col_name]) - np.abs(object_df[data_ob.flux_err_col_name])
    flux_err_ratio = np.abs(object_df[data_ob.flux_col_name]) > 2 * object_df[data_ob.flux_err_col_name]
    index = (np.abs(flux_and_error_diff) > 10) & flux_err_ratio
    object_df = object_df[index]

    if len(object_df) == 0:
        penalty = 0

    else:

        max_flux_pos = np.argmax(object_df[data_ob.flux_col_name])

        # length =len(object_df['mjd'])
        max_flux_date = object_df[data_ob.time_col_name][max_flux_pos]
        max_flux_val = object_df[data_ob.flux_col_name][max_flux_pos]
        # print(object_df['flux_err'])

        time_from_max = np.abs(object_df[data_ob.time_col_name] - max_flux_date)
        time_from_max[np.where(time_from_max < 7)] = 0
        # filtered_flux = object_df['flux']
        length = np.sum(np.where(time_from_max >= 0))

        # normalization = np.sum(index_greater_than_half)
        penalty = np.sum(np.abs(object_df[data_ob.flux_col_name]) * np.abs(time_from_max)) / max_flux_val
        print(penalty)
        penalty = np.log(np.abs(penalty)+1)
    return penalty



def PLAsTiCC_transient_filter(data_ob, cut=8):
    object_ids = data_ob.get_all_object_ids()
    filter_result = np.zeros(len(object_ids), dtype=bool)
    for i, object_id in enumerate(object_ids):
        mask = data_ob.df_data[data_ob.object_id_col_name] == object_id
        object_df = data_ob.df_data[mask]

        penalty = calc_priodic_penalty(data_ob, object_df)
        max_per_band = []
        if np.log(penalty + 1) > cut:
            continue
        for band in range(6):
            band_mask = data_ob.df_data[data_ob.band_col_name] == band
            band_df = data_ob.df_data[band_mask * mask]
            if len(band_df) > 0:
                max_per_band.append(np.amax(band_df[data_ob.flux_col_name]))
            else:
                max_per_band.append(0)

        if (max_per_band[1] < 11000) & (max_per_band[2] < 10000) & (max_per_band[3] < 8000) & (
                max_per_band[4] < 8000) & (max_per_band[5] < 8000):
            filter_result[i] = True
    np.save("filter_result", filter_result)
    data_ob.df_metadata['filter_result'] = filter_result
    return filter_result


def transient_filter_load_saved(data_ob, path="filter_result.npy"):
    filter_result = np.load(path)
    data_ob.df_metadata['filter_result'] = filter_result
