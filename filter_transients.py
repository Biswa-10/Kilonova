import numpy as np
from astropy.table import Table
from dataframe import Data


def calc_priodic_penalty(data_ob, object_df):
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
        # print(penalty)

    return penalty


def PLAsTiCC_transient_filter(data_ob):
    object_ids = data_ob.get_all_object_ids()
    filter_result = np.zeros(len(object_ids), dtype=bool)
    for i, object_id in enumerate(object_ids):
        mask = data_ob.df_data[data_ob.object_id_col_name] == object_id
        object_df = data_ob.df_data[mask]
        flux_and_error_diff = np.abs(object_df[data_ob.flux_col_name]) - np.abs(object_df[data_ob.flux_err_col_name])
        flux_err_ratio = np.abs(object_df[data_ob.flux_col_name]) > 2 * object_df[data_ob.flux_err_col_name]
        index = (np.abs(flux_and_error_diff) > 10) & flux_err_ratio
        object_df = object_df[index]
        penalty = calc_priodic_penalty(data_ob, object_df)
        max_per_band = []
        if np.log(penalty+1) > 8:
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
