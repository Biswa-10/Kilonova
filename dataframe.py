import numpy as np
from astropy.table import vstack
from Predict_lc import PredictLightCurve
from random import random
# change in future
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import pandas as pd


def sample_from_df(data_df, sample_numbers, shuffle=False):
    final_df = None
    data_df = data_df.sample(frac=1).reset_index(drop=True)
    for key, value in sample_numbers.items():
        if value == 0:
            continue
        current_type_df = data_df[data_df['type'] == key]
        len_current_df = len(current_type_df)
        if len_current_df == 0:
            print("event type not found")
        if value > len_current_df:
            value = len_current_df
            sample_numbers[key] = len_current_df
        current_type_df = current_type_df.sample(value)
        # print(current_type_df)
        if final_df is None:
            final_df = current_type_df
        else:
            final_df = pd.concat([final_df, current_type_df], ignore_index=True)
    if shuffle:
        final_df = final_df.sample(frac=1).reset_index(drop=True)
    return final_df, sample_numbers


class Data:

    def __init__(self, df_data, object_id_col_name, time_col_name, band_col_name, band_map, df_metadata=None,
                 mag_or_flux=0, bands=None, flux_col_name=None, flux_err_col_name=None, mag_col_name=None,
                 mag_err_col_name=None, target_col_name=None):

        self.object_id_col_name = object_id_col_name
        self.time_col_name = time_col_name
        self.flux_col_name = flux_col_name
        self.mag_col_name = mag_col_name
        self.flux_err_col_name = flux_err_col_name
        self.mag_err_col_name = mag_err_col_name
        self.band_col_name = band_col_name
        self.target_col_name = target_col_name
        self.df_metadata = df_metadata
        self.df_data = df_data
        self.band_map = band_map
        if bands is None:
            self.bands = list(band_map.keys())
        else:
            self.bands = bands

        self.features_df = None
        self.prediction_type_nos = None
        self.prediction_stat_df = None
        self.sample_numbers = None

        if mag_or_flux == 1:
            if mag_col_name is None:
                print("error")

        elif mag_or_flux == 0:
            if flux_col_name is None:
                print("error")

        else:
            print("error")

    def get_all_object_ids(self):
        if self.df_metadata is not None:
            return self.df_metadata[self.object_id_col_name]
        else:
            return np.unique(self.df_data[self.object_id_col_name])

    def get_ids_of_event_type(self, target):
        if isinstance(target, int):
            if self.target_col_name is None:
                print("Target name not given")
            else:
                event = self.df_metadata[self.target_col_name]
                index = event == target
                object_ids = self.get_all_object_ids()
                class_ids = object_ids[index]
        else:
            class_ids = None

            for target_id in target:
                event = self.df_metadata[self.target_col_name]
                index = event == target_id
                object_ids = self.get_all_object_ids()
                if class_ids is None:
                    class_ids = object_ids[index]
                else:
                    class_ids = vstack([class_ids, object_ids[index]])

        return class_ids

    def get_data_of_event(self, target):
        index = self.df_data[self.object_id_col_name] == target
        return self.df_data[index]

    def get_band_data(self, band):
        index = self.df_data[self.band_col_name] == band
        return self.df_data[index]

    def get_object_type_number(self, object_id):
        if self.target_col_name is None:
            print("Target name not given")
        object_type = np.array(
            self.df_metadata[self.target_col_name][np.argwhere(self.df_metadata[self.object_id_col_name] == object_id)])
        return object_type[0][0]

    def get_object_type_for_PLAsTiCC(self, object_id):
        if self.target_col_name is None:
            print("Target name not given")
        object_type = np.array(
            self.df_metadata[self.target_col_name][np.argwhere(self.df_metadata[self.object_id_col_name] == object_id)])
        object_num = object_type[0][0]
        if object_num == 90:
            return "SN-Ia"
        elif object_num == 67:
            return "SN-Ia-91bg"
        elif object_num == 52:
            return "SN-Iax"
        elif object_num == 42:
            return "SNII"
        elif object_num == 62:
            return "SNIbc"
        elif object_num == 95:
            return "SLSN-I"
        elif object_num == 15:
            return "TDE"
        elif object_num == 64:
            return "KN"
        elif object_num == 88:
            return "AGN"
        elif object_num == 92:
            return "RRL"
        elif object_num == 65:
            return "M-dwarf"
        elif object_num == 16:
            return "EB"

        elif object_num == 53:
            return "Mira"
        elif object_num == 6:
            return "micro-lens Single"
        else:
            return "unknown"

    def is_transient(self, object_id):
        if self.target_col_name is None:
            print("Target name not given")
        object_num = np.array(
            self.df_metadata[self.target_col_name][np.argwhere(self.df_metadata[self.object_id_col_name] == object_id)])
        object_num = object_num[0][0]
        if (object_num == 90) | (object_num == 67) | (object_num == 52) | (object_num == 42) | (object_num == 62) | (
                object_num == 95) | (object_num == 15) | (object_num == 64) | (object_num == 65):
            return 1
        elif (object_num == 88) | (object_num == 92) | (object_num == 16) | (object_num == 53) | (object_num == 6):
            return 0
        else:
            return None

    def add_y_val(self):
        self.features_df['y_true'] = self.features_df['id'].map(
            lambda ob_id: 1 if self.get_object_type_number(ob_id) in self.prediction_type_nos else 0)
        return self.features_df

    def create_features_df(self, prediction_type_nos, features_path=None, sample_numbers=None,
                           decouple_prediction_bands=True,
                           decouple_pc_bands=False, mark_maximum=False, min_flux_threshold=20, num_pc_components=3,
                           color_band_dict=None, use_random_current_date=False, plot_prediction=False):

        if isinstance(prediction_type_nos, int):
            self.prediction_type_nos = [prediction_type_nos]
        else:
            self.prediction_type_nos = prediction_type_nos
            self.prediction_type_nos.sort()

        if features_path is None:
            data_dict = {'id': [],
                         'type': [], }

            object_ids = self.data_ob.get_all_object_ids()
            # data_object_ids = np.random.permutation(data_object_ids)
            self.data_ob.df_data.sort([self.data_ob.object_id_col_name, self.data_ob.time_col_name])
            for object_id in tqdm(object_ids):
                pc = PredictLightCurve(self.data_ob, object_id=object_id)
                current_date = None
                if use_random_current_date:
                    median_date = np.median(pc.lc.dates_of_maximum)
                    current_date = median_date + random() * 50 - 25

                coeff_dict, num_pts_dict = pc.predict_lc_coeff(current_date=current_date,
                                                               num_pc_components=num_pc_components,
                                                               decouple_pc_bands=decouple_pc_bands,
                                                               decouple_prediction_bands=decouple_prediction_bands,
                                                               min_flux_threshold=min_flux_threshold, bands=self.bands)
                data_dict['id'].append(object_id)
                for i, band in enumerate(self.bands):
                    for j in range(1, num_pc_components + 1):
                        col_name = str(i) + 'pc' + str(j)
                        if col_name not in data_dict.keys():
                            data_dict[col_name] = []
                        data_dict[col_name].append(coeff_dict[band][j - 1])
                    col_name = str(i) + 'n'
                    if col_name not in data_dict.keys():
                        data_dict[col_name] = []
                    data_dict[col_name].append(num_pts_dict[band])
                object_type = self.data_ob.get_object_type_number(object_id)
                data_dict['type'].append(object_type)
                if plot_prediction:
                    fig = pc.plot_predicted_bands(all_band_coeff_dict=coeff_dict, color_band_dict=color_band_dict,
                                                  mark_maximum=mark_maximum, axes_lims=False)
                    plt.show()
                    plt.close('all')

            data_df = pd.DataFrame(data_dict)
            data_df = data_df.sample(frac=1).reset_index(drop=True)
            temp_df = data_df
        else:
            temp_df = pd.read_csv(features_path)

        self.features_df, sample_numbers = sample_from_df(temp_df, sample_numbers, shuffle=True)
        self.sample_numbers = sample_numbers
        self.add_y_val()

