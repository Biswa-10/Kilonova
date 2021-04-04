import numpy as np
from astropy.table import vstack
from src.Predict_lc import PredictLightCurve
from random import random
# change in future
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import pandas as pd


def sample_from_df(data_df, sample_numbers=None, shuffle=False):
    """
    samples a fixed number of objects of each time, based on sample_numbers

    :param data_df: data for all the events from which to sample
    :param sample_numbers: a dict with keys as event types and values as numbrer of objects of that type to be sampled.
        if not passed, the dict will be updated with number of events of each type.
    :param shuffle: shuffle the output (if false then objects of same type will be placed together)
    ????
    :return: final_df: data after sampling required number of events
             sample_numbers: final dict with number of events of each type
    """
    if sample_numbers is None:
        sample_numbers = {}
        keys = np.unique(data_df['type'])
        for key in keys:
            val = np.sum(data_df['type'] == key)
            sample_numbers[key] = val
        return data_df, sample_numbers
    else:
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
            if value > 0:
                current_type_df = current_type_df.sample(value)
                # print(current_type_df)
                if final_df is None:
                    final_df = current_type_df
                else:
                    final_df = pd.concat([final_df, current_type_df], ignore_index=True)
        if shuffle:
            final_df = final_df.sample(frac=1).reset_index(drop=True)
        return final_df, sample_numbers


# TODO: change names

class Data:
    """
    class that stores the data of all light curves. This class helps to code to adapt to different datasets
    :param df_data: data of with mjd and flux values
    :param object_id_col_name: name of the column containing object ids
    :param time_col_name: name of column storing time data
    :param band_col_name: name of column with band/filter/channel information
    :param band_map: used to replace the names of bands in plots (Todo: necessary?)
    :param df_metadata: ????
    :param mag_or_flux: to use mag_or_flux (Todo: remove this option and use brightness_col_name)
    :param bands: list of bands used to be used for predictions
    :param flux_col_name: column name of the col that stores flux values
    :param flux_err_col_name: column name of the col that stores flux error values
    :param mag_col_name: column name of the col that stores magnitude values
    :param mag_err_col_name: column name of the col that stores magnitude error values
    :param target_col_name: column name of the col that stores event type (None if not available)
    """

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
        self.num_pc_components = None

        if mag_or_flux == 1:
            if mag_col_name is None:
                print("error")

        elif mag_or_flux == 0:
            if flux_col_name is None:
                print("error")

        else:
            print("error")

    def get_all_object_ids(self):
        """
        :return: np array with all object ids
        """
        if self.df_metadata is not None:
            return np.array(self.df_metadata[self.object_id_col_name])
        else:
            return np.unique(np.array(self.df_data[self.object_id_col_name]))

    def get_ids_of_event_type(self, target):
        """
        :param target: event types whose ids we want to extract
        :return: numpy array with list of ids
        """
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

    def get_data_of_event(self, object_id):
        """
        :param object_id: object id of the event that we want to extract
        :return: data of the required event
        """
        index = self.df_data[self.object_id_col_name] == object_id
        return self.df_data[index]

    def get_band_data(self, band):
        """
        :param band: band whose events we want ot extract
        :return: data of events belonging to a particular band
        """
        index = self.df_data[self.band_col_name] == band
        return self.df_data[index]

    def get_object_type_number(self, object_id):
        """
        :param object_id: object id whose event type we want o extract
        :return: event type of the object selected
        """
        if self.target_col_name is None:
            print("Target name not given")
        index = self.df_metadata[self.object_id_col_name] == object_id
        object_type = np.array(self.df_metadata[self.target_col_name][index])
        # print(object_type)
        return object_type[0]

    def get_object_type_for_PLAsTiCC(self, object_id):
        """
        :param object_id: object id whose event type we want o extract
        :return: event type of the object selected in string format
        """
        if self.target_col_name is None:
            print("Target name not given")
        index = self.df_metadata[self.object_id_col_name] == object_id
        object_num = np.array(self.df_metadata[self.target_col_name][index])
        object_num = object_num[0]
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
        """
        :param object_id: object id of the object concerned
        :return: if the object is transient or not
        """
        if self.target_col_name is None:
            print("Target name not given")
        object_num = np.array(
            self.df_metadata[self.target_col_name][
                np.argwhere(self.df_metadata[self.object_id_col_name] == object_id)])
        object_num = object_num[0][0]
        if (object_num == 90) | (object_num == 67) | (object_num == 52) | (object_num == 42) | (object_num == 62) | (
                object_num == 95) | (object_num == 15) | (object_num == 64) | (object_num == 65):
            return 1
        elif (object_num == 88) | (object_num == 92) | (object_num == 16) | (object_num == 53) | (object_num == 6):
            return 0
        else:
            return None

    def add_y_val(self):
        """
        adds label to the features dataset, 1 if the object type is in the list of prediction types

        :return: features data with added column for label.
        """
        self.features_df['y_true'] = self.features_df['id'].map(
            lambda ob_id: 1 if self.get_object_type_number(ob_id) in self.prediction_type_nos else 0)
        return self.features_df

    def create_features_df(self, prediction_type_nos, features_path=None, sample_numbers=None,
                           decouple_prediction_bands=True,
                           decouple_pc_bands=False, mark_maximum=False, min_flux_threshold=20, num_pc_components=3,
                           color_band_dict=None, use_random_current_date=False, plot_predicted_curve_of_type=None,
                           plot_all_predictions=False, band_choice='z', save_fig_path=None, classifier=None,
                           num_alert_days=None, skip_random_date_event_types=None):

        """
        creates a pandas dataframe with the features.
        Note that this function makes predictions and plots for all the data! [Todo: break down the plot part]

        :param prediction_type_nos: list with the type numbers to be identified
        :param features_path: if the features are already calculated, add path to the saved features
        :param sample_numbers: dict value that stores the number of events to be sampled corresponding to each type
        :param decouple_prediction_bands: if True (recommended) each band will have its own midpoint. If False, the date
            of mid point will be the same for all bands
        :param decouple_pc_bands: use different sets of PCs for different bands (should be used only for LSST data)
        :param mark_maximum: mark the maximum recorded flux of each band on plots
        :param min_flux_threshold: minimum value of the amplitude necessary to make predictions
        :param num_pc_components: number of PC components to be used
        :param color_band_dict: dict with the different bands and the corresponding colors to be used to make plots
        :param use_random_current_date: Try to mimic alerts by choosing selecting data only upto a specific date for
            making plots
        :param plot_predicted_curve_of_type: plot predictions of curves only of certain types
        :param plot_all_predictions: plot predictions of all data types
        :param band_choice: bands on which fit it to be generated
        :param save_fig_path: path in which the plots are to be saved
        :param classifier: trained classified to be used for making predictions (will call classifier.predict)
        :param num_alert_days: the number of days of data to be considered for making fits (can be used in combination
            with random_current_date
        :param skip_random_date_event_types: ?? [Todo: remove this]
        :return:
        """

        if plot_all_predictions:
            plot_predicted_curve_of_type = np.unique(self.df_metadata[self.target_col_name])

        if isinstance(prediction_type_nos, int):
            self.prediction_type_nos = [prediction_type_nos]
        else:
            self.prediction_type_nos = prediction_type_nos
            self.prediction_type_nos.sort()

        self.num_pc_components = num_pc_components

        if features_path is None:
            data_dict = {'id': [],
                         'type': [], }

            object_ids = self.get_all_object_ids()
            # data_object_ids = np.random.permutation(data_object_ids)
            self.df_data.sort([self.object_id_col_name, self.time_col_name])

            current_dates = []
            for object_id in tqdm(object_ids):
                pc = PredictLightCurve(self, object_id=object_id)
                current_date = None
                if use_random_current_date:
                    if skip_random_date_event_types is not None:
                        if self.get_object_type_number(object_id) not in skip_random_date_event_types:
                            current_min = np.amin(pc.lc.df[self.time_col_name])
                            current_max = np.amax(pc.lc.df[self.time_col_name])
                            current_date = int(random() * (current_max - current_min) + current_min)

                    else:
                        # median_date = np.median(pc.lc.dates_of_maximum)
                        # current_date = median_date + random() * 50 - 25
                        current_min = np.amin(pc.lc.df[self.time_col_name])
                        current_max = np.amax(pc.lc.df[self.time_col_name])
                        current_date = int(random() * (current_max - current_min) + current_min)
                current_dates.append(current_date)

                coeff_dict, num_pts_dict = pc.predict_lc_coeff(current_date=current_date,
                                                               num_pc_components=num_pc_components,
                                                               decouple_pc_bands=decouple_pc_bands,
                                                               decouple_prediction_bands=decouple_prediction_bands,
                                                               min_flux_threshold=min_flux_threshold, bands=self.bands,
                                                               band_choice=band_choice,
                                                               num_alert_days=num_alert_days)

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
                object_type = self.get_object_type_number(object_id)

                data_dict['type'].append(object_type)

                if plot_predicted_curve_of_type is not None:

                    if color_band_dict is None:
                        print("error: pass color of each band")

                    if object_type in plot_predicted_curve_of_type:
                        if classifier is not None:
                            coeff_list = [[]]
                            for band in self.bands:
                                coeff_list[0].extend(coeff_dict[band])
                            correct_pred = ((classifier.predict(coeff_list)[0] == 1) & (
                                    object_type in self.prediction_type_nos)) | (
                                                   (classifier.predict(coeff_list)[0] == 0) & (
                                                   object_type not in self.prediction_type_nos))
                            print(correct_pred)
                            print("----------------------------------------------------------------------")
                            if not correct_pred:
                                fig = pc.plot_predicted_bands(all_band_coeff_dict=coeff_dict,
                                                              color_band_dict=color_band_dict,
                                                              mark_maximum=mark_maximum, axes_lims=False,
                                                              object_name=str(object_type))
                                if save_fig_path is not None:
                                    fig.savefig(save_fig_path + "incorrect/" + str(object_type) + "_" + str(object_id))
                            else:
                                fig = pc.plot_predicted_bands(all_band_coeff_dict=coeff_dict,
                                                              color_band_dict=color_band_dict,
                                                              mark_maximum=mark_maximum, axes_lims=False,
                                                              object_name=str(object_type))
                                if save_fig_path is not None:
                                    fig.savefig(save_fig_path + "correct/" + str(object_type) + "_" + str(object_id))

                        else:
                            fig = pc.plot_predicted_bands(all_band_coeff_dict=coeff_dict,
                                                          color_band_dict=color_band_dict,
                                                          mark_maximum=mark_maximum, axes_lims=False,
                                                          object_name=str(object_type))
                            if save_fig_path is not None:
                                fig.savefig(save_fig_path + str(object_type) + "_" + str(object_id))
                        plt.show()
                        plt.close('all')

            if use_random_current_date:
                data_dict['curr_date'] = np.asarray(current_dates)

            data_df = pd.DataFrame(data_dict)
            data_df = data_df.sample(frac=1).reset_index(drop=True)
            temp_df = data_df
        else:
            temp_df = pd.read_csv(features_path)

        # Todo: sample before fitting!!!
        self.features_df, sample_numbers = sample_from_df(temp_df, sample_numbers, shuffle=True)
        self.sample_numbers = sample_numbers
        self.add_y_val()

    def plot_features_correlation_helper(self, class_features_df, bands=None, color_band_dict=None, fig=None,
                                         x_limits=None, y_limits=None, mark_xlabel=False, mark_ylabel=False,
                                         band_map=None, set_ax_title=False, label=""):
        """
        plots correlations between PCs of each band for only 1 class of data: ex KN

        :param class_features_df: dataframe of events of current class (KN and non-KN)
        :param bands: bands for which plots are to be generated
        :param color_band_dict: colors to be used for corresponding bands
        :param fig: fig on which plot is generated. If None, new fig is created
        :param x_limits: x limits of the plot
        :param y_limits: y limits of the plot
        :param mark_xlabel: mark x label or not
        :param mark_ylabel: to mark y label or not
        :param band_map: renaming bands/filter/channel name in plots
        :param set_ax_title: title of the axes ojbect on which plot is made
        :param label: string label of the current class (ex "KN" or "non-KN")
        :return: figure with the plots
        """
        num_rows = len(self.bands)
        num_cols = int(self.num_pc_components * (self.num_pc_components - 1) / 2)

        if fig is None:
            fig, axs = plt.subplots(num_rows, num_cols, figsize=(self.num_pc_components * 5, len(self.bands) * 5))
            # fig.subplots_adjust(wspace=.5,hspace=.5)
            ax_list = fig.axes
        else:
            ax_list = fig.axes

        if self.bands is None:
            bands = self.band_map.keys()

        for i, band in enumerate(bands):
            for x in range(self.num_pc_components):
                for y in range(x):
                    ax_current = ax_list[int(i * num_cols + (x - 1) * (x) / 2 + y)]

                    colx_name = str(i) + "pc" + str(x + 1)
                    coly_name = str(i) + "pc" + str(y + 1)
                    if mark_xlabel: ax_current.set_xlabel("PC" + str(x + 1), fontsize=20)
                    if mark_ylabel: ax_current.set_ylabel("PC" + str(y + 1), fontsize=20)

                    PCx = class_features_df[colx_name].values
                    PCy = class_features_df[coly_name].values

                    if color_band_dict is not None:
                        ax_current.scatter(PCx, PCy, color=color_band_dict[band], alpha=.5, label=label)
                    else:
                        # TODO: change the yellow
                        ax_current.scatter(PCx, PCy, color="yellow", alpha=.4, label=label)

                    if x_limits is not None: ax_current.set_xlim(x_limits)
                    if y_limits is not None: ax_current.set_ylim(y_limits)

                    if set_ax_title:
                        if band_map is None:
                            ax_current.set_title("PCs for " + str(band) + "-band", fontsize=20)
                        else:
                            ax_current.set_title("PCs for " + str(band_map[band]) + "-band", fontsize=20)
                    if label != "":
                        ax_current.legend(loc="upper right")
                    ax_current.set_aspect('equal', 'box')
        fig.tight_layout()
        return fig

    def plot_features_correlation(self, color_band_dict, bands=None,
                                  x_limits=None, y_limits=None, mark_xlabel=True, mark_ylabel=True, band_map=None,
                                  set_ax_title=True):
        """
        plots correlations between the PCs of each band

        :param class_features_df: dataframe of events of current class (KN and non-KN)
        :param bands: bands for which plots are to be generated
        :param color_band_dict: colors to be used for corresponding bands
        :param x_limits: x limits of the plot
        :param y_limits: y limits of the plot
        :param mark_xlabel: mark x label or not
        :param mark_ylabel: to mark y label or not
        :param band_map: renaming bands/filter/channel name in plots
        :param set_ax_title: title of the axes ojbect on which plot is made
        :return: figure with the plots
        """
        kn_df = self.features_df[self.features_df['y_true'] == 1]
        non_kn_df = self.features_df[self.features_df['y_true'] == 0]
        if bands is None:
            bands = self.bands

        num_rows = len(bands)
        num_cols = int(self.num_pc_components * (self.num_pc_components - 1) / 2)

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(self.num_pc_components * 5, len(bands) * 5))
        # fig.subplots_adjust(wspace=.5,hspace=.5)

        self.plot_features_correlation_helper(non_kn_df, fig=fig, band_map=band_map,
                                              color_band_dict=None, bands=bands, x_limits=x_limits, y_limits=y_limits,
                                              mark_xlabel=mark_xlabel, mark_ylabel=mark_ylabel,
                                              set_ax_title=set_ax_title,
                                              label="non-KN")
        self.plot_features_correlation_helper(kn_df, fig=fig, band_map=band_map,
                                              color_band_dict=color_band_dict, bands=bands, x_limits=x_limits,
                                              y_limits=y_limits, mark_xlabel=mark_xlabel, mark_ylabel=mark_ylabel,
                                              set_ax_title=set_ax_title, label="KN")

        return fig

    def plot_band_correlation_helper(self, current_class_df, bands, color_band_dict=None, fig=None,
                                     x_limits=None, y_limits=None, mark_xlabel=False, mark_ylabel=False,
                                     band_map=None, set_ax_title=False, label=""):
        """

        :param current_class_df: dataframe of events of current class (KN and non-KN)
        :param bands: bands for which plots are to be generated
        :param color_band_dict: colors to be used for corresponding bands
        :param fig: fig on which plot is generated. If None, new fig is created
        :param x_limits: x limits of the plot
        :param y_limits: y limits of the plot
        :param mark_xlabel: mark x label or not
        :param mark_ylabel: to mark y label or not
        :param band_map: renaming bands/filter/channel name in plots
        :param set_ax_title: title of the axes ojbect on which plot is made
        :param label: string label of the current class (ex "KN" or "non-KN")
        :return: figure with the plots
        """
        num_rows = int(len(bands) * (len(bands) - 1) / 2)
        num_cols = self.num_pc_components
        if fig is None:
            fig, axs = plt.subplots(num_rows, num_cols, figsize=(self.num_pc_components * 5, len(bands) * 5))
            fig.subplots_adjust(wspace=.5, hspace=.5)
            ax_list = fig.axes
        else:
            ax_list = fig.axes

        for i in range(self.num_pc_components):
            # print("pc "+str(i))
            for x, band in enumerate(bands):
                for y in range(x):

                    x_band = bands[x]
                    y_band = bands[y]
                    # print("x " +str(x))
                    # print("y " +str(y))
                    # print(int(i*len(bands)*(len(bands)-1)/2 + (x-1)*(x-2)/2 +y))
                    ax_current = ax_list[int(i * num_rows + (x - 1) * (x - 2) / 2 + y)]

                    colx_name = str(x) + "pc" + str(i + 1)
                    coly_name = str(y) + "pc" + str(i + 1)

                    # print(coeff_plot_data)
                    PCx = current_class_df[colx_name].values
                    PCy = current_class_df[coly_name].values

                    if color_band_dict is not None:
                        ax_current.scatter(PCx, PCy, color=color_band_dict[band], alpha=.5, label=label)
                    else:
                        ax_current.scatter(PCx, PCy, color="yellow", alpha=.4, label=label)

                    if x_limits is not None: ax_current.set_xlim(x_limits)
                    if y_limits is not None: ax_current.set_ylim(y_limits)
                    if band_map is None:
                        if mark_xlabel: ax_current.set_xlabel(x_band + " band", fontsize=20)
                        if mark_ylabel: ax_current.set_ylabel(y_band + " band", fontsize=20)
                    else:
                        if mark_xlabel: ax_current.set_xlabel(band_map[x_band] + " band", fontsize=20)
                        if mark_ylabel: ax_current.set_ylabel(band_map[y_band] + " band", fontsize=20)
                    if set_ax_title: ax_current.set_title("correlation for PC" + str(i + 1), fontsize=20)
                    if label != "":
                        ax_current.legend(loc="upper right")
                    ax_current.set_aspect('equal', 'box')

        fig.tight_layout()
        return fig

    def plot_band_correlation(self, bands=None, color_band_dict=None, x_limits=None,
                              y_limits=None, mark_xlabel=True, mark_ylabel=True, band_map=None, set_ax_title=True):
        """
        plots correlations between 2 bands for each PC

        :param bands: bands among which correlation is to be plotted
        :param color_band_dict: colors to be used for corresponding bands
        :param fig: fig on which plot is generated. If None, new fig is created
        :param x_limits: x limits of the plot
        :param y_limits: y limits of the plot
        :param mark_xlabel: mark x label or not
        :param mark_ylabel: to mark y label or not
        :param band_map: renaming bands/filter/channel name in plots
        :param set_ax_title: title of the axes ojbect on which plot is made
        :return: figure on which the correlations are plotted
        """
        if bands is None:
            bands = self.bands
        kn_df = self.features_df[self.features_df['y_true'] == 1]
        non_kn_df = self.features_df[self.features_df['y_true'] == 0]

        num_rows = int(len(bands) * (len(bands) - 1) / 2)
        num_cols = self.num_pc_components
        space_between_axes = 0.0

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5))
        # fig.subplots_adjust(wspace=space_between_axes,hspace=space_between_axes)
        self.plot_band_correlation_helper(non_kn_df, bands=bands, fig=fig,
                                          color_band_dict=None, band_map=band_map, x_limits=x_limits, y_limits=y_limits,
                                          mark_xlabel=mark_xlabel, mark_ylabel=mark_ylabel, set_ax_title=set_ax_title,
                                          label="non-KN")
        self.plot_band_correlation_helper(kn_df, bands=bands, fig=fig,
                                          color_band_dict=color_band_dict, band_map=band_map, x_limits=x_limits,
                                          y_limits=y_limits, mark_xlabel=mark_xlabel, mark_ylabel=mark_ylabel,
                                          set_ax_title=set_ax_title, label="KN")
        # plt.xlabel(" correlation ")

        return fig

    def discard_no_featues_events(self):
        """
        discards events if no fit it generated (minimum threshold is not crossed or no data points) have been made
        """
        col_names = []
        non_zero_index = np.ones((len(self.features_df)), dtype='bool')
        for i in range(self.num_pc_components):
            for j in range(len(self.bands)):
                col_name = str(j) + 'pc' + str(i + 1)
                non_zero_index = (non_zero_index) & (self.features_df[col_name] != 0)

        self.features_df = self.features_df[non_zero_index]
        self.features_df.reset_index(drop=True, inplace=True)
        self.df_data = self.df_data[np.isin(self.df_data[self.object_id_col_name], self.features_df['id'])]
        self.df_metadata = self.df_metadata[np.isin(self.df_metadata[self.object_id_col_name], self.features_df['id'])]
        _, self.sample_numbers = sample_from_df(self.features_df, self.sample_numbers)
