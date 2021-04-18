from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.dataframe import Data
from src.Predict_lc import PredictLightCurve, calc_prediction, get_pcs
from src.LightCurve import LightCurve
from tqdm.notebook import tqdm
from random import random


def sample_from_df(data_df, sample_numbers=None, shuffle=False):
    """
    samples a fixed number of objects of each type, based on sample_numbers.

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


class RFModel:

    def __init__(self, prediction_type_nos, sample_numbers_train=None,
                 sample_numbers_test=None, decouple_prediction_bands=True, decouple_pc_bands=False,
                 min_flux_threshold=200, num_pc_components=3, use_random_current_date=False, bands=None,
                 band_choice='u', num_alert_days=None, skip_random_date_event_types=None, train_features_path=None,
                 test_features_path=None):

        self.train_data_ob = None
        self.test_data_ob = None
        self.train_features_path = train_features_path
        self.test_features_path = test_features_path
        if isinstance(prediction_type_nos, int):
            self.prediction_type_nos = [prediction_type_nos]
        else:
            self.prediction_type_nos = prediction_type_nos
            self.prediction_type_nos.sort()
        self.train_features_df = None
        self.test_features_df = None
        self.classifier = RandomForestClassifier(n_estimators=30, max_depth=13)
        self.sample_numbers_train = sample_numbers_train
        self.sample_numbers_test = sample_numbers_test
        self.decouple_prediction_bands = decouple_prediction_bands
        self.decouple_pc_bands = decouple_pc_bands
        self.min_flux_threshold = min_flux_threshold
        self.num_pc_components = num_pc_components
        self.use_random_current_date = use_random_current_date
        self.band_choice = band_choice
        self.bands = bands
        self.num_alert_days = num_alert_days
        self.skip_random_date_event_types = skip_random_date_event_types
        self.classifier_trained = False
        self.use_number_of_points_per_band = False

    def create_features_df(self, data_set, data_ob: Data = None, color_band_dict=None, plot_all_predictions=True,
                           mark_maximum=False, plot_predicted_curve_of_type=None, save_fig_path=None):
        features_path = None
        if data_set == 'train':
            features_path = self.train_features_path

        elif data_set == 'test':
            features_path = self.test_features_path

        # Todo: handle file exception
        if features_path is not None:
            temp_df = pd.read_csv(features_path)
        else:
            data_dict = {'id': [],
                         'type': []}
            object_ids = data_ob.get_all_object_ids()
            data_ob.df_data.sort([data_ob.object_id_col_name, data_ob.time_col_name])
            current_dates = []
            for object_id in tqdm(object_ids):
                pc = PredictLightCurve(data_ob, object_id=object_id)
                current_date = None
                if self.use_random_current_date:
                    if self.skip_random_date_event_types is not None:
                        if data_ob.get_object_type_number(object_id) not in self.skip_random_date_event_types:
                            current_min = np.amin(pc.lc.df[data_ob.time_col_name])
                            current_max = np.amax(pc.lc.df[data_ob.time_col_name])
                            current_date = int(random() * (current_max - current_min) + current_min)
                    else:
                        # median_date = np.median(pc.lc.dates_of_maximum)
                        # current_date = median_date + random() * 50 - 25
                        current_min = np.amin(pc.lc.df[data_ob.time_col_name])
                        current_max = np.amax(pc.lc.df[data_ob.time_col_name])
                        current_date = int(random() * (current_max - current_min) + current_min)
                current_dates.append(current_date)
                coeff_dict, num_pts_dict = pc.predict_lc_coeff(current_date=current_date,
                                                               num_pc_components=self.num_pc_components,
                                                               decouple_pc_bands=self.decouple_pc_bands,
                                                               decouple_prediction_bands=self.decouple_prediction_bands,
                                                               min_flux_threshold=self.min_flux_threshold,
                                                               bands=self.bands,
                                                               band_choice=self.band_choice,
                                                               num_alert_days=self.num_alert_days)
                data_dict['id'].append(object_id)
                for i, band in enumerate(self.bands):
                    for j in range(1, self.num_pc_components + 1):
                        col_name = str(i) + 'pc' + str(j)
                        if col_name not in data_dict.keys():
                            data_dict[col_name] = []
                        data_dict[col_name].append(coeff_dict[band][j - 1])

                    col_name = str(i) + 'n'
                    if col_name not in data_dict.keys():
                        data_dict[col_name] = []
                    data_dict[col_name].append(num_pts_dict[band])

                    col_name = str(i) + "mid_pt"
                    if col_name not in data_dict.keys():
                        data_dict[col_name] = []
                    data_dict[col_name].append(pc.mid_point_dict[band])

                object_type = data_ob.get_object_type_number(object_id)
                data_dict['type'].append(object_type)

            if self.use_random_current_date:
                data_dict['curr_date'] = np.asarray(current_dates)
            data_df = pd.DataFrame(data_dict)
            data_df = data_df.sample(frac=1).reset_index(drop=True)
            temp_df = data_df

        if data_set == 'train':
            self.train_features_df, self.sample_numbers_train = sample_from_df(temp_df, self.sample_numbers_train,
                                                                               shuffle=True)
            self.train_features_df['y_true'] = self.train_features_df['id'].map(
                lambda ob_id: 1 if data_ob.get_object_type_number(ob_id) in self.prediction_type_nos else 0)
            self.train_data_ob = data_ob
        elif data_set == 'test':
            self.test_features_df, self.sample_numbers_test = sample_from_df(temp_df, self.sample_numbers_test,
                                                                             shuffle=True)
            self.test_features_df['y_true'] = self.test_features_df['id'].map(
                lambda ob_id: 1 if data_ob.get_object_type_number(ob_id) in self.prediction_type_nos else 0)
            self.test_data_ob = data_ob

    def get_features_col_names(self):

        col_names = []
        for i, band in enumerate(self.bands):
            for j in range(1, self.num_pc_components + 1):
                col_name = str(i) + 'pc' + str(j)
                col_names.append(col_name)
            if self.use_number_of_points_per_band:
                col_name = str(i) + 'n'
                col_names.append(col_name)

        return col_names

    def train_model(self, use_number_of_points_per_band=False):

        if self.train_features_df is None:
            print("create training features")
            # TODO: handle case properly
        self.use_number_of_points_per_band = use_number_of_points_per_band
        col_names = self.get_features_col_names()

        features = self.train_features_df[col_names]

        self.classifier.fit(features, self.train_features_df['y_true'])

        predict = self.classifier.predict(features)

        self.train_features_df['y_pred'] = predict
        self.classifier_trained = True

        return self.train_features_df[['id', 'y_pred']]

    def predict_test_data(self):

        col_names = self.get_features_col_names()
        features = self.test_features_df[col_names]
        if self.classifier_trained:
            predict = self.classifier.predict(features)
            self.test_features_df['y_pred'] = predict
        else:
            print("Please train classifier first")

        return self.test_features_df[['id', 'y_pred']]

    def plot_prediction(self, color_band_dict=None, plot_all_predictions=True, fig=None,
                        mark_maximum=False, plot_predicted_curve_of_type=None, save_fig_path=None):
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
        # TODO: pass PCs to PredictLightCurve
        pcs = get_pcs(num_pc_components=self.num_pc_components, bands=self.bands, decouple_pc_bands=False,
                      band_choice='u')
        if not self.classifier_trained:
            print("train classifer and predict_test_data func")
            # todo: what to do if test_data is not called before
            return

        if color_band_dict is None:
            print("error: pass color of each band")

        if plot_all_predictions:
            plot_predicted_curve_of_type = self.test_features_df['type'].tolist()

        col_names = self.get_features_col_names()

        if self.test_features_df is not None:

            for _, row in self.test_features_df.iterrows():

                object_type = row['type']
                object_id = row['id']

                if object_type in plot_predicted_curve_of_type:

                    lc = LightCurve(data_ob=self.test_data_ob, object_id=object_id)
                    fig = lc.plot_light_curve(fig=fig, color_band_dict=color_band_dict, alpha=0.3,
                                                   mark_maximum=False,
                                                   mark_label=False, plot_points=True)

                    coeff_list = [row[col_names]]
                    correct_pred = ((self.classifier.predict(coeff_list)[0] == 1) & (
                            object_type in self.prediction_type_nos)) | (
                                           (self.classifier.predict(coeff_list)[0] == 0) & (
                                           object_type not in self.prediction_type_nos))

                    for i, band in enumerate(self.bands):
                        coeff_list = []
                        mid_point = row[str(i) + "mid_pt"]
                        mid_point = mid_point - mid_point % 2
                        print(mid_point)
                        if mid_point == 0:
                            continue
                        for j in range(self.num_pc_components):
                            band_feature_col = str(i) + 'pc' + str(j + 1)
                            coeff_list.append(row[band_feature_col])

                        predicted_lc = calc_prediction(coeff_list, pcs[band])
                        time_data = np.arange(mid_point - 50, mid_point + 52, 2)
                        ax = fig.gca()
                        ax.plot(time_data, predicted_lc, color=color_band_dict[band])

                        fig = lc.plot_light_curve(color_band_dict, start_date=mid_point - 50, end_date=mid_point + 50,
                                                  fig=fig, band=band, alpha=1, mark_maximum=False, plot_points=True)

                    if save_fig_path is not None:
                        if not correct_pred:
                            fig.savefig(save_fig_path + "incorrect/" + str(object_type) + "_" + str(object_id)+".png")
                        else:
                            fig.savefig(save_fig_path + "correct/" + str(object_type) + "_" + str(object_id)+".png")

                plt.show()
                plt.close('all')

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
                    if x_limits is not None:
                        ax_current.set_xlim(x_limits)
                    if y_limits is not None:
                        ax_current.set_ylim(y_limits)
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
