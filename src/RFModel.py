from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from src.dataframe import Data
from src.Predict_lc import PredictLightCurve, calc_prediction, get_pcs
from src.LightCurve import LightCurve
from tqdm.notebook import tqdm
from random import random
import math
from src.io_utils import ztf_ob_type_name
from sklearn import metrics
import pandas as pd


def sample_from_df(data_df, sample_numbers=None, shuffle=False):
    """
    samples a fixed number of objects of each type, based on sample_numbers.

    :param data_df: data for all the events from which to sample
    :param sample_numbers: a dict with keys as event types and values as numbrer of objects of that type to be sampled.
        if not passed, the dict will be updated with number of events of each type.
    :param shuffle: shuffle the output (if false then objects of same type will be placed together)
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

    def __init__(self, prediction_type_nos, pcs=None, sample_numbers_train=None,
                 sample_numbers_test=None, decouple_prediction_bands=True, decouple_pc_bands=False,
                 min_flux_threshold=200, num_pc_components=3, use_random_current_date=False, bands=None,
                 band_choice='u', num_alert_days=None, skip_random_date_event_types=None, train_features_path=None,
                 test_features_path=None):
        """
        creates a pandas dataframe with the features.
        Note that this function makes predictions and plots for all the data!
        :param prediction_type_nos: list with the type numbers to be identified
        :param pcs: pcs to be used for predictions
        :param train_features_path: if the train features are already calculated, pass path to the saved features
        :param test_features_path: if the test features are already calculated, pass path to the saved features
        :param sample_numbers_train: dict value that stores the number of events to be sampled in training set
            corresponding to each type
        :param sample_numbers_test: dict value that stores the number of events to be sampled in training set
                corresponding to each type
        :param decouple_prediction_bands: if True (recommended) each band will have its own midpoint. If False, the date
           of mid point will be the same for all bands
        :param decouple_pc_bands: use different sets of PCs for different bands (should be used only for LSST data,
            only in the case when default function is used)
        :param min_flux_threshold: minimum value of the amplitude necessary to make predictions
        :param num_pc_components: number of PC components to be used
        :param use_random_current_date: Try to mimic alerts by choosing selecting data only upto a specific date for
           making plots
        :param band_choice: bands on which fit it to be generated
        :param num_alert_days: the number of days of data to be considered for making fits (can be used in combination
           with random_current_date
        :param skip_random_date_event_types: skips using the random current date for certain event types
        :return:
        """
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
        self.performance_statistics_df = None
        if pcs is None:
            self.pcs = get_pcs(self.num_pc_components, self.bands, decouple_pc_bands=self.decouple_pc_bands,
                               band_choice=self.band_choice)
        else:
            self.pcs = pcs
            print("done")

    def create_features_df(self, data_set: str, data_ob: Data = None, discard_no_feature_events=True):
        """
        Computes the coefficients of all the events in the data object and stores them in self.train_features_df or
        self.test_features_df based on the selection of dataset

        :param data_set: either "train" or "test"
        :param data_ob: corresponding data object
        :param discard_no_feature_events: discard events where no prediction are made in any band
        """

        features_path = None
        if data_set == 'train':
            features_path = self.train_features_path

        elif data_set == 'test':
            features_path = self.test_features_path

        if features_path is not None:
            try:
                features_data_df = pd.read_csv(features_path)
            except FileNotFoundError:
                print("Features file not found!")
        else:
            data_dict = {'id': [],
                         'type': [],
                         'pred_start_date': [],
                         'pred_end_date': []}
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
                                                               bands=self.bands, pcs=self.pcs,
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

                data_dict['pred_start_date'].append(pc.prediction_start_date)
                data_dict['pred_end_date'].append(pc.prediction_end_date)

            if self.use_random_current_date:
                data_dict['curr_date'] = np.asarray(current_dates)
            features_data_df = pd.DataFrame(data_dict)
            features_data_df = features_data_df.sample(frac=1).reset_index(drop=True)

        temp_df = features_data_df

        if data_set == 'train':
            self.train_features_df, self.sample_numbers_train = sample_from_df(temp_df, self.sample_numbers_train,
                                                                               shuffle=True)
            self.train_features_df['y_true'] = self.train_features_df['id'].map(
                lambda ob_id: 1 if data_ob.get_object_type_number(ob_id) in self.prediction_type_nos else 0)
            self.train_data_ob = data_ob
            if discard_no_feature_events:
                self.discard_no_features_train_events()
        elif data_set == 'test':
            self.test_features_df, self.sample_numbers_test = sample_from_df(temp_df, self.sample_numbers_test,
                                                                             shuffle=True)
            self.test_features_df['y_true'] = self.test_features_df['id'].map(
                lambda ob_id: 1 if data_ob.get_object_type_number(ob_id) in self.prediction_type_nos else 0)
            self.test_data_ob = data_ob
            if discard_no_feature_events:
                self.discard_no_features_test_events()

    def get_features_col_names(self):
        """
        get the name of the features columns to be used for training/testing the classifier

        :return: List of strings with the names of columns to be used by the classifier
        """

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
        """
        function to train the classifier model

        :param use_number_of_points_per_band: flag to either use the number of points in each band or not
        :return: pandas dataframe with objects ID and the predicted classification of the objects in the training set
        """

        if self.train_features_df is None:
            print("create training features")
            return
        self.use_number_of_points_per_band = use_number_of_points_per_band
        col_names = self.get_features_col_names()

        features = self.train_features_df[col_names]

        self.classifier.fit(features, self.train_features_df['y_true'])

        predict = self.classifier.predict(features)
        y_score = self.classifier.predict_proba(features)

        self.train_features_df['y_pred'] = predict
        self.train_features_df['y_score'] = y_score[:, 1]
        self.classifier_trained = True

        return self.train_features_df[['id', 'y_pred']]

    def predict_test_data(self):
        """
        function to predict probability and score for the test data

        :return: pandas dataframe with object ids, prediction, and probability score of test events
        """
        if self.test_features_df is None:
            print("create test features")
            return
        col_names = self.get_features_col_names()
        features = self.test_features_df[col_names]
        if self.classifier_trained:
            predict = self.classifier.predict(features)
            y_score = self.classifier.predict_proba(features)
            self.test_features_df['y_pred'] = predict
            self.test_features_df['y_score'] = y_score[:, 1]
        else:
            print("Please train classifier first")

        return self.test_features_df[['id', 'y_pred', 'y_score']]

    def plot_prediction(self, color_band_dict=None, plot_predicted_curve_of_type=None, save_fig_path=None):
        """
        plots predictions either for all data or a selected type

        :param color_band_dict: dict with the different bands and the corresponding colors to be used to make plots
        :param plot_predicted_curve_of_type: plot predictions of curves only of certain types. If left None, plots are
            made for all events
        :param save_fig_path: path in which the plots are to be saved. If left None, plots are not saved
        """

        if not self.classifier_trained:
            print("train classifer and predict_test_data func")
            return

        if color_band_dict is None:
            print("error: pass color of each band")

        if plot_predicted_curve_of_type is None:
            plot_predicted_curve_of_type = self.test_features_df['type'].tolist()

        col_names = self.get_features_col_names()

        if self.test_features_df is not None:

            for _, row in self.test_features_df.iterrows():

                object_type = row['type']
                object_id = row['id']

                if object_type in plot_predicted_curve_of_type:
                    fig = plt.figure(figsize=(15, 9))

                    lc = LightCurve(data_ob=self.test_data_ob, object_id=object_id)

                    coeff_list = [row[col_names]]
                    correct_pred = ((self.classifier.predict(coeff_list)[0] == 1) & (
                            object_type in self.prediction_type_nos)) | (
                                           (self.classifier.predict(coeff_list)[0] == 0) & (
                                           object_type not in self.prediction_type_nos))

                    prediction = False
                    for i, band in enumerate(self.bands):
                        coeff_list = []
                        mid_point = row[str(i) + "mid_pt"]
                        if math.isnan(mid_point):
                            continue
                        mid_point = mid_point - mid_point % 2
                        prediction = True
                        for j in range(self.num_pc_components):
                            band_feature_col = str(i) + 'pc' + str(j + 1)
                            coeff_list.append(row[band_feature_col])

                        predicted_lc = calc_prediction(coeff_list, self.pcs[band])
                        time_data = np.arange(mid_point - 50, mid_point + 52, 2)
                        ax = fig.gca()
                        ax.plot(time_data, predicted_lc, color=color_band_dict[band], label="band " + str(band) +
                                                                                            " prediction")

                        if self.use_random_current_date:
                            current_date = row['curr_date']
                            band_data_start_date = max(current_date - self.num_alert_days, mid_point - 50)
                            band_data_end_date = min(current_date, mid_point + 50)
                            ax_y_limits = plt.gca().get_ylim()
                            plt.plot([current_date, current_date], [ax_y_limits[0] / 2, ax_y_limits[1] / 2], c='b',
                                     ls="--", label="current date")
                            prediction_points_label = "alert region"

                        else:
                            band_data_start_date = mid_point - 50
                            band_data_end_date = mid_point + 50

                            prediction_points_label = "prediction region"

                        fig = lc.plot_light_curve(color_band_dict, start_date=band_data_start_date, band=band,
                                                  end_date=band_data_end_date, fig=fig, alpha=1, mark_maximum=False,
                                                  plot_points=True, label_postfix=prediction_points_label)

                    plt.axhline(self.min_flux_threshold, label="min amplitude threshold", c='m', ls="--")
                    fig = lc.plot_light_curve(fig=fig, color_band_dict=color_band_dict, alpha=0.3,
                                              mark_maximum=False,
                                              mark_label=True, plot_points=True, label_postfix="light curve")
                    plt.legend(fontsize=15)
                    if prediction and (save_fig_path is not None):
                        if not correct_pred:
                            fig.savefig(
                                save_fig_path + "incorrect/" + str(object_type) + "_" + str(int(object_id)) + ".png")
                        else:
                            fig.savefig(
                                save_fig_path + "correct/" + str(object_type) + "_" + str(int(object_id)) + ".png")

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
            fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_rows * 5, num_cols * 5))
            # fig.subplots_adjust(wspace=.5,hspace=.5)
            ax_list = fig.axes
        else:
            ax_list = fig.axes
        if bands is None:
            bands = self.bands
        print(bands)
        for i, band in enumerate(bands):
            print("i = "+str(i))
            print(band)
            for x in range(self.num_pc_components):
                for y in range(x):
                    print(x)
                    print(y)
                    ax_current = ax_list[int(i * num_cols + (x - 1) * (x) / 2 + y)]
                    colx_name = str(i) + "pc" + str(x + 1)
                    coly_name = str(i) + "pc" + str(y + 1)
                    if mark_xlabel:
                        ax_current.set_xlabel("PC" + str(x + 1), fontsize=20)
                    if mark_ylabel:
                        ax_current.set_ylabel("PC" + str(y + 1), fontsize=20)
                    PCx = class_features_df[colx_name].values
                    PCy = class_features_df[coly_name].values
                    if color_band_dict is not None:
                        ax_current.scatter(PCx, PCy, color=color_band_dict[band], alpha=.5, label=label)
                    else:
                        ax_current.scatter(PCx, PCy, color="yellow", alpha=.4, label=label)
                    if x_limits is not None:
                        print("-----------------------")
                        print(x)
                        print(y)
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
                    #ax_current.axis('square')
        fig.tight_layout()
        return fig

    def plot_features_correlation(self, color_band_dict, bands=None,
                                  x_limits=None, y_limits=None, mark_xlabel=True, mark_ylabel=True, band_map=None,
                                  set_ax_title=True):
        """
        plots correlations between the PCs of each band (with the training set features)

        :param class_features_df: dataframe of events of current class (KN and non-KN)
        :param bands: bands for which plots are to be generated
        :param color_band_dict:test colors to be used for corresponding bands
        :param x_limits: x limits of the plot
        :param y_limits: y limits of the plot
        :param mark_xlabel: mark x label or not
        :param mark_ylabel: to mark y label or not
        :param band_map: renaming bands/filter/channel name in plots
        :param set_ax_title: title of the axes ojbect on which plot is made
        :return: figure with the plots
        """
        kn_df = self.train_features_df[self.train_features_df['y_true'] == 1]
        non_kn_df = self.train_features_df[self.train_features_df['y_true'] == 0]
        if bands is None:
            bands = self.bands
        num_rows = len(bands)
        num_cols = int(self.num_pc_components * (self.num_pc_components - 1) / 2)
        print(num_rows)
        print(num_cols)
        print(self.bands)
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
                    if x_limits is not None:
                        ax_current.set_xlim(x_limits)
                    if y_limits is not None:
                        ax_current.set_ylim(y_limits)
                    if band_map is None:
                        if mark_xlabel:
                            ax_current.set_xlabel(x_band + " band", fontsize=20)
                        if mark_ylabel:
                            ax_current.set_ylabel(y_band + " band", fontsize=20)
                    else:
                        if mark_xlabel:
                            ax_current.set_xlabel(band_map[x_band] + " band", fontsize=20)
                        if mark_ylabel:
                            ax_current.set_ylabel(band_map[y_band] + " band", fontsize=20)
                    if set_ax_title:
                        ax_current.set_title("correlation for PC" + str(i + 1), fontsize=20)
                    if label != "":
                        ax_current.legend(loc="upper right")
                    #ax_current.set_aspect('equal', 'box')
        fig.tight_layout()
        return fig

    def plot_band_correlation(self, bands=None, color_band_dict=None, x_limits=None,
                              y_limits=None, mark_xlabel=True, mark_ylabel=True, band_map=None, set_ax_title=True):
        """
        plots correlations between 2 bands for each PC (with the training set features)

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
        kn_df = self.train_features_df[self.train_features_df['y_true'] == 1]
        non_kn_df = self.train_features_df[self.train_features_df['y_true'] == 0]
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

    def discard_no_features_train_events(self):
        """
        discards events if no fit it generated (minimum threshold is not crossed or no data points) have been made
        """
        col_names = []
        non_zero_index = np.zeros((len(self.train_features_df)), dtype='bool')
        for i in range(self.num_pc_components):
            for j in range(len(self.bands)):
                col_name = str(j) + 'pc' + str(i + 1)
                non_zero_index = non_zero_index | (self.train_features_df[col_name] != 0)

        self.train_features_df = self.train_features_df[non_zero_index]
        self.train_features_df.reset_index(drop=True, inplace=True)
        self.train_data_ob.df_data = self.train_data_ob.df_data[
            np.isin(self.train_data_ob.df_data[self.train_data_ob.object_id_col_name], self.train_features_df['id'])]
        self.train_data_ob.df_metadata = self.train_data_ob.df_metadata[
            np.isin(self.train_data_ob.df_metadata[self.train_data_ob.object_id_col_name],
                    self.train_features_df['id'])]
        _, self.sample_numbers_train = sample_from_df(self.train_features_df, self.sample_numbers_train)

    def discard_no_features_test_events(self):
        """
        discards events if no fit it generated (minimum threshold is not crossed or no data points) have been made
        """
        col_names = []
        non_zero_index = np.zeros((len(self.test_features_df)), dtype='bool')
        for i in range(self.num_pc_components):
            for j in range(len(self.bands)):
                col_name = str(j) + 'pc' + str(i + 1)
                non_zero_index = non_zero_index | (self.test_features_df[col_name] != 0)

        self.test_features_df = self.test_features_df[non_zero_index]
        self.test_features_df.reset_index(drop=True, inplace=True)
        self.test_data_ob.df_data = self.test_data_ob.df_data[
            np.isin(self.test_data_ob.df_data[self.test_data_ob.object_id_col_name], self.test_features_df['id'])]
        self.test_data_ob.df_metadata = self.test_data_ob.df_metadata[
            np.isin(self.test_data_ob.df_metadata[self.test_data_ob.object_id_col_name], self.test_features_df['id'])]
        _, self.sample_numbers_test = sample_from_df(self.test_features_df, self.sample_numbers_test)

    def get_performance_statistics_df(self):
        """
        functions to evaluate performance for each event type
        :return: df with number of events of each type: correctly classified, total number of events of the type and
            number of events of the type in training set
        """
        prediction_stat = {}
        for i, object_id in enumerate(self.test_features_df['id']):
            # print(1)
            type_no = self.test_features_df['type'].values[np.where(self.test_features_df['id'] == object_id)][0]
            # print(self.train_sample_numbers)
            num_training_events = self.sample_numbers_train[type_no]
            if num_training_events == 0:
                type_no = 0

            if type_no not in prediction_stat:
                prediction_stat[type_no] = [0, 1, num_training_events]
            else:
                prediction_stat[type_no][1] = prediction_stat[type_no][1] + 1

            if (type_no in self.prediction_type_nos) & (self.test_features_df['y_pred'].values[i] == 1):
                prediction_stat[type_no][0] = prediction_stat[type_no][0] + 1

            elif (self.test_features_df['y_pred'].values[i] == 0) & (type_no not in self.prediction_type_nos):
                prediction_stat[type_no][0] = prediction_stat[type_no][0] + 1
        stat_df = pd.DataFrame(prediction_stat)
        return stat_df.reindex(sorted(stat_df.columns), axis=1)

    def plot_contamination_statistics(self, ax=None):
        """
        plot displaying total number of events and number of events correctly classified for each event type.
        :param ax: axes on which plot is to be made
        """

        if self.performance_statistics_df is None:
            self.performance_statistics_df = self.get_performance_statistics_df()

        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_axes([0, 0, .9, .9])

        col_type_nos = np.array(self.performance_statistics_df.columns)
        # print(col_type_nos)
        pred_col_names = [ztf_ob_type_name(item) for item in self.prediction_type_nos]
        non_pred_types = col_type_nos[~np.isin(col_type_nos, self.prediction_type_nos)]
        # print(pred_col_names)
        non_pred_type_names = [ztf_ob_type_name(item) for item in non_pred_types]
        col_type_names = [ztf_ob_type_name(item) for item in col_type_nos]

        # print(col_type_nos)
        # print(non_pred_types)

        # print(np.where(np.in1d(non_pred_types, col_type_nos)))
        ax.barh(col_type_names, self.performance_statistics_df.loc[1], alpha=.6, tick_label=col_type_names,
                color='bisque', ec='black', linewidth=1, label="total number of events")
        ax.barh(non_pred_type_names, self.performance_statistics_df[non_pred_types].loc[0], alpha=.6, color='red',
                ec='black', label='Correctly classified: class 0')
        ax.barh(pred_col_names, self.performance_statistics_df[self.prediction_type_nos].loc[0], alpha=.6,
                color='chartreuse', ec='black', label='Correctly classified: class 1')
        # plt.rc('ytick', labelsize=15)
        # plt.rc('xtick', labelsize=15)
        ax.tick_params(axis='both', labelsize=20)
        # print(col_type_nos)
        for i, v in enumerate(col_type_nos):
            ax.text(self.performance_statistics_df[v].values[1] + 10, i - .1,
                    str(self.performance_statistics_df[v].values[0]) + "/" + str(
                        self.performance_statistics_df[v].values[1]) + " | " + str(
                        self.performance_statistics_df[v].values[2]),
                    color='blue', fontweight='bold', fontsize=10)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
        plt.xlim(right=np.max(self.performance_statistics_df.loc[1].values) * 120 / 100)

    def plot_confusion_matrix(self, ax, cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.

        :param ax: axes on which plot is to be generated
        :param cmap: color map for plotting
        :return:
        """
        title = ""

        # Compute confusion matrix
        cm = metrics.confusion_matrix(self.test_features_df['y_true'], self.test_features_df['y_pred'])
        # Only use the labels that appear in the data
        classes = [0, 1]

        # fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        # ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes)
        ax.set_xlabel('Predicted label', fontsize=20)
        ax.set_ylabel('True label', fontsize=20)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=15)
        # fig.tight_layout()
        ax.axis("equal")
        return ax

    def get_ignored_events(self):
        """
        gets event types that are ignored for during training and testing
        :return unknown: List of events that were discarded only during training
        :return dropped: List of events that were discarded during both training and testing
        """
        unknown = ""
        dropped = ""
        for key in self.sample_numbers_train:
            if (key in self.sample_numbers_train.keys()) & (self.sample_numbers_test[key] != 0):
                continue
            else:
                if self.sample_numbers_test[key] == 0:
                    if dropped == "":
                        dropped = ztf_ob_type_name(key)
                    else:
                        dropped = dropped + ", " + ztf_ob_type_name(key)
                else:
                    if unknown == "":
                        unknown = ztf_ob_type_name(key)
                    else:
                        unknown = unknown + ", " + ztf_ob_type_name(key)
        if unknown == "":
            unknown = 'None'
        if dropped == "":
            dropped = 'None'
        return unknown, dropped

    def plot_roc_curve(self, fpr, tpr, roc_auc, ax=None):
        """
        function to plot roc auc

        :param fpr: false positive rate
        :param tpr: true positive rate
        :param roc_auc: roc auc score
        :param ax: axes object
        """

        if ax is None:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.gca()
        print(roc_auc)
        ax.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)

        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc='lower right')
        # plt.plot([0, 1], [0, 1],'r--')

        ax.set_ylim([0, 1])
        ax.set_xlim([0, 1])
        # plt.axis("square")
        plt.ylabel('True Positive Rate', fontsize=20)
        plt.xlabel('False Positive Rate', fontsize=20)
        # plt.gca().set_aspect("equal")

    def prediction_type_names(self):
        """
        :return: for each type number, returns the name
        """
        name = ""
        for type_no in self.prediction_type_nos:
            if name == "":
                name = ztf_ob_type_name(type_no)
            else:
                name = name + ", " + ztf_ob_type_name(type_no)
        return name

    def plot_performance_statistics(self):
        """
        makes a plot with contamination statistics, roc curve, confusion matrix and train-test set metadata

        :return: figure with the plot
        """

        if self.performance_statistics_df is None:
            self.performance_statistics_df = self.get_performance_statistics_df()
        fig = plt.figure(figsize=(24, 12))
        # plt.subplot2grid((12,25), (0,0), colspan=25, rowspan=1, fig = fig)
        # plt.title("performance statistics [Predict:"+self.prediction_type_names()+"]", loc = "center")
        plt.subplot2grid((12, 24), (0, 5), colspan=9, rowspan=11, fig=fig)
        self.plot_contamination_statistics(ax=plt.gca())
        if ('y_true' not in self.test_features_df.keys()) | ('y_score' not in self.test_features_df.keys()):
            self.predict_test_data()
        fpr, tpr, thresholds = metrics.roc_curve(self.test_features_df['y_true'].values,
                                                 self.test_features_df['y_score'].values)
        roc_auc = metrics.auc(fpr, tpr)

        plt.subplot2grid((12, 24), (0, 18), rowspan=3, colspan=5, fig=fig)
        self.plot_roc_curve(fpr=fpr, tpr=tpr, roc_auc=roc_auc, ax=plt.gca())
        plt.subplot2grid((12, 24), (4, 19), rowspan=3, colspan=3, fig=fig)
        self.plot_confusion_matrix(ax=plt.gca())
        cm = metrics.confusion_matrix(self.test_features_df['y_true'], self.test_features_df['y_pred'])
        plt.annotate('KN correctly identified = ' + str(100 * cm[1][1] / (cm[1][0] + cm[1][1]))[:5] + "%",
                     xy=(.79, .355), xycoords='figure fraction',
                     horizontalalignment='center', verticalalignment='top',
                     fontsize=20)
        plt.annotate('Purity = ' + str(100 * cm[1][1] / (cm[0][1] + cm[1][1]))[:5] + "%",
                     xy=(.79, .315), xycoords='figure fraction',
                     horizontalalignment='center', verticalalignment='top',
                     fontsize=20)
        plt.annotate('Training data [class 1: ' + str(
            np.sum(self.performance_statistics_df[self.prediction_type_nos].loc[2].values)) + " | class 0:"
                     + str(np.sum(self.performance_statistics_df.loc[2].values) - np.sum(
            self.performance_statistics_df[self.prediction_type_nos].loc[2].values)) + "]",
                     xy=(.79, .275), xycoords='figure fraction',
                     horizontalalignment='center', verticalalignment='top',
                     fontsize=20)
        # plt.annotate("Num trees ="+, )
        plt.annotate('Performance statistics [Predict:' + self.prediction_type_names() + ']',
                     xy=(.5, .1), xycoords='figure fraction',
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=30)
        unknown_events, dropped_events = self.get_ignored_events()
        plt.annotate('Dropped events: ' + dropped_events,
                     xy=(.79, .23), xycoords='figure fraction',
                     horizontalalignment='center', verticalalignment='top',
                     fontsize=10)
        plt.annotate('Unknown events: ' + unknown_events,
                     xy=(.79, .20), xycoords='figure fraction',
                     horizontalalignment='center', verticalalignment='top',
                     fontsize=10)
        fig.tight_layout()
        return fig
