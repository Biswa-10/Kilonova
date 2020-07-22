from LightCurve import LightCurve
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def calc_prediction(coeff, PCs, bias=None):
    predicted_lc = np.zeros_like(PCs.shape[1])
    for a, b in zip(PCs, coeff): predicted_lc = np.add(predicted_lc, b * a)
    if bias is not None:
        predicted_lc = predicted_lc + bias
    return predicted_lc


def calc_loss(coeff, PCs, light_curve_seg, bias=None):
    index = light_curve_seg != 0
    y_pred = calc_prediction(coeff, PCs, bias=bias)
    diff = light_curve_seg - y_pred
    neg_index = y_pred < 0
    diff = diff[index | neg_index]
    error = np.sum(np.square(diff, diff))
    return error


class PredictLightCurve:

    def __init__(self, data_ob, object_id, num_pc_components=3):

        self.lc = LightCurve(data_ob, object_id)
        self.current_date = None
        self.num_pc_components = num_pc_components
        self.bands = None
        self.pcs = None
        self.decouple_prediction_bands = True

        self.min_flux_threshold = 0
        self.num_prediction_days = 51
        self.mid_point_dict = None
        self.prediction_start_date = np.inf
        self.prediction_end_date = 0

    def get_time_segment(self, start_date, end_date, current_date=None):
        start_index = self.lc.df[self.lc.time_col_name] >= start_date
        end_index = self.lc.df[self.lc.time_col_name] <= end_date
        if current_date is None:
            return self.lc.df[start_index & end_index]
        else:
            past_index = self.lc.df[self.lc.time_col_name] <= current_date
            return self.lc.df[start_index & end_index & past_index]

    def get_pcs(self, num_pc_components, decouple_pc_bands=False, band_choice='z'):

        if decouple_pc_bands:
            pc_dict = np.load("principal_components/PC_all_bands_diff_mid_pt_dict.npy")
            pc_dict = pc_dict.item()
            pc_out = {0: pc_dict['u'][0:num_pc_components], 1: pc_dict['r'][0:num_pc_components],
                      2: pc_dict['i'][0:num_pc_components], 3: pc_dict['g'][0:num_pc_components],
                      4: pc_dict['z'][0:num_pc_components], 5: pc_dict['Y'][0:num_pc_components]}
            # num_pc_components = int(num_pc_components)
            # print(pc_dict['u'])

        else:
            pc_out = {}
            pc_dict = np.load("principal_components/PC_all_bands_diff_mid_pt_dict.npy")
            pc_dict = pc_dict.item()
            for band in self.bands:
                pc_out[band] = pc_dict[band_choice][0:num_pc_components]

        return pc_out

    def get_binned_time(self, df):
        return df[self.lc.time_col_name] - df[self.lc.time_col_name] % 2

    def optimize_coeff(self, band_df, mid_point_date):

        if len(band_df) > 0:

            start_date = mid_point_date - (self.num_prediction_days - 1)
            end_date = mid_point_date + (self.num_prediction_days - 1)
            start_index = band_df[self.lc.time_col_name] >= start_date
            end_index = band_df[self.lc.time_col_name] <= end_date

            past_index = band_df[self.lc.time_col_name] <= self.current_date
            fit_df = band_df[start_index & end_index & past_index]

            if len(fit_df) > 0:
                binned_dates = self.get_binned_time(fit_df)
                b2 = (binned_dates - mid_point_date + self.num_prediction_days - 1) / 2
                b2 = b2.astype(int)
                light_curve_seg = np.zeros(self.num_prediction_days)
                light_curve_seg[b2[:]] = fit_df[self.lc.brightness_col_name]
                # initial_guess = np.amax(fit_df[self.lc.brightness_col_name])*np.array([.93,.03 ,.025])
                #initial_guess = np.asarray([.93, .03, .025])
                initial_guess = np.zeros(self.num_pc_components)
                result = minimize(calc_loss, initial_guess, args=(self.pcs, light_curve_seg))
                #result = minimize(calc_loss, args=(self.pcs, light_curve_seg))

                return result.x

        return np.zeros(self.num_pc_components)

    def get_mid_pt_dict(self):

        mid_point_dict = {}

        event_df = self.lc.df

        if self.current_date is not None:
            date_difference = event_df[self.lc.time_col_name] - self.current_date
            past_index = (date_difference >= -50) & (date_difference <= 0)
            event_df = event_df[past_index]
            # print(event_df)
            if not self.decouple_prediction_bands:
                band_mid_points = []
                for i, band in enumerate(self.bands):
                    # print(band)
                    band_index = event_df[self.lc.band_col_name] == band
                    band_df = event_df[band_index]
                    # print(band_df)
                    if len(band_df) > 0:
                        max_index = np.argmax(band_df[self.lc.brightness_col_name])
                        band_mid_points.append(band_df[self.lc.time_col_name][max_index])
                if len(band_mid_points) > 0:
                    for band in self.bands:
                        mid_point_dict[band] = np.median(np.array(band_mid_points))
                else:
                    return None
            else:
                for band in self.bands:
                    # print(band)
                    band_index = event_df[self.lc.band_col_name] == band
                    band_df = event_df[band_index]
                    # print(band_df)
                    if len(band_df) > 0:
                        max_index = np.argmax(band_df[self.lc.brightness_col_name])
                        if band_df[self.lc.brightness_col_name][max_index] > self.min_flux_threshold:
                            mid_point_dict[band] = band_df[self.lc.time_col_name][max_index]
                        else:
                            mid_point_dict[band] = None
                    else:
                        mid_point_dict[band] = None

        else:

            priority_regions = self.lc.find_region_priority()
            priority_region1 = priority_regions[0]
            median = np.median(np.asarray(priority_region1))
            event_df = self.get_time_segment(median - 50, median + 50)

            if self.decouple_prediction_bands:
                for band in self.bands:
                    # print(band)
                    band_df = self.lc.extract_band_data(band, event_df)
                    # print(band_df)
                    if len(band_df) > 0:
                        max_index = np.argmax(band_df[self.lc.brightness_col_name])
                        # print(min_flux_threshold)
                        if band_df[self.lc.brightness_col_name][max_index] > self.min_flux_threshold:
                            mid_point_dict[band] = band_df[self.lc.time_col_name][max_index]
                        else:
                            mid_point_dict[band] = None
                    else:
                        mid_point_dict[band] = None
            else:
                for band in self.bands:
                    band_df = self.lc.extract_band_data(band, event_df)
                    if len(band_df) > 0:
                        # print(min_flux_threshold)
                        if np.amax(band_df[self.lc.brightness_col_name] > self.min_flux_threshold):
                            mid_point_dict[band] = median
                        else:
                            mid_point_dict[band] = None
                    else:
                        mid_point_dict[band] = None

        # print(mid_point_dict)

        return mid_point_dict

    def predict_lc_coeff(self, current_date, num_pc_components, bands, decouple_pc_bands, decouple_prediction_bands,
                         min_flux_threshold, band_choice='u'):
        self.current_date = current_date
        self.num_pc_components = num_pc_components
        self.bands = bands
        self.pcs = self.get_pcs(num_pc_components, decouple_pc_bands=decouple_pc_bands, band_choice=band_choice)
        self.decouple_prediction_bands = decouple_prediction_bands

        self.min_flux_threshold = min_flux_threshold
        self.num_prediction_days = 51
        self.mid_point_dict = self.get_mid_pt_dict()

        coeff_all_band = {}
        num_points_dict = {}

        if self.mid_point_dict is not None:

            for band in self.bands:
                mid_point_date = self.mid_point_dict[band]
                if mid_point_date is None:
                    coeff_all_band[band] = np.zeros(num_pc_components)
                    num_points_dict[band] = 0
                    continue

                prediction_start_date = mid_point_date - (self.num_prediction_days - 1)
                prediction_end_date = mid_point_date + (self.num_prediction_days - 1)
                event_df = self.get_time_segment(prediction_start_date, prediction_end_date, self.current_date)

                band_index = event_df[self.lc.band_col_name] == band
                band_df = event_df[band_index]
                # print(band_df)
                pcs = self.pcs[band]
                if len(band_df) > 0:
                    print(band)

                    binned_dates = self.get_binned_time(band_df)
                    if mid_point_date - self.num_prediction_days + 1 < self.prediction_start_date:
                        self.prediction_start_date = mid_point_date - self.num_prediction_days + 1
                    if mid_point_date + self.num_prediction_days - 1 > self.prediction_end_date:
                        self.prediction_end_date = mid_point_date + self.num_prediction_days - 1
                    b2 = (binned_dates - mid_point_date + self.num_prediction_days - 1) / 2
                    b2 = b2.astype(int)
                    light_curve_seg = np.zeros(self.num_prediction_days)
                    light_curve_seg[b2[:]] = band_df[self.lc.brightness_col_name]
                    #initial_guess = np.amax(band_df[self.lc.brightness_col_name]) * np.array([.93, .03, .025])
                    initial_guess = np.zeros(self.num_pc_components)
                    result = minimize(calc_loss, initial_guess, args=(pcs, light_curve_seg))
                    coeff_all_band[band] = list(result.x)
                    print(result.x)
                    num_points_dict[band] = len(b2)

                else:
                    coeff_all_band[band] = np.zeros(num_pc_components)
                    num_points_dict[band] = 0

        else:
            for band in self.bands:
                coeff_all_band[band] = np.zeros(num_pc_components)
                num_points_dict[band] = 0

        return coeff_all_band, num_points_dict

    def plot_predicted_bands(self, all_band_coeff_dict, color_band_dict, mark_maximum=False, object_name=None,
                             axes_lims=True, buffer_days=20, mark_threshold=True):

        fig = self.lc.plot_light_curve(color_band_dict=color_band_dict, alpha=0.3, mark_maximum=False, mark_label=False,
                                       plot_points=True)

        if self.mid_point_dict is not None:
            median_date = None
            for band, coeff in all_band_coeff_dict.items():
                mid_point_date = self.mid_point_dict[band]
                if not self.decouple_prediction_bands:
                    median_date = mid_point_date
                # print(mid_point_date)
                if mid_point_date is not None:

                    if self.current_date is None:
                        end_date = mid_point_date + 50
                    else:
                        end_date = self.current_date

                    if mark_maximum:
                        fig = self.lc.plot_light_curve(color_band_dict, fig=fig, start_date=mid_point_date - 50,
                                                       end_date=end_date, band=band, alpha=1, mark_maximum=True,
                                                       plot_points=True)
                    else:
                        fig = self.lc.plot_light_curve(color_band_dict, fig=fig, start_date=mid_point_date - 50,
                                                       end_date=end_date, band=band, alpha=1, mark_maximum=False,
                                                       plot_points=True)

                    if len(coeff) != 0:
                        predicted_lc = calc_prediction(coeff, self.pcs[band])
                        # plt.plot(x_data, predicted_lc, color = color_band_dict[band])
                        time_data = np.arange(0, 102, 2) + mid_point_date - 50
                    else:
                        predicted_lc = []
                        time_data = []

                    plt.plot(time_data, predicted_lc, color=color_band_dict[band])

            if axes_lims:
                if self.prediction_start_date is not np.inf:
                    plt.xlim(left=self.prediction_start_date - buffer_days)
                if self.prediction_end_date != 0:
                    plt.xlim(right=self.prediction_end_date + buffer_days)

            if not self.decouple_prediction_bands:
                _, _, ymin, ymax = plt.axis()
                plt.plot([median_date, median_date], [ymin / 2, ymax / 2], color="slateblue", ls="dashed",
                         label="median of max dates")

        xmin, xmax, ymin, ymax = plt.axis()
        if self.current_date is not None:
            plt.plot([self.current_date, self.current_date], [ymin / 2, ymax / 2], color="darkorange", ls="dashed",
                     label="current date")

        ax = plt.gca()
        if mark_threshold:
            ax.axhline(y=self.min_flux_threshold, color='r', linestyle='--', label='band threshold')
        plt.text(.01, .94, "ID: " + str(self.lc.object_id), fontsize=15, transform=ax.transAxes)
        if object_name is not None:
            # print(self.lc.get_object_type_for_PLAsTiCC(object_id))
            plt.text(.01, .88, "Type: " + object_name, fontsize=15, transform=ax.transAxes)

        plt.xlabel("mjd", fontsize=20)
        plt.ylabel("flux", fontsize=20)

        plt.legend(loc="upper right")

        return fig
