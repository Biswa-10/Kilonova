from LightCurve import LightCurve
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def calc_prediction(coeff, PCs, bias=None):
    """
    given teh coefficients and PCs, it calculates the prediction as a linear combination

    :param coeff: coefficients of the linear combinations for the PCs
    :param PCs: The PCs that are being used as templates
    :param bias: constant term to be added (currently 0: Todo)
    :return: prediction
    """
    predicted_lc = np.zeros_like(PCs.shape[1])
    for a, b in zip(PCs, coeff): predicted_lc = np.add(predicted_lc, b * a)
    if bias is not None:
        predicted_lc = predicted_lc + bias
    return predicted_lc


def calc_loss(coeff, PCs, light_curve_seg, bias=None):
    """
    function to calculate the loss to be optimized

    :param coeff: current value of coefficients
    :param PCs: principal components to the used for the prediction
    :param light_curve_seg: segment of lightcurve that is to be predicted
    :param bias: constant to be added to the fit [currently none ToDo]
    :return: loss that is to be optimized
    """
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
        self.num_alert_days = None
        self.data_start_date = np.inf
        self.data_end_date = 0

    def get_time_segment(self, start_date, end_date):
        """
        extracts a time segment (TODO: use function already defined in LightCurve class)
        :param start_date: start date of the extracted slice
        :param end_date:  end date of the extracted slice
        :return: data of the time segment before start and end date
        """
        start_index = self.lc.df[self.lc.time_col_name] >= start_date
        end_index = self.lc.df[self.lc.time_col_name] <= end_date
        return self.lc.df[start_index & end_index]

    def get_pcs(self, num_pc_components, path=None, decouple_pc_bands=False, band_choice='z'):
        """
        load PCs

        :param num_pc_components: number of PCs to be used for predictions
        :param path: path to saved PCs
        :param decouple_pc_bands: If set to True, then corresponding PCs are used for each band. If set to False, then
            then same set of PCs is used to fit all the bands
        :param band_choice: choice of PCs if the same set of PCs are to be used for all predictions.
        :return: dict of PCs
        """
        if decouple_pc_bands:
            if path is None:
                path = "principal_components/PC_all_bands_diff_mid_pt_dict.npy"
            pc_dict = np.load(path, allow_pickle=True)
            pc_dict = pc_dict.item()
            pc_out = {0: pc_dict['u'][0:num_pc_components], 1: pc_dict['r'][0:num_pc_components],
                      2: pc_dict['i'][0:num_pc_components], 3: pc_dict['g'][0:num_pc_components],
                      4: pc_dict['z'][0:num_pc_components], 5: pc_dict['Y'][0:num_pc_components]}
            # num_pc_components = int(num_pc_components)
            # print(pc_dict['u'])

        else:
            pc_out = {}
            if path is None:
                if band_choice == 'all':
                    path = "principal_components/PCs_shifted_mixed.npy"
                else:
                    path = "principal_components/PC_all_bands_diff_mid_pt_dict.npy"

            pc_dict = np.load(path, allow_pickle=True)
            pc_dict = pc_dict.item()
            for band in self.bands:
                pc_out[band] = pc_dict[band_choice][0:num_pc_components]

        return pc_out

    def get_binned_time(self, df):
        """
        returns the binned data  in bins of 2 days (hard coded here)

        :param df: dataframe on which binning is to be made
        :return: returns the binned time column
        """
        return df[self.lc.time_col_name] - df[self.lc.time_col_name] % 2

    def optimize_coeff(self, band_df, mid_point_date):
        """
        function to optimize the coefficients of PCs

        :param band_df: data of a particular band of the light curve (fitting is done band wise)
        :param mid_point_date: data of mid point of the band
        :return: optimized set of coefficients
        """

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
                # initial_guess = np.asarray([.93, .03, .025])
                initial_guess = np.zeros(self.num_pc_components)
                result = minimize(calc_loss, initial_guess, args=(self.pcs, light_curve_seg))
                # result = minimize(calc_loss, args=(self.pcs, light_curve_seg))

                return result.x

        return np.zeros(self.num_pc_components)

    def get_mid_pt_dict(self):
        """
        function to calculate the mid point of each band.
        :return: mid point of each band to be used for the fit
        """
        mid_point_dict = {}

        event_df = self.lc.df
        if self.current_date is not None:
            date_difference = event_df[self.lc.time_col_name] - self.current_date
            past_index = (date_difference >= -self.num_alert_days) & (date_difference <= 0)
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

    def get_alert_segment(self, current_date, num_alert_days):
        """
        extract time segments to mimic alerts. The returned data will consist data of certain number of days before the
        current date

        :param current_date: today
        :param num_alert_days: number of days to be kept in alerts
        :return: returns data within the alert range
        """
        upper_time_lim_index = self.lc.df[self.lc.time_col_name] <= current_date
        lower_time_lim_index = self.lc.df[self.lc.time_col_name] >= current_date - num_alert_days

        return self.lc.df[upper_time_lim_index & lower_time_lim_index]

    def predict_lc_coeff(self, num_pc_components, bands, decouple_pc_bands, decouple_prediction_bands,
                         min_flux_threshold, current_date = None, template_path=None, band_choice='u',
                         num_alert_days=None):
        """
        function to predict the coefficients of light-curve.

        :param num_pc_components: number of PC components to be used
        :param bands: bands to be used for prediction
        :param decouple_pc_bands: option to use corresponding PCs for each band. ex, PC for bang g is used to predict
            the band g of new light curve
        :param decouple_prediction_bands: If true, different bands have different mid points. (Recommended).
        :param min_flux_threshold: value that sets the prediction threshold. Prediction for a band is made only if the
            maximum flux for the band in above the threshold. Note that if the maximum is above the threshold all points
            (even those below the threshold) are used for the fit
        :param current_date: today's date (to mimic light-curves)
        :param template_path: path where PCs are saved
        :param band_choice: choice of bands for the PCs (used only if decouple_pc_bands is false)
        :param num_alert_days: Number of days of the alerts (used only if current date is passed). The data used for
            predictions lies only in the rage of <<num_alert_days>> before today
        :return: tuple of dictionaries with the optimized coefficients and number of points in each band
        """
        self.current_date = current_date
        self.num_pc_components = num_pc_components
        self.bands = bands
        self.pcs = self.get_pcs(num_pc_components, path=template_path, decouple_pc_bands=decouple_pc_bands,
                                band_choice=band_choice)
        self.decouple_prediction_bands = decouple_prediction_bands

        self.min_flux_threshold = min_flux_threshold
        self.num_prediction_days = 51
        if current_date is not None:
            if num_alert_days is None:
                num_alert_days = 50
        self.num_alert_days = num_alert_days
        self.mid_point_dict = self.get_mid_pt_dict()
        if current_date is not None:
            self.data_start_date = current_date - self.num_alert_days
            self.data_end_date = current_date
        else:
            self.data_start_date = np.amin(self.lc.df[self.lc.time_col_name])
            self.data_end_date = np.amax(self.lc.df[self.lc.time_col_name])
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
                # if current_date is None:
                #    self.data_start_date = prediction_start_date
                #    self.data_end_date = prediction_end_date

                event_df = self.get_time_segment(max([self.data_start_date, prediction_start_date]),
                                                 min([self.data_end_date, prediction_end_date]))

                band_index = event_df[self.lc.band_col_name] == band
                band_df = event_df[band_index]
                # print(band_df)
                pcs = self.pcs[band]
                if len(band_df) > 0:

                    binned_dates = self.get_binned_time(band_df)
                    if mid_point_date - self.num_prediction_days + 1 < self.prediction_start_date:
                        self.prediction_start_date = mid_point_date - self.num_prediction_days + 1
                    if mid_point_date + self.num_prediction_days - 1 > self.prediction_end_date:
                        self.prediction_end_date = mid_point_date + self.num_prediction_days - 1
                    b2 = (binned_dates - mid_point_date + self.num_prediction_days - 1) / 2
                    b2 = b2.astype(int)
                    light_curve_seg = np.zeros(self.num_prediction_days)
                    light_curve_seg[b2[:]] = band_df[self.lc.brightness_col_name]
                    # initial_guess = np.amax(band_df[self.lc.brightness_col_name]) * np.array([.93, .03, .025])
                    initial_guess = np.zeros(self.num_pc_components)
                    result = minimize(calc_loss, initial_guess, args=(pcs, light_curve_seg))
                    coeff_all_band[band] = list(result.x)
                    # print(result.x)
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
                             axes_lims=True, buffer_days=20, mark_threshold=True, linestyle="solid", fig=None):
        """
        plot of predictions of each band

        :param all_band_coeff_dict: coeff dict for all the bands
        :param color_band_dict: dict with colors corresponding to each band
        :param mark_maximum: option to mark maximum
        :param object_name: name/object type of the current object
        :param axes_lims: boolean to limit the axes on in the region of prediction
        :param buffer_days: buffer region beyond the prediction where plot is made. So if prediction in between the days
            120 - 220 and buffer_days is 5, plot is limited to days 115 - 225.
        :param mark_threshold: Mark the minimum threshold
        :param linestyle: linetyle of the plot
        :param fig: fig on which plot is to be made. If nothing is passed, new fig is created.
        :return: figure with the plots
        """
        fig = self.lc.plot_light_curve(fig=fig, color_band_dict=color_band_dict, alpha=0.3, mark_maximum=False,
                                       mark_label=False, plot_points=True)

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
                        start_date = mid_point_date - 50

                    else:
                        end_date = self.current_date
                        start_date = self.current_date - self.num_alert_days

                    if mark_maximum:
                        fig = self.lc.plot_light_curve(color_band_dict, fig=fig, start_date=self.data_start_date,
                                                       end_date=self.data_end_date, band=band, alpha=1,
                                                       mark_maximum=True,
                                                       plot_points=True)
                    else:
                        fig = self.lc.plot_light_curve(color_band_dict, fig=fig, start_date=self.data_start_date,
                                                       end_date=self.data_end_date, band=band, alpha=1,
                                                       mark_maximum=False,
                                                       plot_points=True)

                    if len(coeff) != 0:
                        predicted_lc = calc_prediction(coeff, self.pcs[band])
                        # plt.plot(x_data, predicted_lc, color = color_band_dict[band])
                        time_data = np.arange(0, 102, 2) + mid_point_date - 50
                    else:
                        predicted_lc = []
                        time_data = []

                    plt.plot(time_data, predicted_lc, color=color_band_dict[band], linestyle=linestyle)

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
            ax.axhline(y=self.min_flux_threshold, color='r', linestyle='--', label='threshold')
        plt.text(.01, .94, "ID: " + str(self.lc.object_id), fontsize=15, transform=ax.transAxes)
        if object_name is not None:
            # print(self.lc.get_object_type_for_PLAsTiCC(object_id))
            plt.text(.01, .88, "Type: " + object_name, fontsize=15, transform=ax.transAxes)

        plt.xlabel("mjd", fontsize=25)
        plt.ylabel("flux", fontsize=25)
        ax.tick_params(axis='both', which='major', labelsize=12)
        plt.legend(loc="upper right")

        return fig
